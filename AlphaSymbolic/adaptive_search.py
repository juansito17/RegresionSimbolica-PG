"""Training-data-only adaptive search for :class:`AlphaSymbolicRegressor`.

This module deliberately has no dataset registry and receives no metadata other
than numeric X/y.  Column labels are restored by the public estimator only
after model selection.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
import time
import warnings
from typing import Any, Callable, Iterable

import numpy as np
from sklearn.utils import check_random_state

from .sklearn_estimator import (
    _AdaptiveCandidate,
    _format_number,
    _formula_with_feature_names,
    _linear_fallback_formula,
    _load_engine_class,
    _normalise_formula,
    _parse_formula,
    _release_cuda_memory,
    _seeded_engine_runtime,
    _sparse_polynomial_formula,
    evaluate_formula,
)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if not np.isfinite(y_pred).all():
        return math.inf
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _target_scale(y: np.ndarray) -> float:
    scale = float(np.std(y))
    return scale if math.isfinite(scale) and scale > 1e-15 else 1.0


def _wrap_target_formula(formula: str, transform: str) -> str:
    return f"exp({formula})" if transform == "log" else formula


def _transform_target(y: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log":
        if np.any(y <= 0.0):
            raise ValueError("log target transform requires strictly positive y")
        return np.log(y)
    return np.asarray(y, dtype=np.float64)


def _iid_folds(n_samples: int, seed: int | None) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(n_samples, dtype=np.int64)
    if n_samples <= 2:
        return [(indices, indices)]
    rng = check_random_state(seed)
    shuffled = rng.permutation(indices)
    if n_samples < 40:
        return [
            (indices[indices != held_out], np.asarray([held_out], dtype=np.int64))
            for held_out in shuffled
        ]
    n_folds = 5 if n_samples < 500 else 3
    validation_parts = np.array_split(shuffled, n_folds)
    result = []
    for validation in validation_parts:
        mask = np.ones(n_samples, dtype=bool)
        mask[validation] = False
        result.append((indices[mask], np.sort(validation).astype(np.int64)))
    return result


def _boundary_fold(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = X.shape[0]
    indices = np.arange(n_samples, dtype=np.int64)
    if n_samples < 6:
        return indices, indices
    centered_x = X - np.mean(X, axis=0)
    centered_y = y - np.mean(y)
    denominator = np.linalg.norm(centered_x, axis=0) * np.linalg.norm(centered_y)
    correlation = np.divide(
        np.abs(centered_x.T @ centered_y),
        denominator,
        out=np.zeros(X.shape[1], dtype=np.float64),
        where=denominator > 1e-15,
    )
    axis = int(np.argmax(correlation))
    ordered = np.argsort(X[:, axis], kind="mergesort")
    edge = max(1, int(round(0.1 * n_samples)))
    validation = np.unique(np.concatenate((ordered[:edge], ordered[-edge:])))
    mask = np.ones(n_samples, dtype=bool)
    mask[validation] = False
    training = indices[mask]
    if training.size < 2:
        return indices, indices
    return training, np.sort(validation).astype(np.int64)


def _feature_subsets(estimator: Any, X: np.ndarray, y: np.ndarray, seed: int | None):
    limit = min(X.shape[1], int(estimator.max_active_variables))
    primary = estimator._select_features(X, y, seed, limit)
    subsets = [primary]

    # Pure interactions can have zero marginal correlation.  Screen products
    # on a bounded pool and force the best pair into a candidate subset.
    pool = estimator._select_features(X, y, seed, min(X.shape[1], 32))
    if pool.size >= 2:
        centered_y = y - np.mean(y)
        y_norm = np.linalg.norm(centered_y)
        best_pair = None
        best_score = -1.0
        for left_position, left in enumerate(pool[:-1]):
            for right in pool[left_position + 1 :]:
                product = X[:, left] * X[:, right]
                centered = product - np.mean(product)
                denominator = np.linalg.norm(centered) * y_norm
                score = (
                    abs(float(centered @ centered_y)) / denominator
                    if denominator > 1e-15
                    else 0.0
                )
                if score > best_score:
                    best_score = score
                    best_pair = (int(left), int(right))
        if best_pair is not None:
            interaction = list(best_pair)
            interaction.extend(
                int(index) for index in primary if int(index) not in interaction
            )
            subsets.append(np.sort(np.asarray(interaction[:limit], dtype=np.int64)))

    # Mandatory random exploration is deterministic for a fixed seed.
    if X.shape[1] > limit:
        random_subset = np.sort(
            check_random_state(seed).choice(X.shape[1], size=limit, replace=False)
        ).astype(np.int64)
        subsets.append(random_subset)

    unique = []
    seen = set()
    for subset in subsets:
        key = tuple(int(value) for value in subset)
        if key not in seen:
            seen.add(key)
            unique.append(np.asarray(subset, dtype=np.int64))
    return unique


def _parametric_candidate(
    *,
    name: str,
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    folds: Iterable[tuple[np.ndarray, np.ndarray]],
    boundary: tuple[np.ndarray, np.ndarray],
    transform: str,
    fitter: Callable[[np.ndarray, np.ndarray], str],
    degree: int | None = None,
) -> _AdaptiveCandidate:
    started = time.perf_counter()
    transformed = _transform_target(y, transform)
    scale = _target_scale(y)
    iid_scores: list[float] = []
    for training, validation in folds:
        fold_formula = _wrap_target_formula(
            fitter(X[training][:, indices], transformed[training]), transform
        )
        prediction = evaluate_formula(fold_formula, X[validation][:, indices])
        iid_scores.append(_rmse(y[validation], prediction) / scale)
    boundary_training, boundary_validation = boundary
    boundary_formula = _wrap_target_formula(
        fitter(X[boundary_training][:, indices], transformed[boundary_training]),
        transform,
    )
    boundary_prediction = evaluate_formula(
        boundary_formula, X[boundary_validation][:, indices]
    )
    boundary_score = _rmse(y[boundary_validation], boundary_prediction) / scale
    full_formula = _wrap_target_formula(fitter(X[:, indices], transformed), transform)
    complexity = sum(1 for _ in ast.walk(_parse_formula(full_formula)))
    iid_mean = float(np.mean(iid_scores))
    iid_se = (
        float(np.std(iid_scores, ddof=1) / math.sqrt(len(iid_scores)))
        if len(iid_scores) > 1
        else 0.0
    )
    score = 0.75 * iid_mean + 0.25 * boundary_score
    mse_ratio = max((score * scale) ** 2 / max(scale**2, 1e-30), 1e-30)
    mdl = X.shape[0] * math.log(mse_ratio) + complexity * math.log(
        max(X.shape[0], 2)
    )
    return _AdaptiveCandidate(
        name=name,
        family=family,
        formula=full_formula,
        indices=indices,
        iid_scores=iid_scores,
        boundary_score=boundary_score,
        score=score,
        score_se=iid_se,
        mdl=mdl,
        complexity=complexity,
        elapsed_sec=time.perf_counter() - started,
        degree=degree,
    )


def _grammar_arms() -> dict[str, dict[str, Any]]:
    disabled = {
        "USE_OP_MOD": False,
        "USE_OP_SIN": False,
        "USE_OP_COS": False,
        "USE_OP_TAN": False,
        "USE_OP_LOG": False,
        "USE_OP_EXP": False,
        "USE_OP_FACT": False,
        "USE_OP_FLOOR": False,
        "USE_OP_GAMMA": False,
        "USE_OP_LGAMMA": False,
        "USE_OP_ASIN": False,
        "USE_OP_ACOS": False,
        "USE_OP_ATAN": False,
        "USE_OP_CEIL": False,
        "USE_OP_SIGN": False,
        "USE_OP_SQRT": True,
        "USE_OP_ABS": True,
    }
    common = {
        **disabled,
        "USE_STRUCTURAL_SEEDS": False,
        "USE_PATTERN_SEEDS": False,
        "USE_SNIPER": False,
        "USE_PATTERN_MEMORY": False,
        "USE_PARETO_SELECTION": True,
        "USE_LEXICASE_SELECTION": True,
    }
    return {
        "algebraic": {**common},
        "periodic": {**common, "USE_OP_SIN": True, "USE_OP_COS": True},
        "transcendental": {
            **common,
            "USE_OP_LOG": True,
            "USE_OP_EXP": True,
        },
        "combinatorial": {
            **common,
            "USE_OP_LOG": True,
            "USE_OP_EXP": True,
            "USE_OP_MOD": True,
            "USE_OP_FACT": True,
            "USE_OP_GAMMA": True,
            "USE_OP_LGAMMA": True,
        },
    }


def _calibrated_structure_candidate(
    *,
    name: str,
    family: str,
    raw_formula: str,
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    boundary: tuple[np.ndarray, np.ndarray],
    transform: str,
    elapsed_sec: float,
) -> _AdaptiveCandidate:
    transformed = _transform_target(y, transform)
    raw = evaluate_formula(raw_formula, X[:, indices])
    if not np.isfinite(raw).all():
        raise ValueError("engine structure is non-finite on full training data")

    def coefficients(training: np.ndarray) -> tuple[float, float]:
        design = np.column_stack((raw[training], np.ones(training.size)))
        result = np.linalg.lstsq(design, transformed[training], rcond=None)[0]
        return float(result[0]), float(result[1])

    def predicted(training: np.ndarray, validation: np.ndarray) -> np.ndarray:
        coefficient, intercept = coefficients(training)
        values = coefficient * raw[validation] + intercept
        return np.exp(np.clip(values, -745.0, 709.0)) if transform == "log" else values

    scale = _target_scale(y)
    iid_scores = [
        _rmse(y[validation], predicted(training, validation)) / scale
        for training, validation in folds
    ]
    boundary_training, boundary_validation = boundary
    boundary_score = (
        _rmse(
            y[boundary_validation],
            predicted(boundary_training, boundary_validation),
        )
        / scale
    )
    full_indices = np.arange(X.shape[0], dtype=np.int64)
    coefficient, intercept = coefficients(full_indices)
    calibrated = (
        f"({_format_number(coefficient)})*({raw_formula})"
        f"+({_format_number(intercept)})"
    )
    full_formula = _wrap_target_formula(calibrated, transform)
    complexity = sum(1 for _ in ast.walk(_parse_formula(full_formula)))
    iid_mean = float(np.mean(iid_scores))
    iid_se = (
        float(np.std(iid_scores, ddof=1) / math.sqrt(len(iid_scores)))
        if len(iid_scores) > 1
        else 0.0
    )
    score = 0.75 * iid_mean + 0.25 * boundary_score
    mdl = X.shape[0] * math.log(max(score**2, 1e-30)) + complexity * math.log(
        max(X.shape[0], 2)
    )
    return _AdaptiveCandidate(
        name=name,
        family=family,
        formula=full_formula,
        indices=indices,
        iid_scores=iid_scores,
        boundary_score=boundary_score,
        score=score,
        score_se=iid_se,
        mdl=mdl,
        complexity=complexity,
        elapsed_sec=elapsed_sec,
    )


def _pareto_front(candidates: list[_AdaptiveCandidate]) -> list[_AdaptiveCandidate]:
    front = []
    for candidate in candidates:
        objectives = (candidate.score, candidate.mdl, candidate.complexity, candidate.elapsed_sec)
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            other_objectives = (other.score, other.mdl, other.complexity, other.elapsed_sec)
            if all(a <= b for a, b in zip(other_objectives, objectives)) and any(
                a < b for a, b in zip(other_objectives, objectives)
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda item: (item.score, item.mdl, item.complexity))


def fit_adaptive(estimator: Any, X: Any, y: Any):
    estimator._validate_parameters()
    started = time.perf_counter()
    deadline = started + min(float(estimator.max_time), 60.0)
    values, target, columns, feature_names, is_frame = estimator._coerce_fit_data(X, y)
    values, fill_values = estimator._fit_imputation(values)
    seed = estimator._seed()
    folds = _iid_folds(values.shape[0], seed)
    boundary = _boundary_fold(values, target)
    subsets = _feature_subsets(estimator, values, target, seed)
    transforms = [estimator.target_transform]
    if estimator.target_transform == "auto":
        transforms = ["identity"]
        if np.all(target > 0.0):
            transforms.append("log")
    if bool(estimator.use_log):
        warnings.warn(
            "use_log is deprecated; use target_transform='log'",
            DeprecationWarning,
            stacklevel=2,
        )
        transforms = ["log"]

    stage_times: dict[str, float] = {"profile_sec": time.perf_counter() - started}
    candidates: list[_AdaptiveCandidate] = []
    failures: list[dict[str, str]] = []
    attempts = 0

    def try_candidate(**kwargs: Any) -> None:
        nonlocal attempts
        attempts += 1
        try:
            candidate = _parametric_candidate(**kwargs)
            if math.isfinite(candidate.score):
                candidates.append(candidate)
        except Exception as exc:
            failures.append({"candidate": str(kwargs.get("name")), "error": f"{type(exc).__name__}: {exc}"})

    explorer_started = time.perf_counter()
    for transform in transforms:
        for subset_number, subset in enumerate(subsets):
            suffix = f"{transform}_s{subset_number}"
            try_candidate(
                name=f"mean_{suffix}",
                family="constant",
                X=values,
                y=target,
                indices=subset,
                folds=folds,
                boundary=boundary,
                transform=transform,
                fitter=lambda data, response: _linear_fallback_formula(data, response, "mean"),
            )
            try_candidate(
                name=f"linear_{suffix}",
                family="algebraic",
                X=values,
                y=target,
                indices=subset,
                folds=folds,
                boundary=boundary,
                transform=transform,
                fitter=lambda data, response: _linear_fallback_formula(data, response, "linear"),
                degree=1,
            )
            if time.perf_counter() >= deadline:
                break
            max_degree = min(int(estimator.polynomial_degree), 3)
            for degree in range(2, max_degree + 1):
                if degree == 3 and subset.size > 4:
                    continue
                try_candidate(
                    name=f"polynomial_d{degree}_{suffix}",
                    family="algebraic",
                    X=values,
                    y=target,
                    indices=subset,
                    folds=folds,
                    boundary=boundary,
                    transform=transform,
                    fitter=lambda data, response, degree=degree: _sparse_polynomial_formula(
                        data,
                        response,
                        degree,
                        float(estimator.ridge_alpha),
                        int(estimator.sparse_polynomial_terms),
                    ),
                    degree=degree,
                )
            if time.perf_counter() >= deadline:
                break
        if time.perf_counter() >= deadline:
            break
    stage_times["explorers_sec"] = time.perf_counter() - explorer_started
    if not candidates:
        raise RuntimeError("adaptive search could not construct a finite baseline")

    best_baseline = min(candidates, key=lambda item: item.score)
    chosen_transform = "log" if best_baseline.formula.startswith("exp(") else "identity"
    engine_subset = best_baseline.indices
    arms = _grammar_arms()
    posterior = {name: [1.0, 1.0] for name in arms}
    allocations: list[dict[str, Any]] = []
    engine_started = time.perf_counter()
    remaining = deadline - time.perf_counter()
    exact_simple = best_baseline.score <= 1e-10 and best_baseline.complexity <= 24
    engine_is_warm = "AlphaSymbolic.core.gpu.engine" in sys.modules
    if remaining > 0.1 and not exact_simple and (engine_is_warm or remaining >= 20.0):
        arm_count = 4 if remaining >= 16 else 3 if remaining >= 8 else 2 if remaining >= 4 else 1
        rng = check_random_state(seed)
        ordered_arms = sorted(
            arms,
            key=lambda name: (-rng.beta(*posterior[name]), name),
        )[:arm_count]
        for arm_number, arm_name in enumerate(ordered_arms):
            remaining = deadline - time.perf_counter()
            arms_left = len(ordered_arms) - arm_number
            if remaining <= 0.1:
                break
            timeout = max(0.05, min(5.0, remaining * 0.45 / max(arms_left, 1)))
            search_training, _ = folds[0]
            coreset = search_training
            if coreset.size > 1024:
                coreset = np.sort(
                    check_random_state((seed or 0) + arm_number + 1).choice(
                        coreset, size=1024, replace=False
                    )
                )
            transformed = _transform_target(target, chosen_transform)
            engine = None
            arm_started = time.perf_counter()
            attempts += 1
            try:
                with _seeded_engine_runtime(
                    None if seed is None else seed + arm_number,
                    estimator.generations,
                    arms[arm_name],
                ):
                    engine_class = _load_engine_class()
                    engine_device = estimator.device
                    if isinstance(engine_device, str):
                        import torch

                        engine_device = torch.device(engine_device)
                    engine = engine_class(
                        device=engine_device,
                        pop_size=int(estimator.pop_size),
                        max_len=int(estimator.max_len),
                        num_variables=int(engine_subset.size),
                        max_constants=int(estimator.max_constants),
                        n_islands=int(estimator.n_islands),
                    )
                    engine_values = values[coreset][:, engine_subset]
                    raw_formula = _normalise_formula(
                        engine.run(
                            engine_values[:, 0] if engine_subset.size == 1 else engine_values,
                            transformed[coreset],
                            seeds=[],
                            timeout_sec=timeout,
                            use_log=False,
                        )
                    )
                candidate = _calibrated_structure_candidate(
                    name=f"engine_{arm_name}",
                    family=arm_name,
                    raw_formula=raw_formula,
                    X=values,
                    y=target,
                    indices=engine_subset,
                    folds=folds,
                    boundary=boundary,
                    transform=chosen_transform,
                    elapsed_sec=time.perf_counter() - arm_started,
                )
                candidates.append(candidate)
                improvement = candidate.score < best_baseline.score
                posterior[arm_name][0 if improvement else 1] += 1.0
                allocations.append(
                    {"arm": arm_name, "timeout_sec": timeout, "score": candidate.score, "improved": improvement}
                )
            except Exception as exc:
                posterior[arm_name][1] += 1.0
                failures.append({"candidate": f"engine_{arm_name}", "error": f"{type(exc).__name__}: {exc}"})
                allocations.append({"arm": arm_name, "timeout_sec": timeout, "error": f"{type(exc).__name__}: {exc}"})
            finally:
                engine = None
                _release_cuda_memory()

        # Successive halving: only the best validated grammar receives the
        # remaining deep-search budget.  This is a fresh run on a rotated
        # coreset; it does not ensemble predictions with any scout.
        successful = [item for item in candidates if item.name.startswith("engine_")]
        remaining = deadline - time.perf_counter()
        if successful and remaining > 2.0:
            best_scout = min(successful, key=lambda item: item.score)
            arm_name = best_scout.family
            deep_started = time.perf_counter()
            deep_timeout = max(0.05, remaining - 0.1)
            search_training, _ = folds[-1]
            coreset = search_training
            if coreset.size > 1024:
                coreset = np.sort(
                    check_random_state((seed or 0) + 10_003).choice(
                        coreset, size=1024, replace=False
                    )
                )
            transformed = _transform_target(target, chosen_transform)
            engine = None
            attempts += 1
            try:
                with _seeded_engine_runtime(
                    None if seed is None else seed + 10_003,
                    estimator.generations,
                    arms[arm_name],
                ):
                    engine_class = _load_engine_class()
                    engine_device = estimator.device
                    if isinstance(engine_device, str):
                        import torch

                        engine_device = torch.device(engine_device)
                    engine = engine_class(
                        device=engine_device,
                        pop_size=int(estimator.pop_size),
                        max_len=int(estimator.max_len),
                        num_variables=int(engine_subset.size),
                        max_constants=int(estimator.max_constants),
                        n_islands=int(estimator.n_islands),
                    )
                    engine_values = values[coreset][:, engine_subset]
                    raw_formula = _normalise_formula(
                        engine.run(
                            engine_values[:, 0] if engine_subset.size == 1 else engine_values,
                            transformed[coreset],
                            seeds=[],
                            timeout_sec=deep_timeout,
                            use_log=False,
                        )
                    )
                deep_candidate = _calibrated_structure_candidate(
                    name=f"engine_{arm_name}_deep",
                    family=arm_name,
                    raw_formula=raw_formula,
                    X=values,
                    y=target,
                    indices=engine_subset,
                    folds=folds,
                    boundary=boundary,
                    transform=chosen_transform,
                    elapsed_sec=time.perf_counter() - deep_started,
                )
                candidates.append(deep_candidate)
                improvement = deep_candidate.score < best_scout.score
                posterior[arm_name][0 if improvement else 1] += 1.0
                allocations.append(
                    {
                        "arm": arm_name,
                        "stage": "deep",
                        "timeout_sec": deep_timeout,
                        "score": deep_candidate.score,
                        "improved": improvement,
                    }
                )
            except Exception as exc:
                posterior[arm_name][1] += 1.0
                failures.append(
                    {
                        "candidate": f"engine_{arm_name}_deep",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            finally:
                engine = None
                _release_cuda_memory()
    stage_times["evolution_sec"] = time.perf_counter() - engine_started

    best_score_candidate = min(candidates, key=lambda item: item.score)
    threshold = best_score_candidate.score + best_score_candidate.score_se
    statistically_equivalent = [item for item in candidates if item.score <= threshold]
    winner = min(
        statistically_equivalent,
        key=lambda item: (item.mdl, item.complexity, item.elapsed_sec, item.name),
    )
    pareto = _pareto_front(candidates)

    fallback_indices = subsets[0]
    fallback_formula = _linear_fallback_formula(
        values[:, fallback_indices], target, estimator.fallback_strategy
    )
    estimator.n_features_in_ = values.shape[1]
    estimator.feature_names_in_ = np.asarray(feature_names, dtype=object)
    estimator.feature_fill_values_ = fill_values
    estimator._input_columns_ = columns
    estimator._fitted_with_dataframe_ = is_frame
    estimator.n_samples_seen_ = values.shape[0]
    estimator.validation_size_ = int(sum(len(validation) for _, validation in folds))
    estimator.random_seed_ = seed
    estimator.fallback_feature_indices_ = fallback_indices
    estimator.fallback_formula_ = fallback_formula
    estimator.formula_ = winner.formula
    estimator.formula = winner.formula
    estimator.sympy_formula_ = _formula_with_feature_names(
        winner.formula, [feature_names[index] for index in winner.indices]
    )
    estimator.selected_feature_indices_ = winner.indices
    estimator.selected_feature_names_ = np.asarray(
        [feature_names[index] for index in winner.indices], dtype=object
    )
    estimator.engine_feature_indices_ = engine_subset
    estimator.polynomial_feature_indices_ = fallback_indices
    estimator.fit_status_ = winner.name
    estimator.selection_reason_ = "one_standard_error_then_mdl_size_time"
    estimator.symbolic_complexity_ = winner.complexity
    estimator.complexity_ = winner.complexity
    estimator.mdl_ = winner.mdl
    estimator.fold_scores_ = list(winner.iid_scores)
    estimator.validation_candidates_ = [item.public() for item in candidates]
    estimator.pareto_front_ = [item.public() for item in pareto]
    estimator.candidate_formula_ = next(
        (item.formula for item in candidates if item.name.startswith("engine_")), None
    )
    estimator.polynomial_formula_ = next(
        (item.formula for item in candidates if item.name.startswith("polynomial")), None
    )
    estimator.polynomial_degree_ = winner.degree if winner.family == "algebraic" else None
    estimator.engine_validation_rmse_ = min(
        (item.score for item in candidates if item.name.startswith("engine_")),
        default=math.inf,
    )
    estimator.polynomial_validation_rmse_ = min(
        (item.score for item in candidates if item.name.startswith("polynomial")),
        default=math.inf,
    )
    estimator.fallback_validation_rmse_ = min(
        (item.score for item in candidates if item.name.startswith("linear_")),
        default=math.inf,
    )
    estimator.engine_error_ = next(
        (failure["error"] for failure in failures if failure["candidate"].startswith("engine_")),
        None,
    )
    estimator.n_gpu_samples_ = int(min(values.shape[0], 1024))
    elapsed = time.perf_counter() - started
    estimator.search_elapsed_sec_ = elapsed
    estimator.search_budget_sec_ = min(float(estimator.max_time), 60.0)
    estimator.energy_joules_ = None
    configuration = estimator.get_params(deep=False)
    configuration.pop("random_state", None)
    configuration["random_state_policy"] = "external_repetition_seed"
    configuration["max_time"] = float(configuration["max_time"])
    configuration_hash = hashlib.sha256(
        json.dumps(
            configuration,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    estimator.configuration_hash_ = configuration_hash
    estimator.search_report_ = {
        "protocol": "adaptive-v1",
        "configuration_sha256": configuration_hash,
        "budget_sec": estimator.search_budget_sec_,
        "elapsed_sec": elapsed,
        "budget_exhausted": elapsed >= estimator.search_budget_sec_,
        "cv": "leave-one-out" if values.shape[0] < 40 else "5-fold" if values.shape[0] < 500 else "3-fold",
        "score_weights": {"iid": 0.75, "boundary": 0.25},
        "structure_validation": "shared_structure_with_foldwise_constant_refit",
        "target_transforms_considered": transforms,
        "feature_subsets": [[int(value) for value in subset] for subset in subsets],
        "bandit": {"posterior": posterior, "allocations": allocations},
        "engine_cold_start_skipped_for_budget": bool(
            not engine_is_warm
            and estimator.search_budget_sec_ < 20.0
            and not exact_simple
        ),
        "stages": stage_times,
        "invalid_fraction": len(failures) / max(attempts, 1),
        "failures": failures,
        "winner": winner.public(),
        "single_expression": True,
        "energy_joules": None,
    }
    return estimator


__all__ = ["fit_adaptive"]
