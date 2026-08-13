"""Separated development and clean N-Queens evaluation suites.

The estimator always receives only numeric training X/y.  Ground-truth
functions and held-out values live in this evaluation script and are never
passed as estimator metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from AlphaSymbolic.sklearn import AlphaSymbolicRegressor


NQUEENS_COUNTS = np.asarray(
    [
        1, 1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200,
        73712, 365596, 2279184, 14772512, 95815104, 666090624,
        4968057848, 39029188884, 314666222712, 2691008701644,
        24233937684440, 227514171973736, 2207893435808352,
        22317699616364044, 234907967154122528,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class DevelopmentCase:
    key: str
    expression_id: str
    variables: int
    function: Callable[[np.ndarray], np.ndarray]
    train_bounds: tuple[tuple[float, float], ...]
    test_bounds: tuple[tuple[float, float], ...]
    noise: float = 0.0
    irrelevant: int = 0


DEVELOPMENT_CASES = (
    DevelopmentCase(
        "dev-algebraic-cubic",
        "sha-dev-x3-2x-7",
        1,
        lambda x: x[:, 0] ** 3 - 2.0 * x[:, 0] + 7.0,
        ((-2.0, 2.0),),
        ((-3.0, 3.0),),
    ),
    DevelopmentCase(
        "dev-rational-two-var",
        "sha-dev-rational-offset",
        2,
        lambda x: (x[:, 0] + 0.5) / (1.0 + x[:, 1] ** 2),
        ((-2.0, 2.0), (-1.5, 1.5)),
        ((-3.0, 3.0), (-2.0, 2.0)),
    ),
    DevelopmentCase(
        "dev-periodic-mixed",
        "sha-dev-sin17-cos03",
        2,
        lambda x: 1.7 * np.sin(x[:, 0]) + 0.3 * np.cos(2.0 * x[:, 1]),
        ((-3.0, 3.0), (-3.0, 3.0)),
        ((-5.0, 5.0), (-5.0, 5.0)),
    ),
    DevelopmentCase(
        "dev-exponential-positive",
        "sha-dev-exp037-plus2",
        1,
        lambda x: np.exp(0.37 * x[:, 0]) + 2.0,
        ((-2.0, 3.0),),
        ((-3.0, 5.0),),
    ),
    DevelopmentCase(
        "dev-pure-interaction",
        "sha-dev-product-no-marginal",
        2,
        lambda x: x[:, 0] * x[:, 1],
        ((-2.0, 2.0), (-2.0, 2.0)),
        ((-3.0, 3.0), (-3.0, 3.0)),
        irrelevant=6,
    ),
    DevelopmentCase(
        "dev-noisy-smooth",
        "sha-dev-quadratic-noise",
        2,
        lambda x: 0.8 * x[:, 0] ** 2 - 1.2 * x[:, 1] + 0.4,
        ((-2.0, 2.0), (-2.0, 2.0)),
        ((-2.5, 2.5), (-2.5, 2.5)),
        noise=0.02,
        irrelevant=4,
    ),
)


def _sample_case(case: DevelopmentCase, seed: int, n_train: int, n_test: int):
    train_rng = np.random.default_rng(seed)
    test_rng = np.random.default_rng(seed + 1_000_003)

    def sample(rng: np.random.Generator, size: int, bounds):
        signal = np.column_stack([rng.uniform(low, high, size) for low, high in bounds])
        if case.irrelevant:
            noise_columns = rng.normal(size=(size, case.irrelevant))
            signal = np.column_stack((signal, noise_columns))
        return signal

    X_train = sample(train_rng, n_train, case.train_bounds)
    X_test = sample(test_rng, n_test, case.test_bounds)
    y_train = case.function(X_train[:, : case.variables])
    y_test = case.function(X_test[:, : case.variables])
    if case.noise:
        y_train = y_train + train_rng.normal(
            scale=case.noise * max(float(np.std(y_train)), 1e-12), size=n_train
        )
    return X_train, y_train, X_test, y_test


def _estimator(mode: str, seed: int, max_time: float) -> AlphaSymbolicRegressor:
    return AlphaSymbolicRegressor(
        pop_size=50_000,
        search_mode=mode,
        target_transform="auto",
        max_time=min(float(max_time), 60.0),
        random_state=seed,
        max_active_variables=8,
    )


def _development_row(
    case: DevelopmentCase,
    mode: str,
    seed: int,
    max_time: float,
    n_train: int,
    n_test: int,
) -> dict:
    X_train, y_train, X_test, y_test = _sample_case(case, seed, n_train, n_test)
    model = _estimator(mode, seed, max_time)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - started
    prediction = model.predict(X_test)
    relative_scale = np.maximum(np.abs(y_test), max(float(np.std(y_test)), 1e-12))
    relative_error = np.abs(prediction - y_test) / relative_scale
    return {
        "suite": "development",
        "case": case.key,
        "expression_id": case.expression_id,
        "mode": mode,
        "seed": seed,
        "r2": float(r2_score(y_test, prediction)),
        "rmse": float(np.sqrt(np.mean((prediction - y_test) ** 2))),
        "median_relative_error": float(np.median(relative_error)),
        "recovered_numeric": bool(np.max(relative_error) < 1e-6 and case.noise == 0.0),
        "complexity": int(model.symbolic_complexity_),
        "time_sec": elapsed,
        "formula": model.sympy_formula_,
        "configuration_sha256": getattr(model, "configuration_hash_", None),
    }


def _log_polynomial_baseline(degree: int, n_train: np.ndarray, y_train: np.ndarray, n_test: np.ndarray):
    expansion = PolynomialFeatures(degree=degree, include_bias=False)
    design = expansion.fit_transform(n_train.reshape(-1, 1))
    model = Ridge(alpha=1e-9).fit(design, np.log(y_train))
    return np.exp(model.predict(expansion.transform(n_test.reshape(-1, 1))))


def _nqueens_rows(mode: str, seeds: int, max_time: float) -> list[dict]:
    n_train = np.arange(8, 25, dtype=np.float64)
    n_test = np.arange(25, 28, dtype=np.float64)
    X_train = n_train.reshape(-1, 1)
    X_test = n_test.reshape(-1, 1)
    y_train = NQUEENS_COUNTS[n_train.astype(int)]
    y_test = NQUEENS_COUNTS[n_test.astype(int)]
    rows = []
    for seed in range(seeds):
        model = _estimator(mode, seed, max_time)
        started = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - started
        prediction = model.predict(X_test)
        rows.append(
            {
                "suite": "nqueens-clean",
                "mode": mode,
                "seed": seed,
                "input_columns": 1,
                "train_n": [8, 24],
                "test_n": [25, 27],
                "log_rmse": float(
                    np.sqrt(np.mean((np.log(np.maximum(prediction, 1.0)) - np.log(y_test)) ** 2))
                ),
                "mean_relative_error": float(np.mean(np.abs(prediction - y_test) / y_test)),
                "rounded_count_accuracy": float(np.mean(np.rint(prediction) == y_test)),
                "complexity": int(model.symbolic_complexity_),
                "time_sec": elapsed,
                "formula": model.sympy_formula_,
                "predictions": [float(value) for value in prediction],
                "configuration_sha256": getattr(model, "configuration_hash_", None),
            }
        )
    model_predictions = np.asarray(
        [row["predictions"] for row in rows if row.get("mode") == mode],
        dtype=np.float64,
    )
    rows.append(
        {
            "suite": "nqueens-clean",
            "mode": f"{mode}-summary",
            "seed": None,
            "input_columns": 1,
            "seeds": seeds,
            "log_rmse_std": float(
                np.std([row["log_rmse"] for row in rows if row.get("mode") == mode])
            ),
            "mean_log_prediction_std": float(
                np.mean(np.std(np.log(np.maximum(model_predictions, 1.0)), axis=0))
            ),
            "unique_formulas": len(
                {row["formula"] for row in rows if row.get("mode") == mode}
            ),
        }
    )
    for degree in (1, 2, 3):
        prediction = _log_polynomial_baseline(degree, n_train, y_train, n_test)
        rows.append(
            {
                "suite": "nqueens-clean",
                "mode": f"trainable-log-polynomial-{degree}",
                "seed": None,
                "input_columns": 1,
                "log_rmse": float(np.sqrt(np.mean((np.log(prediction) - np.log(y_test)) ** 2))),
                "mean_relative_error": float(np.mean(np.abs(prediction - y_test) / y_test)),
                "rounded_count_accuracy": float(np.mean(np.rint(prediction) == y_test)),
            }
        )
    return rows


def _suite_hash() -> str:
    payload = [
        {
            "key": case.key,
            "expression_id": case.expression_id,
            "train_bounds": case.train_bounds,
            "test_bounds": case.test_bounds,
            "noise": case.noise,
            "irrelevant": case.irrelevant,
        }
        for case in DEVELOPMENT_CASES
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("development", "nqueens"), default="development")
    parser.add_argument("--mode", choices=("adaptive", "legacy", "both"), default="adaptive")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-time", type=float, default=60.0)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    modes = ("adaptive", "legacy") if args.mode == "both" else (args.mode,)
    rows = []
    if args.suite == "development":
        for mode in modes:
            for case in DEVELOPMENT_CASES:
                for seed in range(int(args.seeds)):
                    rows.append(
                        _development_row(
                            case,
                            mode,
                            seed,
                            args.max_time,
                            args.train_samples,
                            args.test_samples,
                        )
                    )
        suite_hash = _suite_hash()
    else:
        for mode in modes:
            rows.extend(_nqueens_rows(mode, int(args.seeds), args.max_time))
        suite_hash = hashlib.sha256(b"nqueens-clean-n8-24-test25-27-x-only-v1").hexdigest()
    for row in rows:
        row["suite_sha256"] = suite_hash
        print(json.dumps(row, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
