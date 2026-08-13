"""Default WarpSymbolic estimator runner for the pinned SRBench harness.

The harness prepares the split and external scaling.  This module deliberately
contains only the algorithm-specific bridge so it can be hashed, replaced, and
audited independently through ``--runner module:function``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from warpsymbolic.api.sklearn import WarpSymbolicRegressor


_CONTEXT_CONTROLLED = {
    "pop_size",
    "generations",
    "max_time",
    "random_state",
}


def _effective_params(context: Any, n_train: int) -> tuple[Dict[str, Any], str]:
    custom = dict(context.runner_params)
    conflicting = sorted(_CONTEXT_CONTROLLED.intersection(custom))
    if conflicting:
        raise ValueError(
            "use the harness flags for context-controlled parameters, not "
            f"--runner-param: {', '.join(conflicting)}"
        )
    params: Dict[str, Any] = {
        "n_islands": 20,
        "max_len": 48,
        "max_constants": 10,
        "max_gpu_variables": 4,
        "max_gpu_samples": 1024,
        "feature_selection": "hybrid",
        "fallback_strategy": "linear",
        "polynomial_degree": 3,
        "max_polynomial_variables": 8,
        "ridge_alpha": 1e-6,
        "validation_fraction": 0.2,
        "device": None,
        "use_log": False,
        "search_mode": "adaptive",
        "target_transform": "auto",
        "max_active_variables": 8,
    }
    params.update(custom)
    params.update(
        {
            "pop_size": int(context.population_size),
            "generations": int(context.generations),
            "max_time": min(float(context.fit_time_limit_sec), 60.0),
            "random_state": int(context.random_state),
        }
    )
    return params, "universal_frozen"


def _candidate_summary(model: WarpSymbolicRegressor) -> list[Mapping[str, Any]]:
    result = []
    for candidate in getattr(model, "validation_candidates_", []):
        result.append(
            {
                key: candidate.get(key)
                for key in ("name", "degree", "rmse", "complexity")
            }
        )
    return result


def run(prepared: Any, context: Any) -> Mapping[str, Any]:
    """Fit WarpSymbolic and return predictions in scaled target coordinates."""

    params, budget_policy = _effective_params(context, len(prepared.y_train))
    train_frame = pd.DataFrame(
        np.asarray(prepared.X_train, dtype=np.float64),
        columns=list(prepared.feature_names),
    )
    test_frame = pd.DataFrame(
        np.asarray(prepared.X_test, dtype=np.float64),
        columns=list(prepared.feature_names),
    )
    estimator = WarpSymbolicRegressor(**params)
    started = time.perf_counter()
    estimator.fit(train_frame, np.asarray(prepared.y_train, dtype=np.float64))
    fit_elapsed = time.perf_counter() - started
    train_prediction = np.asarray(estimator.predict(train_frame), dtype=np.float64)
    test_prediction = np.asarray(estimator.predict(test_frame), dtype=np.float64)

    return {
        "train_predictions_scaled": train_prediction,
        "test_predictions_scaled": test_prediction,
        "training_time_sec": fit_elapsed,
        "model_size": int(estimator.symbolic_complexity_),
        # ``fit`` already parsed, simplified, and cached this expression.
        # Re-running ``to_sympy_string`` can spend tens of seconds simplifying
        # a 100+ term polynomial a second time.
        "symbolic_model": estimator.sympy_formula_,
        "params": params,
        "metadata": {
            "winner": estimator.fit_status_,
            "budget_policy": budget_policy,
            "selection_reason": estimator.selection_reason_,
            "selected_feature_indices": estimator.selected_feature_indices_,
            "selected_feature_names": estimator.selected_feature_names_,
            "validation_size": int(estimator.validation_size_),
            "validation_candidates": _candidate_summary(estimator),
            "polynomial_degree": estimator.polynomial_degree_,
            "engine_validation_rmse": estimator.engine_validation_rmse_,
            "polynomial_validation_rmse": estimator.polynomial_validation_rmse_,
            "fallback_validation_rmse": estimator.fallback_validation_rmse_,
            "engine_error": estimator.engine_error_,
            "gpu_training_samples": int(estimator.n_gpu_samples_),
            "configuration_sha256": getattr(estimator, "configuration_hash_", None),
            "search_report": getattr(estimator, "search_report_", None),
            "pareto_front": getattr(estimator, "pareto_front_", []),
        },
    }


__all__ = ["run"]
