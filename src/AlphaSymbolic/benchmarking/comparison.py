"""Honest train/holdout benchmark used by the Gradio benchmark tab.

Each displayed method has a distinct implementation.  The polynomial method
is a transparent sanity baseline, not a symbolic-regression SOTA competitor.
For publishable comparisons use ``scripts/benchmark_scientific.py``.
"""

from __future__ import annotations

import gc
import math
import re
import time
from typing import Callable, Dict, List, Optional

import numpy as np


BENCHMARK_PROBLEMS = [
    {"id": "poly-x2", "name": "x²", "formula": lambda x: x**2, "level": "Easy", "range": (-5, 5)},
    {"id": "poly-x3", "name": "x³ - 2x", "formula": lambda x: x**3 - 2*x, "level": "Easy", "range": (-3, 3)},
    {"id": "poly-quad", "name": "2x² + 3x + 1", "formula": lambda x: 2*x**2 + 3*x + 1, "level": "Easy", "range": (-5, 5)},
    {"id": "nguyen-1", "name": "Nguyen-1", "formula": lambda x: x**3 + x**2 + x, "level": "Easy", "range": (-1, 1)},
    {"id": "nguyen-5", "name": "Nguyen-5", "formula": lambda x: np.sin(x**2)*np.cos(x) - 1, "level": "Medium", "range": (-1, 1)},
    {"id": "nguyen-7", "name": "Nguyen-7", "formula": lambda x: np.log(x+1) + np.log(x**2+1), "level": "Medium", "range": (0.1, 2)},
    {"id": "nguyen-8", "name": "Nguyen-8 √x", "formula": lambda x: np.sqrt(x), "level": "Easy", "range": (0, 4)},
    {"id": "trig-sin", "name": "sin(x)", "formula": lambda x: np.sin(x), "level": "Easy", "range": (-3.14, 3.14)},
    {"id": "mixed-xsinx", "name": "x·sin(x)", "formula": lambda x: x*np.sin(x), "level": "Medium", "range": (-5, 5)},
    {"id": "mixed-exp", "name": "e^(-x)·sin(x)", "formula": lambda x: np.exp(-x)*np.sin(x), "level": "Hard", "range": (0, 6)},
]

SUPPORTED_METHODS = ("gpu_gp", "polynomial")
_CARET_POWER = re.compile(r"(?<!\*)\^(?!\*)")


def _normalize_power_syntax(formula: str) -> str:
    """Convert engine-style ``^`` powers while preserving Python ``**``."""
    return _CARET_POWER.sub("**", formula)


def _generate_data(problem: dict, n_points: int = 50, seed: int = 42):
    """Generate one deterministic sample; retained for API compatibility."""
    rng = np.random.default_rng(seed)
    x_min, x_max = problem["range"]
    x = np.sort(rng.uniform(x_min, x_max, n_points))
    y = np.asarray(problem["formula"](x), dtype=np.float64)
    valid = np.isfinite(y)
    return x[valid], y[valid]


def _generate_train_test(problem: dict, seed: int, n_train=60, n_test=256):
    x_train, y_train = _generate_data(problem, n_train, seed)
    # A separate stream prevents accidental evaluation on training points.
    x_test, y_test = _generate_data(problem, n_test, seed + 1_000_003)
    return x_train, y_train, x_test, y_test


def _compute_rmse(formula_str: str, x_vals, y_vals) -> float:
    """Evaluate an engine formula with a restricted numerical environment."""
    try:
        expression = _normalize_power_syntax(formula_str)
        x_arr = np.asarray(x_vals, dtype=np.float64)
        safe = {
            "x0": x_arr, "x": x_arr, "np": np,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
            "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
            "abs": np.abs, "pi": np.pi, "e": np.e,
            "floor": np.floor, "ceil": np.ceil, "sign": np.sign,
            "gamma": np.vectorize(math.gamma),
            "lgamma": np.vectorize(math.lgamma),
            "fact": np.vectorize(lambda value: math.gamma(value + 1.0)),
        }
        with np.errstate(all="ignore"):
            y_pred = eval(expression, {"__builtins__": {}}, safe)
        y_true = np.asarray(y_vals, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if y_pred.ndim == 0:
            y_pred = np.full_like(y_true, float(y_pred))
        if y_pred.shape != y_true.shape or not np.all(np.isfinite(y_pred)):
            return float("inf")
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        return rmse if np.isfinite(rmse) else float("inf")
    except Exception:
        return float("inf")


def _run_gpu_gp(x_train, y_train, x_test, y_test, timeout, _seed):
    import torch
    from warpsymbolic.gpu import TensorGeneticEngine
    from warpsymbolic.gpu.config import GpuGlobals

    flag_values = {
        "USE_OP_PLUS": True, "USE_OP_MINUS": True, "USE_OP_MULT": True,
        "USE_OP_DIV": True, "USE_OP_POW": True, "USE_OP_MOD": False,
        "USE_OP_SIN": True, "USE_OP_COS": True, "USE_OP_TAN": False,
        "USE_OP_LOG": True, "USE_OP_EXP": True, "USE_OP_SQRT": True,
        "USE_OP_ABS": True, "USE_OP_FACT": False, "USE_OP_GAMMA": False,
        "USE_OP_LGAMMA": False,
    }
    if not GpuGlobals.GPU_EXECUTION_LOCK.acquire(blocking=False):
        raise RuntimeError("La GPU ya está ocupada por otra tarea de la aplicación.")
    previous = {name: getattr(GpuGlobals, name) for name in flag_values}
    engine = None
    try:
        for name, value in flag_values.items():
            setattr(GpuGlobals, name, value)
        engine = TensorGeneticEngine(num_variables=1, max_constants=10)
        formula = engine.run(
            x_train, y_train, seeds=[], timeout_sec=timeout, use_log=False
        )
        return (
            formula,
            _compute_rmse(formula, x_train, y_train) if formula else float("inf"),
            _compute_rmse(formula, x_test, y_test) if formula else float("inf"),
        )
    finally:
        for name, value in previous.items():
            setattr(GpuGlobals, name, value)
        GpuGlobals.GPU_EXECUTION_LOCK.release()
        if engine is not None:
            del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_polynomial(x_train, y_train, x_test, y_test, _timeout, _seed):
    """Degree-five least-squares sanity baseline."""
    degree = min(5, max(0, len(x_train) - 1))
    fitted = np.polynomial.Polynomial.fit(x_train, y_train, degree).convert()
    coefficients = np.asarray(fitted.coef, dtype=np.float64)
    terms = [
        f"({coefficient:.17g})" if power == 0
        else f"({coefficient:.17g})*x0**{power}"
        for power, coefficient in enumerate(coefficients)
        if abs(coefficient) > 1e-14
    ]
    formula = " + ".join(terms) or "0"
    train_pred = fitted(x_train)
    test_pred = fitted(x_test)
    train_rmse = float(np.sqrt(np.mean((train_pred - y_train) ** 2)))
    test_rmse = float(np.sqrt(np.mean((test_pred - y_test) ** 2)))
    return formula, train_rmse, test_rmse


METHOD_RUNNERS = {
    "gpu_gp": _run_gpu_gp,
    "polynomial": _run_polynomial,
}


def run_comparison_benchmark(
    model=None,
    device=None,
    methods: List[str] | None = None,
    gp_timeout: int = 30,
    beam_width: int = 50,
    n_problems: int = 10,
    progress_callback: Optional[Callable] = None,
    seed: int = 42,
) -> Dict:
    """Run paired implementations on independent train and holdout samples."""
    del model, device, beam_width  # Legacy API parameters.
    methods = list(methods or ["gpu_gp"])
    unsupported = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unsupported:
        raise ValueError(
            f"Métodos no implementados en este benchmark: {unsupported}. "
            f"Disponibles: {list(SUPPORTED_METHODS)}"
        )

    problems = BENCHMARK_PROBLEMS[: min(n_problems, len(BENCHMARK_PROBLEMS))]
    total_tasks = max(1, len(problems) * len(methods))
    results = []
    per_method_stats: Dict[str, dict] = {
        method: {
            "solved": 0,
            "total": len(problems),
            "valid_runs": 0,
            "failed": 0,
            "rmse_sum": 0.0,
            "time_sum": 0.0,
            "avg_rmse": float("inf"),
            "avg_time": 0.0,
            "score": 0.0,
        }
        for method in methods
    }

    task_counter = 0
    for problem_index, problem in enumerate(problems):
        x_train, y_train, x_test, y_test = _generate_train_test(
            problem, seed + problem_index * 1009
        )
        y_scale = max(float(np.std(y_test)), 1e-12)

        for method in methods:
            task_counter += 1
            if progress_callback:
                progress_callback(
                    task_counter / total_tasks,
                    desc=f"[{method.upper()}] {problem['name']}...",
                )

            started = time.perf_counter()
            error = None
            try:
                formula, train_rmse, test_rmse = METHOD_RUNNERS[method](
                    x_train, y_train, x_test, y_test, gp_timeout,
                    seed + problem_index * 1009,
                )
            except Exception as exc:
                formula = None
                train_rmse = test_rmse = float("inf")
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - started
            test_nrmse = test_rmse / y_scale if np.isfinite(test_rmse) else float("inf")
            success = test_nrmse < 0.01

            results.append({
                "problem_name": problem["name"],
                "level": problem["level"],
                "method": method,
                "formula": formula or "No solution",
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "test_nrmse": test_nrmse,
                "rmse": test_rmse,  # Backward-compatible alias.
                "time": elapsed,
                "success": success,
                "error": error,
                "seed": seed + problem_index * 1009,
            })

            stats = per_method_stats[method]
            stats["time_sum"] += elapsed
            if success:
                stats["solved"] += 1
            if np.isfinite(test_rmse):
                stats["valid_runs"] += 1
                stats["rmse_sum"] += test_rmse
            else:
                stats["failed"] += 1

    for stats in per_method_stats.values():
        valid = stats["valid_runs"]
        total = stats["total"]
        stats["avg_rmse"] = stats["rmse_sum"] / valid if valid else float("inf")
        stats["avg_time"] = stats["time_sum"] / total if total else 0.0
        stats["score"] = stats["solved"] / total * 100.0 if total else 0.0

    return {
        "results": results,
        "summary": per_method_stats,
        "protocol": {
            "seed": seed,
            "train_points": 60,
            "test_points": 256,
            "holdout": "independent_random_stream",
            "success": "test_nrmse < 0.01",
        },
    }
