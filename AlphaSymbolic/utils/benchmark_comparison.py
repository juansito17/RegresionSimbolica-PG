"""
benchmark_comparison.py — AlphaSymbolic Benchmark Utility
Provides run_comparison_benchmark for the Gradio UI benchmark tab.
Uses the GPU TensorGeneticEngine to solve standard symbolic regression problems.
"""

import time
import numpy as np
from typing import List, Dict, Optional, Callable


# ─────────────────────────────────────────────────────────────────
# Standard benchmark problems
# ─────────────────────────────────────────────────────────────────

BENCHMARK_PROBLEMS = [
    {"id": "poly-x2",   "name": "x²",           "formula": lambda x: x**2,              "level": "Easy",   "range": (-5, 5)},
    {"id": "poly-x3",   "name": "x³ - 2x",       "formula": lambda x: x**3 - 2*x,        "level": "Easy",   "range": (-3, 3)},
    {"id": "poly-quad", "name": "2x² + 3x + 1",  "formula": lambda x: 2*x**2 + 3*x + 1, "level": "Easy",   "range": (-5, 5)},
    {"id": "nguyen-1",  "name": "Nguyen-1",       "formula": lambda x: x**3 + x**2 + x,  "level": "Easy",   "range": (-1, 1)},
    {"id": "nguyen-5",  "name": "Nguyen-5",       "formula": lambda x: np.sin(x**2)*np.cos(x) - 1, "level": "Medium", "range": (-1, 1)},
    {"id": "nguyen-7",  "name": "Nguyen-7",       "formula": lambda x: np.log(x+1) + np.log(x**2+1), "level": "Medium", "range": (0.1, 2)},
    {"id": "nguyen-8",  "name": "Nguyen-8 √x",    "formula": lambda x: np.sqrt(x),        "level": "Easy",   "range": (0, 4)},
    {"id": "trig-sin",  "name": "sin(x)",         "formula": lambda x: np.sin(x),         "level": "Easy",   "range": (-3.14, 3.14)},
    {"id": "mixed-xsinx","name": "x·sin(x)",      "formula": lambda x: x*np.sin(x),       "level": "Medium", "range": (-5, 5)},
    {"id": "mixed-exp", "name": "e^(-x)·sin(x)",  "formula": lambda x: np.exp(-x)*np.sin(x), "level": "Hard", "range": (0, 6)},
]


def _generate_data(problem: dict, n_points: int = 50, seed: int = 42):
    """Generate x/y data for a benchmark problem."""
    rng = np.random.RandomState(seed)
    x_min, x_max = problem["range"]
    x = np.sort(rng.uniform(x_min, x_max, n_points))
    y = problem["formula"](x)
    # Filter out invalid values
    valid = np.isfinite(y)
    return x[valid].tolist(), y[valid].tolist()


def _compute_rmse(formula_str: str, x_vals: list, y_vals: list) -> float:
    """Evaluate a formula string and compute RMSE."""
    try:
        import math
        x_arr = np.array(x_vals)
        safe = {
            "x0": x_arr, "x": x_arr, "np": np,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
            "abs": np.abs, "pi": np.pi, "e": np.e,
            "log2": np.log2, "log10": np.log10,
            "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
            "floor": np.floor, "ceil": np.ceil, "sign": np.sign,
            "gamma": lambda z: np.array([math.gamma(v) for v in np.atleast_1d(z)]),
            "lgamma": lambda z: np.array([math.lgamma(abs(v)) for v in np.atleast_1d(z)]),
        }
        y_pred = eval(formula_str, {"__builtins__": {}}, safe)
        y_true = np.array(y_vals)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.full_like(y_true, float(y_pred))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        return rmse if np.isfinite(rmse) else float("inf")
    except Exception:
        return float("inf")


def run_comparison_benchmark(
    model,
    device,
    methods: List[str] = None,
    gp_timeout: int = 30,
    beam_width: int = 50,
    n_problems: int = 10,
    progress_callback: Optional[Callable] = None,
) -> Dict:
    """
    Run the comparison benchmark over a set of standard problems.

    Parameters
    ----------
    model : unused (legacy parameter for compatibility)
    device : unused (legacy parameter for compatibility)
    methods : list of method names to test (e.g. ["hybrid"])
    gp_timeout : seconds per problem for the GPU engine
    beam_width : unused (kept for API compatibility)
    n_problems : how many benchmark problems to solve (max 10)
    progress_callback : optional callable(progress_float, desc=str)

    Returns
    -------
    dict with keys:
        'results' : list of per-problem result dicts
        'summary' : dict by method with aggregated stats
    """
    from AlphaSymbolic.core.gpu import TensorGeneticEngine

    if methods is None or len(methods) == 0:
        methods = ["hybrid"]

    problems = BENCHMARK_PROBLEMS[:min(n_problems, len(BENCHMARK_PROBLEMS))]
    total_tasks = len(problems) * len(methods)
    task_counter = 0

    results = []
    per_method_stats: Dict[str, dict] = {}

    for method in methods:
        per_method_stats[method] = {
            "solved": 0,
            "total": len(problems),
            "rmse_sum": 0.0,
            "time_sum": 0.0,
            "avg_rmse": float("inf"),
            "avg_time": 0.0,
            "score": 0.0,
        }

    for prob in problems:
        x_vals, y_vals = _generate_data(prob, n_points=60)

        for method in methods:
            task_counter += 1
            progress_pct = task_counter / total_tasks
            if progress_callback:
                progress_callback(progress_pct, desc=f"[{method.upper()}] {prob['name']}...")

            t0 = time.time()
            formula_found = None
            rmse = float("inf")

            try:
                # Always use the GPU engine — method name currently controls settings
                engine = TensorGeneticEngine(num_variables=1, max_constants=10)
                formula_found = engine.run(
                    x_vals,
                    y_vals,
                    seeds=[],
                    timeout_sec=gp_timeout,
                    use_log=False,
                )
                del engine

                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if formula_found:
                    rmse = _compute_rmse(formula_found, x_vals, y_vals)

            except Exception as e:
                formula_found = f"Error: {e}"

            elapsed = time.time() - t0
            success = rmse < 0.01

            results.append({
                "problem_name": prob["name"],
                "level": prob["level"],
                "method": method,
                "formula": formula_found or "No solution",
                "rmse": rmse,
                "time": elapsed,
                "success": success,
            })

            stats = per_method_stats[method]
            if success:
                stats["solved"] += 1
            if rmse < 1e9:
                stats["rmse_sum"] += rmse
            stats["time_sum"] += elapsed

    # Finalize summary stats
    for method, stats in per_method_stats.items():
        n = stats["total"]
        stats["avg_rmse"] = stats["rmse_sum"] / n if n > 0 else float("inf")
        stats["avg_time"] = stats["time_sum"] / n if n > 0 else 0.0
        stats["score"] = (stats["solved"] / n * 100) if n > 0 else 0.0

    return {
        "results": results,
        "summary": per_method_stats,
    }
