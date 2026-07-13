import argparse
import json
import os
import random
import sys
import time
import statistics
from datetime import datetime, timezone

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from AlphaSymbolic.core.gpu import TensorGeneticEngine
from AlphaSymbolic.core.gpu.config import GpuGlobals


def _configure(args, use_logspace_prior: bool) -> None:
    GpuGlobals.POP_SIZE = int(args.pop_size)
    GpuGlobals.NUM_ISLANDS = int(args.islands)
    GpuGlobals.GENERATIONS = int(args.generations)
    GpuGlobals.PROGRESS_REPORT_INTERVAL = 100000
    GpuGlobals.USE_INITIAL_POP_CACHE = False
    GpuGlobals.USE_INITIAL_FORMULA = False
    GpuGlobals.USE_STRUCTURAL_SEEDS = False
    GpuGlobals.USE_PATTERN_SEEDS = False
    GpuGlobals.USE_SNIPER = False
    GpuGlobals.USE_RESIDUAL_BOOSTING = False
    GpuGlobals.USE_LOG_TRANSFORMATION = True
    GpuGlobals.ALLOW_WARMUP_EARLY_EXIT = False
    GpuGlobals.USE_LEXICASE_SELECTION = False
    GpuGlobals.USE_LOGSPACE_ALGEBRAIC_SAMPLING = bool(use_logspace_prior)


def _make_problem(name: str):
    if name == "simple":
        x = np.linspace(0.0, 6.0, 25)
        y = 2.5 * np.exp(0.73 * x)
    elif name == "log_quadratic":
        x = np.linspace(-3.0, 3.0, 31)
        y = np.exp(0.21 * x * x - 0.7 * x + 1.5)
    elif name == "log_cubic":
        x = np.linspace(-2.5, 2.5, 31)
        y = np.exp(0.055 * x * x * x - 0.18 * x * x + 0.62 * x + 1.2)
    elif name == "log_rational":
        x = np.linspace(-3.0, 3.0, 31)
        y = np.exp(1.1 + 0.8 * x / (x * x + 2.0))
    elif name == "positive_quadratic":
        x = np.linspace(-3.0, 3.0, 31)
        y = x * x + 2.0 * x + 6.0
    else:
        raise ValueError(f"unknown problem: {name}")
    return x, y


def _parse_seeds(seed_text: str | None, fallback: int) -> list[int]:
    if not seed_text:
        return [int(fallback)]
    seeds = []
    for part in seed_text.split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    return seeds or [int(fallback)]


def _summary_row(rows: list[dict]) -> dict:
    rmses = [float(r["best_rmse"]) for r in rows]
    elapsed = [float(r["elapsed_sec"]) for r in rows]
    converged = [r for r in rows if float(r["best_rmse"]) < float(GpuGlobals.EXACT_SOLUTION_THRESHOLD)]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "record_type": "exponential_convergence_summary",
        "problem": rows[0]["problem"] if rows else None,
        "use_logspace_prior": rows[0]["use_logspace_prior"] if rows else None,
        "seeds": [r["seed"] for r in rows],
        "pop_size": rows[0]["pop_size"] if rows else None,
        "islands": rows[0]["islands"] if rows else None,
        "generations": rows[0]["generations"] if rows else None,
        "runs": len(rows),
        "converged_runs": len(converged),
        "rmse_min": min(rmses) if rmses else None,
        "rmse_median": statistics.median(rmses) if rmses else None,
        "rmse_mean": statistics.fmean(rmses) if rmses else None,
        "elapsed_mean": statistics.fmean(elapsed) if elapsed else None,
        "generations_completed_median": statistics.median([r["generations_completed"] for r in rows]) if rows else None,
    }


def _run_once(args, problem: str, use_logspace_prior: bool, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    _configure(args, use_logspace_prior)
    x, y = _make_problem(problem)
    engine = TensorGeneticEngine(
        num_variables=1,
        pop_size=int(args.pop_size),
        n_islands=int(args.islands),
        max_constants=GpuGlobals.MAX_CONSTANTS,
        max_len=GpuGlobals.MAX_FORMULA_LENGTH,
    )

    started = time.perf_counter()
    formula = engine.run(x, y, seeds=[], timeout_sec=float(args.timeout_sec), use_log=True)
    elapsed = time.perf_counter() - started

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "record_type": "exponential_convergence",
        "problem": problem,
        "seed": seed,
        "use_logspace_prior": bool(use_logspace_prior),
        "pop_size": int(args.pop_size),
        "islands": int(args.islands),
        "generations": int(args.generations),
        "generations_completed": int(getattr(engine, "last_run_generations", 0)),
        "elapsed_sec": elapsed,
        "best_rmse": float(getattr(engine, "last_run_best_rmse", float("inf"))),
        "formula": getattr(engine, "last_run_best_formula", None) or formula,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark exponential convergence without formula templates.")
    parser.add_argument("--problem", choices=["simple", "log_quadratic", "log_cubic", "log_rational", "positive_quadratic"], default="log_quadratic")
    parser.add_argument("--pop-size", type=int, default=50000)
    parser.add_argument("--islands", type=int, default=10)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--timeout-sec", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Overrides --seed when provided.")
    parser.add_argument("--compare", action="store_true", help="Run baseline and log-space prior variants.")
    parser.add_argument("--output", default="benchmarks/exponential_convergence_metrics.jsonl")
    args = parser.parse_args()

    variants = [False, True] if args.compare else [True]
    seeds = _parse_seeds(args.seeds, int(args.seed))
    rows = []
    for use_prior in variants:
        variant_rows = []
        for seed in seeds:
            row = _run_once(args, args.problem, use_prior, int(seed))
            rows.append(row)
            variant_rows.append(row)
            print(
                f"problem={row['problem']} seed={row['seed']} prior={row['use_logspace_prior']} "
                f"rmse={row['best_rmse']:.9g} gens={row['generations_completed']} "
                f"elapsed={row['elapsed_sec']:.3f}s"
            )
        summary = _summary_row(variant_rows)
        rows.append(summary)
        print(
            f"summary problem={summary['problem']} prior={summary['use_logspace_prior']} "
            f"converged={summary['converged_runs']}/{summary['runs']} "
            f"rmse_median={summary['rmse_median']:.9g} elapsed_mean={summary['elapsed_mean']:.3f}s"
        )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
