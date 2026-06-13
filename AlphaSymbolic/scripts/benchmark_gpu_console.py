"""
Reproducible GPU-console performance benchmark.

Writes one JSON line per measured repeat so speed changes can be tracked over time.
The workload mirrors run_gpu_console.py data, but runs for bounded generations.
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from AlphaSymbolic.core.gpu import TensorGeneticEngine
from AlphaSymbolic.core.gpu.config import GpuGlobals


TARGETS = GpuGlobals.PROBLEM_Y_FULL
indices = np.array(GpuGlobals.PROBLEM_X_FILTERED, dtype=np.float64)
x1_vals = indices % GpuGlobals.VAR_MOD_X1
x2_vals = indices % GpuGlobals.VAR_MOD_X2
X_VALUES = np.column_stack((indices, x1_vals, x2_vals))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return None


def _git_diff_sha() -> str | None:
    try:
        out = subprocess.check_output(
            [
                "git", "diff", "--no-ext-diff", "HEAD", "--", ".",
                ":(exclude)benchmarks/*.jsonl",
                ":(exclude)benchmarks/profile*.jsonl",
            ],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
            stderr=subprocess.DEVNULL,
        )
        return hashlib.sha256(out).hexdigest()[:16]
    except Exception:
        return None


def _trimmed_mean(values: list[float]) -> float:
    if len(values) >= 3:
        trimmed = sorted(values)[1:-1]
        if trimmed:
            return sum(trimmed) / len(trimmed)
    return sum(values) / len(values)


def _configure(args) -> None:
    GpuGlobals.POP_SIZE = int(args.pop_size)
    GpuGlobals.POPULATION_SIZE = int(args.pop_size)
    GpuGlobals.NUM_ISLANDS = int(args.islands)
    GpuGlobals.GENERATIONS = int(args.generations)
    GpuGlobals.PROGRESS_REPORT_INTERVAL = max(1_000_000, int(args.generations) + 1)
    GpuGlobals.CONSOLE_SHOW_PREDICTION_TABLE = False
    GpuGlobals.SKIP_FINAL_FORMULA_BUILD = True
    if hasattr(GpuGlobals, "USE_PATTERN_SEEDS"):
        GpuGlobals.USE_PATTERN_SEEDS = False


def _run_once(args, repeat_idx: int, run_meta: dict) -> dict:
    if args.seed is not None:
        seed = int(args.seed) + repeat_idx
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    engine = TensorGeneticEngine(
        num_variables=3,
        max_constants=GpuGlobals.MAX_CONSTANTS,
        max_len=GpuGlobals.MAX_FORMULA_LENGTH,
    )

    x_input = X_VALUES[:len(TARGETS)] if engine.num_variables != 1 else X_VALUES[:len(TARGETS), 0]

    if args.warmup_generations > 0:
        old_generations = GpuGlobals.GENERATIONS
        GpuGlobals.GENERATIONS = int(args.warmup_generations)
        engine.run(x_input, TARGETS, seeds=[], timeout_sec=args.timeout_sec)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        GpuGlobals.GENERATIONS = old_generations

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = engine.run(x_input, TARGETS, seeds=[], timeout_sec=args.timeout_sec)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generations_completed = int(getattr(engine, "last_run_generations", GpuGlobals.GENERATIONS))
    evals = int(engine.pop_size) * max(0, generations_completed)
    best_rmse = getattr(engine, "last_run_best_rmse", None)
    best_rmse_json = float(best_rmse) if best_rmse is not None and math.isfinite(float(best_rmse)) else None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "record_type": "repeat",
        "git_sha": run_meta["git_sha"],
        "git_dirty": run_meta["git_dirty"],
        "git_diff_sha": run_meta["git_diff_sha"],
        "repeat": repeat_idx + 1,
        "device": run_meta["device"],
        "pop_size": int(engine.pop_size),
        "islands": int(engine.n_islands),
        "generations": int(GpuGlobals.GENERATIONS),
        "warmup_generations": int(args.warmup_generations),
        "cooldown_sec": float(run_meta["cooldown_sec"]),
        "config": run_meta["config"],
        "elapsed_sec": elapsed,
        "total_evals": evals,
        "evals_per_sec": evals / elapsed if elapsed > 0 else 0.0,
        "seconds_per_generation": elapsed / generations_completed if generations_completed > 0 else None,
        "best_rmse": best_rmse_json,
        "generations_completed": generations_completed,
        "converged": bool(getattr(engine, "last_run_converged", result is not None)),
        "best_formula": getattr(engine, "last_run_best_formula", result),
        "result": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark AlphaSymbolic GPU console route.")
    parser.add_argument("--pop-size", type=int, default=GpuGlobals.POP_SIZE)
    parser.add_argument("--islands", type=int, default=GpuGlobals.NUM_ISLANDS)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--warmup-generations", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--discard-first", type=int, default=1,
                        help="Repeats iniciales a excluir de hot_mean/hot_median.")
    parser.add_argument("--cooldown-sec", type=float, default=0.0,
                        help="Pausa entre repeats para reducir thermal throttling en portátiles.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--output", default=os.path.join("benchmarks", "gpu_console_metrics.jsonl"))
    args = parser.parse_args()

    _configure(args)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    rows = []
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_meta = {
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "git_diff_sha": _git_diff_sha(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "config": {
            "max_formula_length": int(getattr(GpuGlobals, "MAX_FORMULA_LENGTH", 0)),
            "max_constants": int(getattr(GpuGlobals, "MAX_CONSTANTS", 0)),
            "base_mutation_rate": float(getattr(GpuGlobals, "BASE_MUTATION_RATE", 0.0)),
            "crossover_rate": float(getattr(GpuGlobals, "DEFAULT_CROSSOVER_RATE", 0.0)),
            "tournament_size": int(getattr(GpuGlobals, "DEFAULT_TOURNAMENT_SIZE", 0)),
            "library_update_interval": int(getattr(GpuGlobals, "LIBRARY_UPDATE_INTERVAL", 0)),
            "library_top_k_max": int(getattr(GpuGlobals, "LIBRARY_TOP_K_MAX", 0)),
            "use_pattern_memory": bool(getattr(GpuGlobals, "USE_PATTERN_MEMORY", False)),
            "pattern_record_interval": int(getattr(GpuGlobals, "PATTERN_RECORD_INTERVAL", 0)),
            "pattern_inject_interval": int(getattr(GpuGlobals, "PATTERN_INJECT_INTERVAL", 0)),
            "pattern_inject_percent": float(getattr(GpuGlobals, "PATTERN_INJECT_PERCENT", 0.0)),
            "bfgs_interval": int(getattr(GpuGlobals, "BFGS_INTERVAL", 0)),
            "bfgs_top_k": int(getattr(GpuGlobals, "BFGS_TOP_K", 0)),
            "bfgs_max_iter": int(getattr(GpuGlobals, "BFGS_MAX_ITER", 0)),
            "pso_interval": int(getattr(GpuGlobals, "PSO_INTERVAL", 0)),
            "pso_particles": int(getattr(GpuGlobals, "PSO_PARTICLES", 0)),
            "pso_steps_normal": int(getattr(GpuGlobals, "PSO_STEPS_NORMAL", 0)),
            "pso_steps_stagnation": int(getattr(GpuGlobals, "PSO_STEPS_STAGNATION", 0)),
            "pso_k_normal": int(getattr(GpuGlobals, "PSO_K_NORMAL", 0)),
            "pso_k_stagnation": int(getattr(GpuGlobals, "PSO_K_STAGNATION", 0)),
            "best_sync_interval": int(getattr(GpuGlobals, "BEST_SYNC_INTERVAL", 0)),
            "repair_invalid_interval": int(getattr(GpuGlobals, "REPAIR_INVALID_INTERVAL", 0)),
            "fitness_sharing_interval": int(getattr(GpuGlobals, "FITNESS_SHARING_INTERVAL", 0)),
            "fitness_sharing_weight": float(getattr(GpuGlobals, "FITNESS_SHARING_WEIGHT", 0.0)),
            "use_cuda_fitness_sharing": bool(getattr(GpuGlobals, "USE_CUDA_FITNESS_SHARING", False)),
            "validate_cuda_random_population": bool(getattr(GpuGlobals, "VALIDATE_CUDA_RANDOM_POPULATION", False)),
        },
        "cooldown_sec": float(args.cooldown_sec),
    }
    for repeat_idx in range(max(1, int(args.repeats))):
        if repeat_idx > 0 and args.cooldown_sec > 0:
            time.sleep(float(args.cooldown_sec))
        row = _run_once(args, repeat_idx, run_meta)
        row["run_id"] = run_id
        rows.append(row)
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"repeat={row['repeat']} elapsed={row['elapsed_sec']:.4f}s "
            f"evals/s={row['evals_per_sec']:,.1f} "
            f"best_rmse={row['best_rmse']} result={row['result']}"
        )

    speeds = [row["evals_per_sec"] for row in rows]
    best_rmses = [row["best_rmse"] for row in rows if row["best_rmse"] is not None]
    discard_first = max(0, min(int(args.discard_first), len(speeds) - 1))
    hot_speeds = speeds[discard_first:] if speeds[discard_first:] else speeds
    hot_rows = rows[discard_first:] if rows[discard_first:] else rows
    hot_rmses = [row["best_rmse"] for row in hot_rows if row["best_rmse"] is not None]
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "record_type": "summary",
        "run_id": run_id,
        "git_sha": run_meta["git_sha"],
        "git_dirty": run_meta["git_dirty"],
        "git_diff_sha": run_meta["git_diff_sha"],
        "device": run_meta["device"],
        "pop_size": int(args.pop_size),
        "islands": int(args.islands),
        "generations": int(args.generations),
        "warmup_generations": int(args.warmup_generations),
        "cooldown_sec": float(args.cooldown_sec),
        "config": run_meta["config"],
        "repeats": len(rows),
        "discard_first": discard_first,
        "min_evals_per_sec": min(speeds),
        "mean_evals_per_sec": sum(speeds) / len(speeds),
        "median_evals_per_sec": statistics.median(speeds),
        "max_evals_per_sec": max(speeds),
        "stdev_evals_per_sec": statistics.stdev(speeds) if len(speeds) > 1 else 0.0,
        "hot_mean_evals_per_sec": sum(hot_speeds) / len(hot_speeds),
        "hot_trimmed_mean_evals_per_sec": _trimmed_mean(hot_speeds),
        "hot_median_evals_per_sec": statistics.median(hot_speeds),
        "hot_min_evals_per_sec": min(hot_speeds),
        "hot_max_evals_per_sec": max(hot_speeds),
        "converged_repeats": sum(1 for row in rows if row["converged"]),
        "best_rmse_min": min(best_rmses) if best_rmses else None,
        "best_rmse_mean": (sum(best_rmses) / len(best_rmses)) if best_rmses else None,
        "best_rmse_median": statistics.median(best_rmses) if best_rmses else None,
        "hot_best_rmse_min": min(hot_rmses) if hot_rmses else None,
        "hot_best_rmse_mean": (sum(hot_rmses) / len(hot_rmses)) if hot_rmses else None,
        "hot_best_rmse_median": statistics.median(hot_rmses) if hot_rmses else None,
    }
    with open(args.output, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, sort_keys=True) + "\n")

    print(
        f"summary repeats={len(rows)} min={min(speeds):,.1f} "
        f"mean={summary['mean_evals_per_sec']:,.1f} "
        f"median={summary['median_evals_per_sec']:,.1f} max={max(speeds):,.1f}"
    )
    print(
        f"hot discard_first={discard_first} "
        f"mean={summary['hot_mean_evals_per_sec']:,.1f} "
        f"trimmed={summary['hot_trimmed_mean_evals_per_sec']:,.1f} "
        f"median={summary['hot_median_evals_per_sec']:,.1f} "
        f"range={summary['hot_min_evals_per_sec']:,.1f}..{summary['hot_max_evals_per_sec']:,.1f}"
    )
    if best_rmses:
        print(
            f"fitness best_rmse_min={summary['best_rmse_min']:.6g} "
            f"hot_min={summary['hot_best_rmse_min']:.6g} "
            f"hot_median={summary['hot_best_rmse_median']:.6g} "
            f"converged={summary['converged_repeats']}/{len(rows)}"
        )
    print(f"wrote={os.path.abspath(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
