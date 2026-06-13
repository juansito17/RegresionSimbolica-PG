import sys
import os
import torch
import time
import cProfile
import pstats
import argparse
import json
from datetime import datetime, timezone

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from AlphaSymbolic.core.gpu.engine import TensorGeneticEngine
from AlphaSymbolic.core.gpu.config import GpuGlobals

def _top_profile_rows(stats: pstats.Stats, limit: int = 20):
    rows = []
    for func, stat in sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:limit]:
        cc, nc, tt, ct, _callers = stat
        filename, line, name = func
        rows.append({
            "file": filename,
            "line": line,
            "function": name,
            "primitive_calls": cc,
            "calls": nc,
            "total_time_sec": tt,
            "cumulative_time_sec": ct,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Profile AlphaSymbolic GPU engine hot path.")
    parser.add_argument("--pop-size", type=int, default=500000)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--warmup-timeout-sec", type=float, default=2.0)
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--output", default=None, help="Optional JSONL output path for profile summaries.")
    args = parser.parse_args()

    torch.manual_seed(42)
    GpuGlobals.POPULATION_SIZE = int(args.pop_size)
    GpuGlobals.POP_SIZE = int(args.pop_size)
    GpuGlobals.GENERATIONS = int(args.generations)
    GpuGlobals.USE_LEXICASE_SELECTION = False
    GpuGlobals.SKIP_FINAL_FORMULA_BUILD = True
    
    x = torch.linspace(-10, 10, 200).unsqueeze(1).to("cuda" if torch.cuda.is_available() else "cpu")
    y = torch.sin(x) + x*0.5
    
    engine = TensorGeneticEngine(pop_size=int(args.pop_size))
    
    print("Running warmup...")
    engine.run(x, y, timeout_sec=float(args.warmup_timeout_sec))
    
    print(f"Running profiler for {GpuGlobals.GENERATIONS} generations...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    start_time = time.time()
    
    engine.run(x, y, timeout_sec=float(args.timeout_sec))
    
    profiler.disable()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal Time ({GpuGlobals.GENERATIONS} gens, {engine.pop_size} pop): {elapsed:.3f}s")
    
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(50)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        gens_completed = int(getattr(engine, "last_run_generations", GpuGlobals.GENERATIONS))
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "record_type": "profile",
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "pop_size": int(engine.pop_size),
            "generations": int(GpuGlobals.GENERATIONS),
            "generations_completed": gens_completed,
            "elapsed_sec": elapsed,
            "evals_per_sec": (int(engine.pop_size) * gens_completed / elapsed) if elapsed > 0 else 0.0,
            "best_rmse": getattr(engine, "last_run_best_rmse", None),
            "top_cumulative": _top_profile_rows(stats, 20),
        }
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"profile_jsonl={args.output}")

if __name__ == "__main__":
    main()
