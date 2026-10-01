"""Convergence benchmark for the GPU engine under a fixed wall-clock budget.

Every run trains on 128 noisy-free (or Friedman-noisy) points and reports the
strict-mode NRMSE of the best individual on 512 held-out points, the time to
an exact solution, generations per second and population statistics.

Example::

    python -m warpsymbolic.cli.benchmark_convergence --seeds 3 --budget 15 \
        --output benchmarks/convergence.jsonl
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import statistics
import sys
import time

import numpy as np
import torch

from warpsymbolic.gpu.config import GpuGlobals
from warpsymbolic.gpu.engine import TensorGeneticEngine

# name: (n_vars, f, train_range, test_range, noise_sd)
PROBLEMS = {
    "nguyen1": (1, lambda x: x[0] ** 3 + x[0] ** 2 + x[0], (-1, 1), (-1, 1), 0.0),
    "nguyen3": (1, lambda x: x[0] ** 5 + x[0] ** 4 + x[0] ** 3 + x[0] ** 2 + x[0], (-1, 1), (-1, 1), 0.0),
    "nguyen5": (1, lambda x: np.sin(x[0] ** 2) * np.cos(x[0]) - 1, (-1, 1), (-1, 1), 0.0),
    "nguyen6": (1, lambda x: np.sin(x[0]) + np.sin(x[0] + x[0] ** 2), (-1, 1), (-1, 1), 0.0),
    "nguyen7": (1, lambda x: np.log(x[0] + 1) + np.log(x[0] ** 2 + 1), (0, 2), (0, 2), 0.0),
    "nguyen8": (1, lambda x: np.sqrt(x[0]), (0, 4), (0, 4), 0.0),
    "feyn_gauss": (1, lambda x: np.exp(-x[0] ** 2 / 2) / np.sqrt(2 * np.pi), (1, 3), (1, 3), 0.0),
    "nguyen10": (2, lambda x: 2 * np.sin(x[0]) * np.cos(x[1]), (0, 1), (0, 1), 0.0),
    "nguyen12": (2, lambda x: x[0] ** 4 - x[0] ** 3 + 0.5 * x[1] ** 2 - x[1], (-1, 1), (-1, 1), 0.0),
    "keijzer11": (2, lambda x: x[0] * x[1] + np.sin((x[0] - 1) * (x[1] - 1)), (-3, 3), (-3, 3), 0.0),
    "vlad1": (2, lambda x: np.exp(-(x[0] - 1) ** 2) / (1.2 + (x[1] - 2.5) ** 2), (0.3, 4), (0.3, 4), 0.0),
    "pagie1": (2, lambda x: 1 / (1 + x[0] ** -4) + 1 / (1 + x[1] ** -4), (-5, 5), (-5, 5), 0.0),
    "coulomb3": (3, lambda x: x[0] * x[1] / x[2] ** 2, (1, 5), (1, 5), 0.0),
    "feyn_gauss3": (3, lambda x: np.exp(-((x[0] - x[1]) / x[2]) ** 2 / 2) / (np.sqrt(2 * np.pi) * x[2]),
                    (1, 3), (1, 3), 0.0),
    "friedman1": (5, lambda x: 10 * np.sin(np.pi * x[0] * x[1]) + 20 * (x[2] - 0.5) ** 2 + 10 * x[3] + 5 * x[4],
                  (0, 1), (0, 1), 1.0),
}


def _parse_override(item: str):
    name, value = item.split("=", 1)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    if isinstance(parsed, list):
        parsed = tuple(parsed)
    return name, parsed


def run_case(name: str, seed: int, budget: float, pop: int) -> dict:
    nv, f, tr, te, noise = PROBLEMS[name]
    rng = np.random.default_rng(seed)
    x_train = rng.uniform(tr[0], tr[1], size=(128, nv))
    x_test = rng.uniform(te[0], te[1], size=(512, nv))
    y_train = f(x_train.T) + noise * rng.standard_normal(128)
    y_test = f(x_test.T)
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        engine = TensorGeneticEngine(pop_size=pop, num_variables=nv)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        formula = engine.run(x_train.astype(np.float32), y_train.astype(np.float32), timeout_sec=budget)
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    metrics = engine.last_run_metrics
    nrmse = float("inf")
    if engine.best_global_rpn is not None:
        xt = torch.tensor(x_test, dtype=engine.dtype, device="cuda")
        yt = torch.tensor(y_test, dtype=engine.dtype, device="cuda")
        res = engine.evaluator.validate_strict(
            engine.best_global_rpn.unsqueeze(0), xt, yt,
            engine.best_global_consts.unsqueeze(0).to(engine.dtype))
        rmse = float(res["rmse"][0])
        if math.isfinite(rmse) and rmse < 1e14:
            nrmse = rmse / float(np.std(y_test))
    generations = metrics.get("generations") or 0
    return {
        "problem": name, "seed": seed, "n_vars": nv, "budget_sec": budget, "population": engine.pop_size,
        "wall_sec": round(wall, 3), "generations": generations,
        "generations_per_sec": round(generations / max(wall, 1e-9), 2),
        "train_rmse": metrics.get("best_rmse"), "converged": metrics.get("converged"),
        "test_nrmse": nrmse, "formula": formula,
        "time_to_rmse_1e-6": (metrics.get("time_to_rmse_sec") or {}).get("1e-06"),
        "invalid_fraction": metrics.get("invalid_fraction"),
        "mean_formula_length": metrics.get("sampled_mean_formula_length"),
    }


def summarize(rows: list[dict]) -> str:
    lines = [f"{'problem':12s} {'solved':>7s} {'median test NRMSE':>18s} {'median wall s':>13s}"]
    names = []
    for row in rows:
        if row["problem"] not in names:
            names.append(row["problem"])
    logs = []
    for name in names:
        rs = [r for r in rows if r["problem"] == name]
        solved = sum(1 for r in rs if r["converged"])
        nr = [r["test_nrmse"] for r in rs]
        lines.append(f"{name:12s} {solved:>3d}/{len(rs):<3d} {statistics.median(nr):18.3e} "
                     f"{statistics.median(r['wall_sec'] for r in rs):13.2f}")
        logs += [math.log10(min(max(v, 1e-9), 10.0)) if math.isfinite(v) else 1.0 for v in nr]
    total = sum(1 for r in rows if r["converged"])
    lines.append(f"solved {total}/{len(rows)}, total wall {sum(r['wall_sec'] for r in rows):.1f}s, "
                 f"geometric-mean test NRMSE {10 ** (sum(logs) / max(len(logs), 1)):.3e}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--problems", default=",".join(PROBLEMS),
                        help="Comma-separated subset of: " + ", ".join(PROBLEMS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed0", type=int, default=100)
    parser.add_argument("--budget", type=float, default=15.0, help="Wall-clock seconds per run")
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="GpuGlobals override, value parsed as JSON when possible")
    parser.add_argument("--output", default=None, help="Optional JSONL output path")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA is required for this benchmark", file=sys.stderr)
        return 2
    for item in args.set:
        name, value = _parse_override(item)
        setattr(GpuGlobals, name, value)
    GpuGlobals.USE_INITIAL_POP_CACHE = False
    pop = int(args.pop_size or GpuGlobals.POP_SIZE)
    names = [p for p in args.problems.split(",") if p]
    unknown = [p for p in names if p not in PROBLEMS]
    if unknown:
        parser.error(f"unknown problems: {unknown}")

    rows = []
    out = open(args.output, "a", encoding="utf-8") if args.output else None
    try:
        for name in names:
            for s in range(args.seeds):
                row = run_case(name, args.seed0 + s, args.budget, pop)
                row["overrides"] = args.set
                rows.append(row)
                if out:
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                print(json.dumps({k: row[k] for k in ("problem", "seed", "wall_sec", "generations",
                                                       "test_nrmse", "converged")}), flush=True)
    finally:
        if out:
            out.close()
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
