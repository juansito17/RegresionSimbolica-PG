"""
Summarize GPU console benchmark JSONL runs.

Groups comparable summary rows by workload and key config knobs so optimization
experiments can be compared without hand-reading the metrics log.
"""
import argparse
import json
import os
import statistics
from collections import defaultdict
from typing import Any


KEY_CONFIG_FIELDS = (
    "bfgs_interval",
    "bfgs_top_k",
    "bfgs_max_iter",
    "pso_interval",
    "pso_steps_normal",
    "pso_k_normal",
    "tournament_size",
    "library_update_interval",
    "library_top_k_max",
    "fitness_sharing_interval",
    "fitness_sharing_weight",
    "use_cuda_fitness_sharing",
    "use_pattern_memory",
    "pattern_record_interval",
    "pattern_inject_interval",
    "pattern_inject_percent",
    "validate_cuda_random_population",
)

DEFAULT_CONFIG_COMPAT = {
    "fitness_sharing_interval": 20,
    "fitness_sharing_weight": 0.05,
    "use_cuda_fitness_sharing": False,
    "use_pattern_memory": True,
    "pattern_record_interval": 30,
    "pattern_inject_interval": 25,
    "pattern_inject_percent": 0.05,
    "validate_cuda_random_population": False,
}


def _load_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _key(row: dict[str, Any], split_code: bool = False) -> tuple[Any, ...]:
    config = row.get("config") or {}
    parts = [
        row.get("pop_size"),
        row.get("islands"),
        row.get("generations"),
        row.get("warmup_generations"),
        row.get("discard_first"),
        tuple((name, config.get(name, DEFAULT_CONFIG_COMPAT.get(name))) for name in KEY_CONFIG_FIELDS
              if name in config or name in DEFAULT_CONFIG_COMPAT),
    ]
    if split_code:
        parts.extend([row.get("git_sha"), row.get("git_dirty"), row.get("git_diff_sha")])
    return tuple(parts)


def _fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _config_label(config: dict[str, Any]) -> str:
    bits = []
    for name in KEY_CONFIG_FIELDS:
        if name in config or name in DEFAULT_CONFIG_COMPAT:
            bits.append(f"{name}={config.get(name, DEFAULT_CONFIG_COMPAT.get(name))}")
    return ", ".join(bits)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze AlphaSymbolic GPU benchmark JSONL.")
    parser.add_argument("--input", default=os.path.join("benchmarks", "gpu_console_metrics.jsonl"))
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--islands", type=int, default=None)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--min-runs", type=int, default=1,
                        help="Minimum summary rows per config before it appears in the ranking.")
    parser.add_argument("--split-code", action="store_true",
                        help="Separate groups by git_sha/git_dirty/git_diff_sha when present.")
    parser.add_argument("--sort", choices=("rmse", "speed", "balanced"), default="rmse")
    args = parser.parse_args()

    rows = [
        row for row in _load_rows(args.input)
        if row.get("record_type") == "summary"
    ]
    if args.pop_size is not None:
        rows = [row for row in rows if row.get("pop_size") == args.pop_size]
    if args.generations is not None:
        rows = [row for row in rows if row.get("generations") == args.generations]
    if args.islands is not None:
        rows = [row for row in rows if row.get("islands") == args.islands]

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_key(row, args.split_code)].append(row)

    summaries = []
    for group_rows in groups.values():
        speeds = [r.get("hot_trimmed_mean_evals_per_sec") for r in group_rows if r.get("hot_trimmed_mean_evals_per_sec") is not None]
        rmses = [r.get("hot_best_rmse_median") for r in group_rows if r.get("hot_best_rmse_median") is not None]
        mins = [r.get("hot_best_rmse_min") for r in group_rows if r.get("hot_best_rmse_min") is not None]
        if len(group_rows) < max(1, args.min_runs) or not speeds:
            continue
        latest = max(group_rows, key=lambda r: r.get("timestamp_utc", ""))
        summaries.append({
            "runs": len(group_rows),
            "latest": latest,
            "speed_median": statistics.median(speeds),
            "speed_best": max(speeds),
            "rmse_median": statistics.median(rmses) if rmses else None,
            "rmse_best": min(mins) if mins else None,
            "code": (latest.get("git_sha"), latest.get("git_dirty"), latest.get("git_diff_sha")),
        })

    if args.sort == "speed":
        summaries.sort(key=lambda r: (-r["speed_median"], r["rmse_median"] if r["rmse_median"] is not None else float("inf")))
    elif args.sort == "balanced":
        valid_speeds = [r["speed_median"] for r in summaries if r["speed_median"]]
        valid_rmse = [r["rmse_median"] for r in summaries if r["rmse_median"] is not None]
        best_speed = max(valid_speeds) if valid_speeds else 1.0
        best_rmse = min(valid_rmse) if valid_rmse else 1.0
        summaries.sort(key=lambda r: (
            (r["rmse_median"] / best_rmse if r["rmse_median"] else float("inf")) +
            (best_speed / r["speed_median"] if r["speed_median"] else float("inf"))
        ))
    else:
        summaries.sort(key=lambda r: (
            r["rmse_median"] if r["rmse_median"] is not None else float("inf"),
            -r["speed_median"],
        ))

    print(
        f"loaded_summaries={len(rows)} grouped_configs={len(summaries)} "
        f"min_runs={max(1, args.min_runs)} split_code={args.split_code} "
        f"sort={args.sort} input={args.input}"
    )
    code_suffix = " | code" if args.split_code else ""
    print(f"rank | runs | pop/gen/islands | hot_trimmed/s | hot_rmse_med | hot_rmse_min | config{code_suffix}")
    for idx, item in enumerate(summaries[:max(1, args.limit)], 1):
        row = item["latest"]
        config = row.get("config") or {}
        workload = f"{row.get('pop_size')}/{row.get('generations')}/{row.get('islands')}"
        line = (
            f"{idx:>4} | {item['runs']:>4} | {workload:<15} | "
            f"{_fmt_num(item['speed_median'], 1):>13} | "
            f"{_fmt_num(item['rmse_median'], 5):>12} | "
            f"{_fmt_num(item['rmse_best'], 5):>12} | "
            f"{_config_label(config)}"
        )
        if args.split_code:
            git_sha, git_dirty, git_diff_sha = item["code"]
            line += f" | sha={git_sha} dirty={git_dirty} diff={git_diff_sha}"
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
