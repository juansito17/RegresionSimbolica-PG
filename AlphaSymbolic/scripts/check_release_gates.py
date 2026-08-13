"""Fail closed unless every adaptive-promotion gate has evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _median(rows: list[dict], key: str) -> float:
    return float(statistics.median(float(row[key]) for row in rows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--nqueens", type=Path, required=True)
    parser.add_argument("--cuda-audit", type=Path, required=True)
    parser.add_argument("--srbench-runs", type=Path, required=True)
    parser.add_argument("--srbench-ranking", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    checks: dict[str, dict] = {}

    development = _jsonl(args.development)
    adaptive = [row for row in development if row.get("mode") == "adaptive"]
    legacy = [row for row in development if row.get("mode") == "legacy"]
    paired = bool(adaptive and legacy)
    checks["development"] = {
        "passed": paired
        and sum(bool(row.get("recovered_numeric")) for row in adaptive)
        > sum(bool(row.get("recovered_numeric")) for row in legacy)
        and _median(adaptive, "rmse") < _median(legacy, "rmse")
        and _median(adaptive, "complexity") <= 0.25 * _median(legacy, "complexity")
        and _median(adaptive, "time_sec") <= 10.0,
        "adaptive_rows": len(adaptive),
        "legacy_rows": len(legacy),
    }

    nqueens = _jsonl(args.nqueens)
    nqueen_model = [row for row in nqueens if row.get("mode") == "adaptive"]
    nqueen_summary = next(
        (row for row in nqueens if row.get("mode") == "adaptive-summary"), None
    )
    baselines = [row for row in nqueens if str(row.get("mode", "")).startswith("trainable-")]
    checks["nqueens"] = {
        "passed": bool(nqueen_model and baselines)
        and _median(nqueen_model, "log_rmse") < min(float(row["log_rmse"]) for row in baselines)
        and all(int(row.get("input_columns", 0)) == 1 for row in nqueen_model)
        and len({row.get("seed") for row in nqueen_model}) == 30
        and nqueen_summary is not None
        and float(nqueen_summary.get("mean_log_prediction_std", float("inf"))) < float("inf"),
        "adaptive_seeds": len(nqueen_model),
    }

    cuda_audit = json.loads(args.cuda_audit.read_text(encoding="utf-8"))
    cuda_regression = float(cuda_audit.get("cuda_regression_percent", float("inf")))
    checks["cuda_regression"] = {
        "passed": cuda_regression <= 5.0,
        "cuda_regression_percent": cuda_regression,
    }

    configuration = json.loads(args.configuration.read_text(encoding="utf-8"))
    expected_hash = configuration["configuration_sha256"]
    runs = _jsonl(args.srbench_runs)
    successful = [row for row in runs if row.get("status") == "ok"]
    run_hashes = {
        (row.get("runner_metadata") or {}).get("configuration_sha256")
        for row in successful
    }
    pairs = {(row.get("dataset"), row.get("seed")) for row in successful}
    checks["frozen_24x30"] = {
        "passed": len(pairs) == 720
        and len({pair[0] for pair in pairs}) == 24
        and len({pair[1] for pair in pairs}) == 30
        and run_hashes == {expected_hash},
        "successful_pairs": len(pairs),
        "hashes": sorted(str(value) for value in run_hashes),
    }

    ranking = json.loads(args.srbench_ranking.read_text(encoding="utf-8"))
    local = next(
        (row for row in ranking.get("summaries", []) if "[local]" in row.get("algorithm", "")),
        None,
    )
    rank_keys = (
        "mean_r2_rank",
        "mean_symbolic_recovery_rank",
        "mean_r2_over_0999_rank",
        "mean_model_size_rank",
        "mean_training_time_rank",
        "mean_energy_rank",
    )
    top_everywhere = bool(local)
    if local:
        for key in rank_keys:
            best = min(float(row.get(key, float("inf"))) for row in ranking["summaries"])
            top_everywhere = top_everywhere and float(local.get(key, float("inf"))) == best
    interval_keys = tuple(key for key in (local or {}) if key.endswith("bootstrap_95ci"))
    checks["srbench_top1_all_metrics"] = {
        "passed": bool(
            local
            and local.get("eligible_full_scope")
            and top_everywhere
            and len(interval_keys) >= 5
            and float(local.get("median_energy_joules", float("nan")))
            == float(local.get("median_energy_joules", float("nan")))
            and float(local.get("mean_symbolic_recovery", float("nan")))
            == float(local.get("mean_symbolic_recovery", float("nan")))
        ),
        "local_summary": local,
    }
    passed = all(check["passed"] for check in checks.values())
    report = {"passed": passed, "checks": checks, "promotion_allowed": passed}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
