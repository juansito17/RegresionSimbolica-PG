"""Freeze the universal evaluation configuration before a benchmark run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from warpsymbolic.api.sklearn import WarpSymbolicRegressor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-time", type=float, default=60.0)
    args = parser.parse_args(argv)
    estimator = WarpSymbolicRegressor(
        pop_size=50_000,
        search_mode="adaptive",
        target_transform="auto",
        max_time=min(float(args.max_time), 60.0),
        random_state=0,
        max_active_variables=8,
    )
    params = estimator.get_params(deep=False)
    params.pop("random_state", None)
    params["random_state_policy"] = "external_repetition_seed"
    canonical = json.dumps(params, sort_keys=True, default=str, separators=(",", ":"))
    payload = {
        "protocol": "adaptive-v1",
        "configuration": params,
        "configuration_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "integrity": {
            "dataset_metadata": False,
            "test_access": False,
            "single_expression": True,
            "max_time_sec": min(float(args.max_time), 60.0),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(payload["configuration_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
