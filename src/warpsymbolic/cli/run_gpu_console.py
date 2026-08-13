"""Run the universal WarpSymbolic estimator from a CSV file.

No dataset or target sequence is embedded here.  The command accepts exactly
the same numeric X/y information as the Python and web APIs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from warpsymbolic.api.sklearn import WarpSymbolicRegressor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV containing feature and target columns")
    parser.add_argument("--target", required=True, help="target column name")
    parser.add_argument("--max-time", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Activar logs detallados")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    frame = pd.read_csv(args.csv)
    if args.target not in frame.columns:
        raise SystemExit(f"target column not found: {args.target!r}")
    y = frame.pop(args.target).to_numpy(dtype=np.float64)
    model = WarpSymbolicRegressor(
        search_mode="legacy" if args.legacy else "adaptive",
        target_transform="auto",
        max_time=min(float(args.max_time), 60.0) if not args.legacy else float(args.max_time),
        random_state=int(args.seed),
        device=args.device,
    ).fit(frame, y)
    prediction = model.predict(frame)
    rmse = float(np.sqrt(np.mean((prediction - y) ** 2)))
    print(model.sympy_formula_)
    print(
        json.dumps(
            {
                "rmse": rmse,
                "complexity": int(model.symbolic_complexity_),
                "configuration_sha256": getattr(model, "configuration_hash_", None),
                "search_report": getattr(model, "search_report_", None),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
