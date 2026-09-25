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
import time

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
    parser.add_argument("--pop-size", type=int, default=100_000)
    parser.add_argument("--islands", type=int, default=20)
    parser.add_argument("--generations", type=int, default=None,
                        help="Límite opcional de generaciones; sin este argumento solo rige --max-time")
    parser.add_argument("--progress-interval", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    frame = pd.read_csv(args.csv)
    if args.target not in frame.columns:
        raise SystemExit(f"target column not found: {args.target!r}")
    y = frame.pop(args.target).to_numpy(dtype=np.float64)
    started = time.monotonic()

    def show_progress(generation, best_rmse, _rpn, _constants, improved, _island):
        label = "mejora" if improved else "avance"
        generation_label = str(generation) if args.generations is None else f"{generation}/{args.generations}"
        print(
            f"[{time.monotonic() - started:7.1f}s] "
            f"Gen {generation_label} | "
            f"mejor RMSE: {best_rmse:.6g} | {label}",
            flush=True,
        )

    model = WarpSymbolicRegressor(
        search_mode="legacy" if args.legacy else "adaptive",
        target_transform="auto",
        max_time=min(float(args.max_time), 60.0) if not args.legacy else float(args.max_time),
        random_state=int(args.seed),
        device=args.device,
        pop_size=args.pop_size,
        n_islands=args.islands,
        generations=args.generations,
    ).fit(
        frame,
        y,
        progress_callback=show_progress if args.legacy else None,
        progress_interval=args.progress_interval if args.legacy else None,
    )
    if getattr(model, "engine_error_", None):
        raise RuntimeError(f"GPU engine failed: {model.engine_error_}")
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
