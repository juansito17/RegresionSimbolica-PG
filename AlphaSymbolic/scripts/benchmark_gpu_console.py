"""Repeatable, dataset-agnostic AlphaSymbolic benchmark command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from AlphaSymbolic.sklearn import AlphaSymbolicRegressor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--target", required=True)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--max-time", type=float, default=60.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.csv)
    y = frame.pop(args.target).to_numpy(dtype=np.float64)
    rows = []
    for seed in range(int(args.seeds)):
        started = time.perf_counter()
        model = AlphaSymbolicRegressor(
            search_mode="adaptive",
            target_transform="auto",
            max_time=min(float(args.max_time), 60.0),
            random_state=seed,
        ).fit(frame, y)
        prediction = model.predict(frame)
        row = {
            "seed": seed,
            "r2": float(r2_score(y, prediction)),
            "rmse": float(np.sqrt(np.mean((prediction - y) ** 2))),
            "time_sec": time.perf_counter() - started,
            "complexity": int(model.symbolic_complexity_),
            "formula": model.sympy_formula_,
            "configuration_sha256": model.configuration_hash_,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
