"""Compatibility entry point for repeated, bounded universal searches.

Each round is an independent 60-second-or-less estimator fit.  The command has
no built-in dataset, formula, derived feature or test value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from warpsymbolic.api.sklearn import WarpSymbolicRegressor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--target", required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-time", type=float, default=60.0)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.csv)
    y = frame.pop(args.target).to_numpy(dtype=np.float64)
    for seed in range(int(args.rounds)):
        model = WarpSymbolicRegressor(
            search_mode="adaptive",
            target_transform="auto",
            max_time=min(float(args.max_time), 60.0),
            random_state=seed,
        ).fit(frame, y)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "formula": model.sympy_formula_,
                    "mdl": model.mdl_,
                    "complexity": model.symbolic_complexity_,
                    "configuration_sha256": model.configuration_hash_,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
