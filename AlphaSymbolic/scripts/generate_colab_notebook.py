"""Generate a dataset-agnostic Colab notebook for the public estimator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cell(kind: str, source: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def notebook() -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"name": "AlphaSymbolic Universal.ipynb", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU",
        },
        "cells": [
            _cell(
                "markdown",
                "# AlphaSymbolic universal\nUpload any CSV. The estimator receives only numeric training `X/y`; no dataset identity or target formula is used.\n",
            ),
            _cell(
                "code",
                "%pip install -q 'git+https://github.com/juansito17/Algoritmo-Genetico---Formulas.git'\n",
            ),
            _cell(
                "code",
                "from google.colab import files\n"
                "import io, pandas as pd\n"
                "uploaded = files.upload()\n"
                "name, content = next(iter(uploaded.items()))\n"
                "frame = pd.read_csv(io.BytesIO(content))\n"
                "display(frame.head())\n",
            ),
            _cell(
                "code",
                "TARGET_COLUMN = 'y'  # change to the target column in your CSV\n"
                "MAX_TIME = 60\n"
                "SEED = 0\n"
                "y = frame[TARGET_COLUMN].to_numpy()\n"
                "X = frame.drop(columns=[TARGET_COLUMN])\n",
            ),
            _cell(
                "code",
                "from AlphaSymbolic.sklearn import AlphaSymbolicRegressor\n"
                "model = AlphaSymbolicRegressor(\n"
                "    search_mode='adaptive', target_transform='auto',\n"
                "    max_time=MAX_TIME, random_state=SEED,\n"
                ").fit(X, y)\n"
                "print(model.sympy_formula_)\n"
                "print(model.search_report_)\n"
                "display(pd.DataFrame(model.pareto_front_))\n",
            ),
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("AlphaSymbolic_Universal_Colab.ipynb"),
    )
    args = parser.parse_args(argv)
    args.output.write_text(json.dumps(notebook(), indent=2), encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
