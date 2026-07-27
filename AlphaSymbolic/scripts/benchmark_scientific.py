"""
Reproducible scientific benchmark for AlphaSymbolic and optional PySR.

Unlike the legacy UI benchmark, this harness uses independent train/test data,
paired random seeds, explicit target transforms, and strict out-of-sample
metrics.  The N-Queens case models the known A000170 count sequence; it is not
a board solver or a counting proof.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy import special


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AlphaSymbolic.core.gpu import TensorGeneticEngine
from AlphaSymbolic.core.gpu.config import GpuGlobals


NQUEENS_COUNTS = np.asarray(
    [
        1, 1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200,
        73712, 365596, 2279184, 14772512, 95815104, 666090624,
        4968057848, 39029188884, 314666222712, 2691008701644,
        24233937684440, 227514171973736, 2207893435808352,
        22317699616364044, 234907967154122528,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ScientificProblem:
    key: str
    label: str
    n_variables: int
    use_log: bool
    make_data: Callable[[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


def _sample_box(
    seed: int,
    n_train: int,
    n_test: int,
    train_bounds: list[tuple[float, float]],
    test_bounds: list[tuple[float, float]],
    function: Callable[[np.ndarray], np.ndarray],
):
    train_rng = np.random.default_rng(seed)
    test_rng = np.random.default_rng(seed + 1_000_003)

    def sample(rng, count, bounds):
        columns = [rng.uniform(low, high, count) for low, high in bounds]
        return np.column_stack(columns)

    x_train = sample(train_rng, n_train, train_bounds)
    x_test = sample(test_rng, n_test, test_bounds)
    return x_train, function(x_train), x_test, function(x_test)


def _nguyen_1(seed, n_train, n_test):
    return _sample_box(
        seed, n_train, n_test, [(-1.0, 1.0)], [(-1.25, 1.25)],
        lambda x: x[:, 0] ** 3 + x[:, 0] ** 2 + x[:, 0],
    )


def _nguyen_5(seed, n_train, n_test):
    return _sample_box(
        seed, n_train, n_test, [(-1.0, 1.0)], [(-1.25, 1.25)],
        lambda x: np.sin(x[:, 0] ** 2) * np.cos(x[:, 0]) - 1.0,
    )


def _feynman_gaussian(seed, n_train, n_test):
    return _sample_box(
        seed, n_train, n_test, [(-3.0, 3.0)], [(-4.0, 4.0)],
        lambda x: np.exp(-(x[:, 0] ** 2) / 2.0) / np.sqrt(2.0 * np.pi),
    )


def _feynman_product(seed, n_train, n_test):
    return _sample_box(
        seed, n_train, n_test,
        [(-2.0, 2.0), (-3.0, 3.0)],
        [(-2.5, 2.5), (-3.5, 3.5)],
        lambda x: x[:, 0] * x[:, 1],
    )


def _friedman_1(seed, n_train, n_test):
    return _sample_box(
        seed, n_train, n_test,
        [(0.0, 1.0)] * 5,
        [(0.0, 1.0)] * 5,
        lambda x: (
            10.0 * np.sin(np.pi * x[:, 0] * x[:, 1])
            + 20.0 * (x[:, 2] - 0.5) ** 2
            + 10.0 * x[:, 3]
            + 5.0 * x[:, 4]
        ),
    )


def _nqueens_prefix(_seed, _n_train, _n_test):
    n_train = np.arange(8, 25, dtype=np.float64)
    n_test = np.arange(25, 28, dtype=np.float64)

    def features(n):
        return np.column_stack((n, n % 6.0, n % 2.0))

    # NQUEENS_COUNTS[n] uses the conventional n=0 origin.
    return (
        features(n_train),
        NQUEENS_COUNTS[n_train.astype(int)],
        features(n_test),
        NQUEENS_COUNTS[n_test.astype(int)],
    )


PROBLEMS = {
    problem.key: problem
    for problem in (
        ScientificProblem("nguyen-1", "Nguyen-1 extrapolation", 1, False, _nguyen_1),
        ScientificProblem("nguyen-5", "Nguyen-5 extrapolation", 1, False, _nguyen_5),
        ScientificProblem("feynman-gaussian", "Feynman I.6.2 Gaussian", 1, False, _feynman_gaussian),
        ScientificProblem("feynman-product", "Feynman product law", 2, False, _feynman_product),
        ScientificProblem("friedman-1", "Friedman-1 (5 variables)", 5, False, _friedman_1),
        ScientificProblem("nqueens-a000170", "A000170 prefix -> n=25..27", 3, True, _nqueens_prefix),
    )
}


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_engine(args, problem: ScientificProblem) -> None:
    GpuGlobals.POP_SIZE = int(args.pop_size)
    if hasattr(GpuGlobals, "POPULATION_SIZE"):
        GpuGlobals.POPULATION_SIZE = int(args.pop_size)
    GpuGlobals.NUM_ISLANDS = int(args.islands)
    GpuGlobals.GENERATIONS = int(args.generations)
    GpuGlobals.PROGRESS_REPORT_INTERVAL = max(1_000_000, int(args.generations) + 1)
    GpuGlobals.USE_INITIAL_FORMULA = False
    GpuGlobals.USE_STRUCTURAL_SEEDS = False
    GpuGlobals.USE_PATTERN_SEEDS = False
    GpuGlobals.USE_SNIPER = False
    GpuGlobals.USE_RESIDUAL_BOOSTING = False
    GpuGlobals.USE_INITIAL_POP_CACHE = False
    # These experimental paths are disabled until their parity properties are
    # established; the benchmark must not reward corrupted fitness.
    GpuGlobals.USE_LIBRARY_LEARNING = False
    GpuGlobals.USE_BFGS_OPTIMIZER = False
    GpuGlobals.USE_SIMPLIFICATION = False

    general_profile = problem.key != "nqueens-a000170"
    operator_flags = {
        "USE_OP_PLUS": True,
        "USE_OP_MINUS": True,
        "USE_OP_MULT": True,
        "USE_OP_DIV": True,
        "USE_OP_POW": True,
        "USE_OP_MOD": False,
        "USE_OP_SIN": general_profile,
        "USE_OP_COS": general_profile,
        "USE_OP_TAN": False,
        "USE_OP_LOG": True,
        "USE_OP_EXP": True,
        "USE_OP_FACT": not general_profile,
        "USE_OP_FLOOR": False,
        "USE_OP_GAMMA": not general_profile,
        "USE_OP_LGAMMA": not general_profile,
        "USE_OP_ASIN": False,
        "USE_OP_ACOS": False,
        "USE_OP_ATAN": False,
        "USE_OP_CEIL": False,
        "USE_OP_SIGN": False,
        "USE_OP_SQRT": True,
        "USE_OP_ABS": general_profile,
    }
    for name, enabled in operator_flags.items():
        setattr(GpuGlobals, name, enabled)


def _formula_complexity(formula: str | None) -> int | None:
    if not formula:
        return None
    expression = re.sub(r"(?<!\*)\^(?!\*)", "**", formula)
    try:
        parsed = ast.parse(expression, mode="eval")
        relevant = (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)
        return sum(isinstance(node, relevant) for node in ast.walk(parsed))
    except Exception:
        return None


def _evaluate_formula(formula: str | None, x: np.ndarray) -> np.ndarray:
    if not formula:
        return np.full(x.shape[0], np.nan)
    expression = re.sub(r"(?<!\*)\^(?!\*)", "**", formula)
    variables = {f"x{i}": x[:, i] for i in range(x.shape[1])}
    environment = {
        **variables,
        "x": x[:, 0],
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "floor": np.floor,
        "ceil": np.ceil,
        "sign": np.sign,
        "gamma": special.gamma,
        "lgamma": special.gammaln,
        "fact": lambda value: special.gamma(value + 1.0),
        "pi": np.pi,
        "e": np.e,
    }
    with np.errstate(all="ignore"):
        prediction = eval(expression, {"__builtins__": {}}, environment)
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.ndim == 0:
        prediction = np.full(x.shape[0], float(prediction))
    return np.ravel(prediction)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_pred.size != y_true.size:
        return {
            "rmse": None, "nrmse": None, "r2": None,
            "relative_mae": None, "log_rmse": None, "invalid_fraction": 1.0,
        }

    valid = np.isfinite(y_pred)
    invalid_fraction = float(1.0 - valid.mean())
    if not valid.all():
        return {
            "rmse": None, "nrmse": None, "r2": None,
            "relative_mae": None, "log_rmse": None,
            "invalid_fraction": invalid_fraction,
        }

    residual = y_pred - y_true
    rmse = float(np.sqrt(np.mean(residual * residual)))
    scale = float(np.std(y_true))
    nrmse = rmse / max(scale, 1e-12)
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - float(np.sum(residual * residual)) / max(sst, 1e-24)
    relative_mae = float(np.mean(np.abs(residual) / np.maximum(np.abs(y_true), 1e-12)))
    if np.all(y_true > 0.0) and np.all(y_pred > 0.0):
        log_rmse = float(np.sqrt(np.mean((np.log(y_pred) - np.log(y_true)) ** 2)))
    else:
        log_rmse = None
    return {
        "rmse": rmse,
        "nrmse": nrmse,
        "r2": r2,
        "relative_mae": relative_mae,
        "log_rmse": log_rmse,
        "invalid_fraction": invalid_fraction,
    }


def _run_alphasymbolic(problem, x_train, y_train, x_test, y_test, args, seed):
    _seed_everything(seed)
    _configure_engine(args, problem)
    engine = TensorGeneticEngine(
        num_variables=problem.n_variables,
        pop_size=int(args.pop_size),
        n_islands=int(args.islands),
        max_len=int(args.max_len),
        max_constants=int(args.max_constants),
    )
    started = time.perf_counter()
    try:
        formula = engine.run(
            x_train if problem.n_variables > 1 else x_train[:, 0],
            y_train,
            seeds=[],
            timeout_sec=float(args.timeout_sec),
            use_log=problem.use_log,
        )
        elapsed = time.perf_counter() - started
        train_prediction = _evaluate_formula(formula, x_train)
        test_prediction = _evaluate_formula(formula, x_test)
        return {
            "method": "alphasymbolic",
            "formula": formula,
            "complexity": _formula_complexity(formula),
            "elapsed_sec": elapsed,
            "generations_completed": int(getattr(engine, "last_run_generations", 0)),
            "training_objective_rmse": float(
                getattr(engine, "last_run_best_rmse", float("inf"))
            ),
            "effective_log_transform": bool(
                getattr(engine, "last_run_used_log_transform", problem.use_log)
            ),
            "train": _metrics(y_train, train_prediction),
            "test": _metrics(y_test, test_prediction),
            "telemetry": getattr(engine, "last_run_metrics", {}),
            "error": None,
        }
    except Exception as exc:
        return {
            "method": "alphasymbolic",
            "formula": None,
            "complexity": None,
            "elapsed_sec": time.perf_counter() - started,
            "generations_completed": int(getattr(engine, "last_run_generations", 0)),
            "training_objective_rmse": None,
            "effective_log_transform": problem.use_log,
            "train": _metrics(y_train, np.full_like(y_train, np.nan)),
            "test": _metrics(y_test, np.full_like(y_test, np.nan)),
            "telemetry": getattr(engine, "last_run_metrics", {}),
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_pysr(problem, x_train, y_train, x_test, y_test, args, seed):
    started = time.perf_counter()
    try:
        from pysr import PySRRegressor

        fit_target = np.log(y_train) if problem.use_log else y_train
        model = PySRRegressor(
            niterations=int(args.pysr_iterations),
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["sin", "cos", "exp", "log", "sqrt", "abs"],
            populations=max(1, int(args.islands)),
            population_size=max(16, min(256, int(args.pop_size) // max(1, int(args.islands)))),
            maxsize=int(args.max_len),
            timeout_in_seconds=float(args.timeout_sec),
            random_state=int(seed),
            deterministic=True,
            parallelism="serial",
            progress=False,
            verbosity=0,
        )
        model.fit(x_train, fit_target, variable_names=[f"x{i}" for i in range(problem.n_variables)])
        raw_train = np.asarray(model.predict(x_train), dtype=np.float64)
        raw_test = np.asarray(model.predict(x_test), dtype=np.float64)
        train_prediction = np.exp(raw_train) if problem.use_log else raw_train
        test_prediction = np.exp(raw_test) if problem.use_log else raw_test
        best = model.get_best()
        raw_formula = str(best["equation"])
        formula = f"exp({raw_formula})" if problem.use_log else raw_formula
        return {
            "method": "pysr",
            "formula": formula,
            "complexity": int(best["complexity"]),
            "elapsed_sec": time.perf_counter() - started,
            "generations_completed": None,
            "training_objective_rmse": float(
                np.sqrt(np.mean((raw_train - fit_target) ** 2))
            ),
            "effective_log_transform": problem.use_log,
            "train": _metrics(y_train, train_prediction),
            "test": _metrics(y_test, test_prediction),
            "telemetry": {},
            "error": None,
        }
    except Exception as exc:
        return {
            "method": "pysr",
            "formula": None,
            "complexity": None,
            "elapsed_sec": time.perf_counter() - started,
            "generations_completed": None,
            "training_objective_rmse": None,
            "effective_log_transform": problem.use_log,
            "train": _metrics(y_train, np.full_like(y_train, np.nan)),
            "test": _metrics(y_test, np.full_like(y_test, np.nan)),
            "telemetry": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


RUNNERS = {"alphasymbolic": _run_alphasymbolic, "pysr": _run_pysr}


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _json_safe(value):
    """Convert tensors/numpy scalars and non-finite floats to strict JSON."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _append_jsonl(output: Path, row: dict) -> None:
    """Persist each completed run immediately so interruptions keep evidence."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n"
        )
        handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=["smoke", "general", "all"], default="smoke",
        help="smoke=Nguyen-1+A000170; general=the five regression problems.",
    )
    parser.add_argument("--problems", default=None, help="Comma-separated problem keys.")
    parser.add_argument("--methods", default="alphasymbolic", help="alphasymbolic,pysr")
    parser.add_argument("--seeds", default="4200,4201,4202")
    parser.add_argument("--train-points", type=int, default=128)
    parser.add_argument("--test-points", type=int, default=512)
    parser.add_argument("--pop-size", type=int, default=100_000)
    parser.add_argument("--islands", type=int, default=20)
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--max-len", type=int, default=48)
    parser.add_argument("--max-constants", type=int, default=10)
    parser.add_argument("--pysr-iterations", type=int, default=100)
    parser.add_argument(
        "--output", default="benchmarks/scientific_metrics.jsonl",
        help="Append-only JSONL path.",
    )
    args = parser.parse_args()

    if args.problems:
        selected_problems = _parse_csv(args.problems)
    elif args.suite == "smoke":
        selected_problems = ["nguyen-1", "nqueens-a000170"]
    elif args.suite == "general":
        selected_problems = [key for key in PROBLEMS if key != "nqueens-a000170"]
    else:
        selected_problems = list(PROBLEMS)

    methods = _parse_csv(args.methods)
    seeds = [int(seed) for seed in _parse_csv(args.seeds)]
    unknown_problems = sorted(set(selected_problems) - set(PROBLEMS))
    unknown_methods = sorted(set(methods) - set(RUNNERS))
    if unknown_problems or unknown_methods:
        parser.error(f"unknown problems={unknown_problems} methods={unknown_methods}")

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_value("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain") not in ("", "unknown")),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda": torch.version.cuda,
        "track": "general_no_templates",
    }

    output = Path(args.output)
    for problem_key in selected_problems:
        problem = PROBLEMS[problem_key]
        for seed in seeds:
            x_train, y_train, x_test, y_test = problem.make_data(
                seed, int(args.train_points), int(args.test_points)
            )
            for method in methods:
                result = RUNNERS[method](
                    problem, x_train, y_train, x_test, y_test, args, seed
                )
                row = {
                    **metadata,
                    "record_type": "scientific_run",
                    "problem": problem.key,
                    "problem_label": problem.label,
                    "n_variables": problem.n_variables,
                    "seed": seed,
                    "train_points": int(len(y_train)),
                    "test_points": int(len(y_test)),
                    "test_protocol": (
                        "prefix_train_8_24_holdout_25_27"
                        if problem.key == "nqueens-a000170"
                        else "independent_holdout"
                    ),
                    "config": {
                        "pop_size": int(args.pop_size),
                        "islands": int(args.islands),
                        "generations": int(args.generations),
                        "timeout_sec": float(args.timeout_sec),
                        "max_len": int(args.max_len),
                        "max_constants": int(args.max_constants),
                    },
                    **result,
                }
                _append_jsonl(output, row)
                test_nrmse = row["test"]["nrmse"]
                print(
                    f"{method:13s} {problem.key:20s} seed={seed} "
                    f"test_nrmse={test_nrmse if test_nrmse is not None else 'invalid'} "
                    f"invalid={row['test']['invalid_fraction']:.3f} "
                    f"elapsed={row['elapsed_sec']:.3f}s"
                )

    print(f"wrote={output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
