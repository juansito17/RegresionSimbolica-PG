"""Scikit-learn and SRBench adapter for the CUDA symbolic-regression engine.

The CUDA engine is intentionally created inside :meth:`fit`.  Keeping
``__init__`` side-effect free is required by scikit-learn cloning and prevents
an imported SRBench method from reserving GPU memory before an experiment
starts.
"""

from __future__ import annotations

import ast
import contextlib
import gc
import math
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import sympy
from scipy import special
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted


_ENGINE_LOCK = threading.RLock()
_VARIABLE_RE = re.compile(r"x(\d+)\Z")


def _load_engine_class():
    """Load the heavy CUDA engine lazily (also a narrow unit-test seam)."""

    from AlphaSymbolic.core.gpu.engine import TensorGeneticEngine

    return TensorGeneticEngine


def _normalise_formula(formula: str) -> str:
    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    formula = formula.strip()
    if not formula or formula.lower() in {"none", "invalid", "nan"}:
        raise ValueError("the engine did not return a usable formula")
    if len(formula) > 1_000_000:
        raise ValueError("formula is unreasonably large")
    return re.sub(r"(?<!\*)\^(?!\*)", "**", formula)


def _parse_formula(formula: str) -> ast.Expression:
    expression = ast.parse(_normalise_formula(formula), mode="eval")
    if sum(1 for _ in ast.walk(expression)) > 10_000:
        raise ValueError("formula contains too many syntax nodes")
    return expression


class _FormulaInterpreter:
    """Interpret a small expression language without Python ``eval``."""

    _binary_numeric = {
        ast.Add: np.add,
        ast.Sub: np.subtract,
        ast.Mult: np.multiply,
        ast.Div: np.divide,
        ast.Pow: np.power,
        ast.Mod: np.mod,
    }
    _binary_symbolic = {
        ast.Add: lambda a, b: sympy.Add(a, b, evaluate=False),
        ast.Sub: lambda a, b: sympy.Add(
            a, sympy.Mul(-1, b, evaluate=False), evaluate=False
        ),
        ast.Mult: lambda a, b: sympy.Mul(a, b, evaluate=False),
        ast.Div: lambda a, b: sympy.Mul(
            a, sympy.Pow(b, -1, evaluate=False), evaluate=False
        ),
        ast.Pow: lambda a, b: sympy.Pow(a, b, evaluate=False),
        ast.Mod: lambda a, b: sympy.Mod(a, b, evaluate=False),
    }
    _numeric_functions = {
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
        "neg": np.negative,
        "pow": np.power,
    }
    _symbolic_functions = {
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
        "asin": sympy.asin,
        "acos": sympy.acos,
        "atan": sympy.atan,
        "exp": sympy.exp,
        "log": sympy.log,
        "sqrt": sympy.sqrt,
        "abs": sympy.Abs,
        "floor": sympy.floor,
        "ceil": sympy.ceiling,
        "sign": sympy.sign,
        "gamma": sympy.gamma,
        "lgamma": sympy.loggamma,
        "fact": sympy.factorial,
        "neg": lambda value: -value,
        "pow": lambda a, b: a**b,
    }

    def __init__(self, variables: Sequence[Any], symbolic: bool):
        self.variables = list(variables)
        self.symbolic = symbolic

    def visit(self, node: ast.AST):
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise ValueError(f"unsupported formula syntax: {type(node).__name__}")
        return method(node)

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only real numeric constants are supported")
        if not math.isfinite(float(node.value)):
            raise ValueError("non-finite constants are not supported")
        if self.symbolic:
            return (
                sympy.Integer(node.value)
                if isinstance(node.value, int)
                else sympy.Float(node.value)
            )
        return float(node.value)

    def visit_Name(self, node: ast.Name):
        match = _VARIABLE_RE.fullmatch(node.id)
        if match:
            index = int(match.group(1))
            if index >= len(self.variables):
                raise ValueError(
                    f"formula refers to x{index}, but only "
                    f"{len(self.variables)} variables were fitted"
                )
            return self.variables[index]
        if node.id == "x" and self.variables:
            return self.variables[0]
        constants = (
            {"pi": sympy.pi, "e": sympy.E, "E": sympy.E}
            if self.symbolic
            else {"pi": np.pi, "e": np.e, "E": np.e}
        )
        if node.id in constants:
            return constants[node.id]
        raise ValueError(f"unknown name in formula: {node.id!r}")

    def visit_UnaryOp(self, node: ast.UnaryOp):
        value = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")

    def visit_BinOp(self, node: ast.BinOp):
        operations = self._binary_symbolic if self.symbolic else self._binary_numeric
        operation = operations.get(type(node.op))
        if operation is None:
            raise ValueError(f"unsupported binary operator: {type(node.op).__name__}")
        return operation(self.visit(node.left), self.visit(node.right))

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError("only direct calls to approved functions are supported")
        functions = self._symbolic_functions if self.symbolic else self._numeric_functions
        function = functions.get(node.func.id)
        if function is None:
            raise ValueError(f"unsupported formula function: {node.func.id!r}")
        return function(*(self.visit(arg) for arg in node.args))


def evaluate_formula(formula: str, X: np.ndarray) -> np.ndarray:
    """Evaluate an engine formula safely on a samples-by-features matrix."""

    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("X must be a two-dimensional samples-by-features matrix")
    tree = _parse_formula(formula)
    variables = [values[:, index] for index in range(values.shape[1])]
    with np.errstate(all="ignore"):
        prediction = _FormulaInterpreter(variables, symbolic=False).visit(tree)
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.ndim == 0:
        prediction = np.full(values.shape[0], float(prediction), dtype=np.float64)
    prediction = prediction.reshape(-1)
    if prediction.size != values.shape[0]:
        raise ValueError(
            f"formula produced {prediction.size} values for {values.shape[0]} samples"
        )
    return prediction


def formula_to_sympy(formula: str, feature_names: Sequence[str]) -> sympy.Expr:
    """Convert an engine formula to a SymPy expression with external names."""

    names = [str(name) for name in feature_names]
    symbols = [sympy.Symbol(name, real=True) for name in names]
    # Keep the parsed tree as-is.  A full ``sympy.simplify`` on every
    # validation candidate dominated runtime for degree-3 portfolios (tens of
    # seconds for 8 variables) without changing predictions or SRBench's node
    # count semantics.
    return _FormulaInterpreter(symbols, symbolic=True).visit(_parse_formula(formula))


def _formula_with_feature_names(
    formula: str, feature_names: Sequence[str]
) -> str:
    """Rename local ``xN`` symbols without invoking expensive CAS routines."""

    names = [str(name) for name in feature_names]

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if index >= len(names):
            raise ValueError(
                f"formula references x{index}, but only {len(names)} variables exist"
            )
        return names[index]

    return re.sub(r"\bx(\d+)\b", replace, _normalise_formula(formula))


def _format_number(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("cannot serialize a non-finite fallback coefficient")
    if abs(value) < 1e-15:
        value = 0.0
    return format(value, ".17g")


def _linear_fallback_formula(
    X: np.ndarray, y: np.ndarray, strategy: str
) -> str:
    mean_target = float(np.mean(y))
    if strategy == "mean" or X.shape[1] == 0:
        return _format_number(mean_target)

    means = np.mean(X, axis=0)
    scales = np.std(X, axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardised = (X - means) / scales
    design = np.column_stack((np.ones(X.shape[0]), standardised))
    try:
        parameters, *_ = np.linalg.lstsq(design, y, rcond=None)
        coefficients = parameters[1:] / scales
        intercept = float(parameters[0] - np.dot(coefficients, means))
        if not np.isfinite(coefficients).all() or not math.isfinite(intercept):
            raise ValueError("linear fallback coefficients are not finite")
    except (np.linalg.LinAlgError, ValueError):
        return _format_number(mean_target)

    terms = [f"({_format_number(intercept)})"]
    terms.extend(
        f"({_format_number(coefficient)})*x{index}"
        for index, coefficient in enumerate(coefficients)
        if abs(float(coefficient)) >= 1e-15
    )
    return " + ".join(terms)


def _polynomial_formula(
    X: np.ndarray,
    y: np.ndarray,
    degree: int,
    ridge_alpha: float,
) -> str:
    """Fit a deterministic polynomial in scaled coordinates and unscale it."""

    means = np.mean(X, axis=0)
    scales = np.std(X, axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardised = (X - means) / scales
    expansion = PolynomialFeatures(degree=int(degree), include_bias=False)
    design = expansion.fit_transform(standardised)
    # Cholesky is deterministic and was ~10x faster than SVD on the largest
    # pinned SRBench training matrix while producing the same RMSE.  Retain
    # SVD only for the unregularized edge case where the Gram matrix may be
    # singular.
    ridge = Ridge(
        alpha=float(ridge_alpha),
        fit_intercept=True,
        solver="cholesky" if float(ridge_alpha) > 0.0 else "svd",
    )
    ridge.fit(design, y)

    scaled_symbols = [
        f"((x{index}-({_format_number(mean)}))/({_format_number(scale)}))"
        for index, (mean, scale) in enumerate(zip(means, scales))
    ]
    terms = [f"({_format_number(float(ridge.intercept_))})"]
    for coefficient, powers in zip(ridge.coef_, expansion.powers_):
        coefficient = float(coefficient)
        if not math.isfinite(coefficient) or abs(coefficient) < 1e-13:
            continue
        factors = [f"({_format_number(coefficient)})"]
        for symbol, power in zip(scaled_symbols, powers):
            if int(power):
                factors.append(
                    symbol if int(power) == 1 else f"({symbol}**{int(power)})"
                )
        terms.append("*".join(factors))
    return " + ".join(terms)


def _normalised_score(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    maximum = float(np.max(values)) if values.size else 0.0
    return values / maximum if maximum > 0.0 else np.zeros_like(values)


def _sparse_polynomial_formula(
    X: np.ndarray,
    y: np.ndarray,
    degree: int,
    ridge_alpha: float,
    max_terms: int = 24,
) -> str:
    """Fit and prune a polynomial before serialising one symbolic model."""

    expansion = PolynomialFeatures(degree=int(degree), include_bias=False)
    raw_design = expansion.fit_transform(X)
    design_means = np.mean(raw_design, axis=0)
    design_scales = np.std(raw_design, axis=0)
    design_scales = np.where(design_scales > 1e-12, design_scales, 1.0)
    design = (raw_design - design_means) / design_scales
    ridge = Ridge(
        alpha=max(float(ridge_alpha), 1e-12),
        fit_intercept=True,
        solver="cholesky",
    ).fit(design, y)
    importance = np.abs(np.asarray(ridge.coef_, dtype=np.float64)) * np.std(
        design, axis=0
    )
    relative_floor = max(
        1e-10,
        float(np.max(importance[np.isfinite(importance)])) * 1e-6
        if np.isfinite(importance).any()
        else 1e-10,
    )
    finite = np.isfinite(importance) & (importance > relative_floor)
    selected = np.flatnonzero(finite)
    if selected.size > int(max_terms):
        ranking = np.lexsort((selected, -importance[selected]))
        selected = np.sort(selected[ranking[: int(max_terms)]])
    if not selected.size:
        return _format_number(float(np.mean(y)))
    refit = Ridge(
        alpha=max(float(ridge_alpha), 1e-12),
        fit_intercept=True,
        solver="cholesky",
    ).fit(design[:, selected], y)
    raw_coefficients = np.asarray(refit.coef_, dtype=np.float64) / design_scales[selected]
    raw_intercept = float(
        refit.intercept_ - np.dot(raw_coefficients, design_means[selected])
    )
    terms = [f"({_format_number(raw_intercept)})"]
    for coefficient, term_index in zip(raw_coefficients, selected):
        coefficient = float(coefficient)
        if not math.isfinite(coefficient) or abs(coefficient) < 1e-12:
            continue
        factors = [f"({_format_number(coefficient)})"]
        for variable_index, power in enumerate(expansion.powers_[term_index]):
            if int(power):
                symbol = f"x{variable_index}"
                factors.append(symbol if int(power) == 1 else f"({symbol}**{int(power)})")
        terms.append("*".join(factors))
    return " + ".join(terms)


@dataclass
class _AdaptiveCandidate:
    name: str
    family: str
    formula: str
    indices: np.ndarray
    iid_scores: list[float]
    boundary_score: float
    score: float
    score_se: float
    mdl: float
    complexity: int
    elapsed_sec: float
    degree: Optional[int] = None
    valid: bool = True

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "formula": self.formula,
            "feature_indices": [int(value) for value in self.indices],
            "fold_scores": [float(value) for value in self.iid_scores],
            "boundary_score": float(self.boundary_score),
            "score": float(self.score),
            "score_se": float(self.score_se),
            "mdl": float(self.mdl),
            "complexity": int(self.complexity),
            "elapsed_sec": float(self.elapsed_sec),
            "degree": self.degree,
            "valid": bool(self.valid),
        }


@contextlib.contextmanager
def _seeded_engine_runtime(
    seed: Optional[int],
    generations: Optional[int],
    overrides: Optional[dict[str, Any]] = None,
):
    """Serialize access to global engine state and restore RNG/configuration."""

    with _ENGINE_LOCK:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_module = None
        torch_state = None
        cuda_states = None
        globals_class = None
        previous_globals: dict[str, Any] = {}
        try:
            import torch

            torch_module = torch
            torch_state = torch.random.get_rng_state()
            if torch.cuda.is_available():
                cuda_states = torch.cuda.get_rng_state_all()
        except Exception:
            torch_module = None

        if generations is not None or overrides:
            from AlphaSymbolic.core.gpu.config import GpuGlobals

            globals_class = GpuGlobals
            if generations is not None:
                previous_globals["GENERATIONS"] = GpuGlobals.GENERATIONS
                GpuGlobals.GENERATIONS = int(generations)
            for name, value in (overrides or {}).items():
                if not hasattr(GpuGlobals, name):
                    raise ValueError(f"unknown engine option: {name}")
                previous_globals[name] = getattr(GpuGlobals, name)
                setattr(GpuGlobals, name, value)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            if torch_module is not None:
                torch_module.manual_seed(seed)
                if torch_module.cuda.is_available():
                    torch_module.cuda.manual_seed_all(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            if globals_class is not None:
                for name, value in previous_globals.items():
                    setattr(globals_class, name, value)
            if torch_module is not None and torch_state is not None:
                torch_module.random.set_rng_state(torch_state)
                if cuda_states is not None:
                    torch_module.cuda.set_rng_state_all(cuda_states)


def _release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class AlphaSymbolicRegressor(RegressorMixin, BaseEstimator):
    """Scikit-learn compatible facade for :class:`TensorGeneticEngine`.

    Parameters controlling ``max_gpu_variables`` and ``max_gpu_samples`` keep
    the default workload inside the current fused-kernel envelope (4 variables,
    1024 samples).  Feature ranking is fitted once and reused verbatim by
    :meth:`predict`.
    """

    def __init__(
        self,
        *,
        pop_size: int = 100_000,
        n_islands: int = 20,
        max_len: int = 48,
        max_constants: int = 10,
        max_gpu_variables: Optional[int] = 4,
        max_gpu_samples: Optional[int] = 1024,
        feature_selection: str = "hybrid",
        fallback_strategy: str = "linear",
        polynomial_degree: int = 3,
        max_polynomial_variables: Optional[int] = 8,
        ridge_alpha: float = 1e-6,
        validation_fraction: float = 0.2,
        generations: Optional[int] = 150,
        max_time: float = 3600.0,
        random_state: Optional[int] = 0,
        device: Optional[str] = None,
        use_log: bool = False,
        search_mode: str = "legacy",
        target_transform: str = "auto",
        max_active_variables: int = 8,
        sparse_polynomial_terms: int = 12,
    ):
        self.pop_size = pop_size
        self.n_islands = n_islands
        self.max_len = max_len
        self.max_constants = max_constants
        self.max_gpu_variables = max_gpu_variables
        self.max_gpu_samples = max_gpu_samples
        self.feature_selection = feature_selection
        self.fallback_strategy = fallback_strategy
        self.polynomial_degree = polynomial_degree
        self.max_polynomial_variables = max_polynomial_variables
        self.ridge_alpha = ridge_alpha
        self.validation_fraction = validation_fraction
        self.generations = generations
        self.max_time = max_time
        self.random_state = random_state
        self.device = device
        self.use_log = use_log
        self.search_mode = search_mode
        self.target_transform = target_transform
        self.max_active_variables = max_active_variables
        self.sparse_polynomial_terms = sparse_polynomial_terms

    def _validate_parameters(self) -> None:
        for name in ("pop_size", "n_islands", "max_len", "max_constants"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_gpu_variables is not None and int(self.max_gpu_variables) <= 0:
            raise ValueError("max_gpu_variables must be positive or None")
        if self.max_gpu_samples is not None and int(self.max_gpu_samples) <= 0:
            raise ValueError("max_gpu_samples must be positive or None")
        if not 0 <= int(self.polynomial_degree) <= 3:
            raise ValueError("polynomial_degree must be between 0 and 3")
        if (
            self.max_polynomial_variables is not None
            and int(self.max_polynomial_variables) <= 0
        ):
            raise ValueError("max_polynomial_variables must be positive or None")
        if float(self.ridge_alpha) < 0:
            raise ValueError("ridge_alpha must be non-negative")
        if self.generations is not None and int(self.generations) <= 0:
            raise ValueError("generations must be positive or None")
        if self.max_time is None or float(self.max_time) <= 0:
            raise ValueError("max_time must be positive")
        if self.feature_selection not in {"hybrid", "correlation", "mutual_info"}:
            raise ValueError(
                "feature_selection must be 'hybrid', 'correlation', or "
                "'mutual_info'"
            )
        if self.fallback_strategy not in {"linear", "mean"}:
            raise ValueError("fallback_strategy must be 'linear' or 'mean'")
        if not 0.0 <= float(self.validation_fraction) < 0.5:
            raise ValueError("validation_fraction must be in [0, 0.5)")
        if self.search_mode not in {"adaptive", "legacy"}:
            raise ValueError("search_mode must be 'adaptive' or 'legacy'")
        if self.target_transform not in {"auto", "identity", "log"}:
            raise ValueError(
                "target_transform must be 'auto', 'identity', or 'log'"
            )
        if int(self.max_active_variables) <= 0 or int(self.max_active_variables) > 8:
            raise ValueError("max_active_variables must be between 1 and 8")
        if int(self.sparse_polynomial_terms) <= 0:
            raise ValueError("sparse_polynomial_terms must be positive")

    @staticmethod
    def _is_dataframe(X: Any) -> bool:
        return all(hasattr(X, attribute) for attribute in ("columns", "loc", "to_numpy"))

    def _coerce_fit_data(self, X: Any, y: Any):
        if self._is_dataframe(X):
            if bool(getattr(X.columns, "has_duplicates", False)):
                raise ValueError("X contains duplicate DataFrame column names")
            input_columns = tuple(X.columns.tolist())
            feature_names = [str(column) for column in input_columns]
            values = X.to_numpy(dtype=np.float64, copy=True)
            fitted_with_dataframe = True
        else:
            values = np.asarray(X, dtype=np.float64)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            if values.ndim != 2:
                raise ValueError("X must be a 1D/2D array or a DataFrame")
            input_columns = None
            feature_names = [f"x{index}" for index in range(values.shape[1])]
            fitted_with_dataframe = False

        target = np.asarray(y, dtype=np.float64)
        if target.ndim == 2 and target.shape[1] == 1:
            target = target[:, 0]
        if target.ndim != 1:
            raise ValueError("AlphaSymbolicRegressor supports one target only")
        if values.shape[0] != target.shape[0]:
            raise ValueError("X and y contain different numbers of samples")
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature")

        finite_target = np.isfinite(target)
        if not finite_target.any():
            raise ValueError("y contains no finite target values")
        values = values[finite_target]
        target = target[finite_target]
        return (
            values,
            target,
            input_columns,
            feature_names,
            fitted_with_dataframe,
        )

    @staticmethod
    def _fit_imputation(X: np.ndarray):
        clean = np.asarray(X, dtype=np.float64).copy()
        clean[~np.isfinite(clean)] = np.nan
        fill_values = np.zeros(clean.shape[1], dtype=np.float64)
        for index in range(clean.shape[1]):
            finite = np.isfinite(clean[:, index])
            fill_values[index] = (
                float(np.median(clean[finite, index])) if finite.any() else 0.0
            )
            clean[~finite, index] = fill_values[index]
        return clean, fill_values

    @staticmethod
    def _apply_imputation(X: np.ndarray, fill_values: np.ndarray) -> np.ndarray:
        clean = np.asarray(X, dtype=np.float64).copy()
        invalid = ~np.isfinite(clean)
        if invalid.any():
            rows, columns = np.nonzero(invalid)
            clean[rows, columns] = fill_values[columns]
        return clean

    def _seed(self) -> Optional[int]:
        if self.random_state is None:
            return None
        if isinstance(self.random_state, (int, np.integer)):
            return int(self.random_state) % (2**32 - 1)
        return int(check_random_state(self.random_state).randint(0, 2**31 - 1))

    def _select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: Optional[int],
        limit: Optional[int],
    ) -> np.ndarray:
        n_features = X.shape[1]
        limit = (
            n_features
            if limit is None
            else min(n_features, int(limit))
        )
        if limit == n_features:
            return np.arange(n_features, dtype=np.int64)

        centered_x = X - np.mean(X, axis=0)
        centered_y = y - np.mean(y)
        denominator = np.linalg.norm(centered_x, axis=0) * np.linalg.norm(centered_y)
        numerator = np.abs(centered_x.T @ centered_y)
        correlations = np.divide(
            numerator,
            denominator,
            out=np.zeros(n_features, dtype=np.float64),
            where=denominator > 1e-15,
        )

        information = np.zeros(n_features, dtype=np.float64)
        if self.feature_selection in {"hybrid", "mutual_info"} and X.shape[0] >= 4:
            try:
                information = mutual_info_regression(
                    X,
                    y,
                    random_state=seed,
                    n_neighbors=min(3, X.shape[0] - 1),
                )
            except (ValueError, FloatingPointError):
                information = np.zeros(n_features, dtype=np.float64)

        if self.feature_selection == "correlation":
            scores = correlations
        elif self.feature_selection == "mutual_info":
            scores = information
        else:
            scores = np.maximum(
                _normalised_score(correlations), _normalised_score(information)
            )
        ranking = np.lexsort((np.arange(n_features), -scores))
        return np.sort(ranking[:limit]).astype(np.int64)

    def _gpu_sample_indices(self, n_samples: int, seed: Optional[int]) -> np.ndarray:
        if self.max_gpu_samples is None or n_samples <= int(self.max_gpu_samples):
            return np.arange(n_samples, dtype=np.int64)
        rng = check_random_state(seed)
        return np.sort(
            rng.choice(n_samples, size=int(self.max_gpu_samples), replace=False)
        ).astype(np.int64)

    def _validation_split(self, n_samples: int, seed: Optional[int]):
        if float(self.validation_fraction) == 0.0 or n_samples < 4:
            indices = np.arange(n_samples, dtype=np.int64)
            return indices, indices
        validation_size = max(
            1, int(round(n_samples * float(self.validation_fraction)))
        )
        validation_size = min(validation_size, n_samples - 2)
        permutation = check_random_state(seed).permutation(n_samples)
        validation = np.sort(permutation[:validation_size]).astype(np.int64)
        training = np.sort(permutation[validation_size:]).astype(np.int64)
        return training, validation

    def _coerce_predict_data(self, X: Any) -> np.ndarray:
        if self._is_dataframe(X) and self._fitted_with_dataframe_:
            missing = [
                column for column in self._input_columns_ if column not in X.columns
            ]
            if missing:
                raise ValueError(
                    "X is missing fitted columns: "
                    + ", ".join(repr(column) for column in missing[:5])
                )
            values = X.loc[:, list(self._input_columns_)].to_numpy(
                dtype=np.float64, copy=True
            )
        else:
            values = np.asarray(X, dtype=np.float64)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have {self.n_features_in_} features; got shape "
                f"{getattr(values, 'shape', None)}"
            )
        invalid = ~np.isfinite(values)
        if invalid.any():
            values = values.copy()
            rows, columns = np.nonzero(invalid)
            values[rows, columns] = self.feature_fill_values_[columns]
        return values

    def fit(self, X: Any, y: Any):
        """Fit exactly one expression using the selected search protocol."""

        if self.search_mode == "adaptive":
            from AlphaSymbolic.adaptive_search import fit_adaptive

            return fit_adaptive(self, X, y)
        return self._fit_legacy(X, y)

    def _fit_legacy(self, X: Any, y: Any):
        """Fit one symbolic expression and retain a deterministic safe fallback."""

        self._validate_parameters()
        (
            values,
            target,
            input_columns,
            feature_names,
            fitted_with_dataframe,
        ) = self._coerce_fit_data(X, y)
        seed = self._seed()
        training_indices, validation_indices = self._validation_split(
            values.shape[0], seed
        )
        clean_training, fill_values = self._fit_imputation(values[training_indices])
        values = self._apply_imputation(values, fill_values)
        training_target = target[training_indices]
        validation_target = target[validation_indices]
        engine_selected = self._select_features(
            clean_training,
            training_target,
            seed,
            self.max_gpu_variables,
        )
        polynomial_selected = self._select_features(
            clean_training,
            training_target,
            seed,
            self.max_polynomial_variables,
        )
        fallback_selected = polynomial_selected
        fallback_training_values = values[training_indices][:, fallback_selected]
        fallback_validation_values = values[validation_indices][:, fallback_selected]

        self.n_features_in_ = values.shape[1]
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.feature_fill_values_ = fill_values
        self.engine_feature_indices_ = engine_selected
        self.polynomial_feature_indices_ = polynomial_selected
        self.fallback_feature_indices_ = fallback_selected
        self._input_columns_ = input_columns
        self._fitted_with_dataframe_ = fitted_with_dataframe
        self.n_samples_seen_ = values.shape[0]
        self.validation_size_ = int(validation_indices.size)
        self.random_seed_ = seed
        self.fallback_formula_ = _linear_fallback_formula(
            fallback_training_values, training_target, self.fallback_strategy
        )
        self.engine_error_ = None
        self.selection_reason_ = "engine_not_run"
        self.engine_validation_rmse_ = math.inf
        self.candidate_formula_ = None
        self.polynomial_formula_ = None
        self.polynomial_degree_ = None
        self.polynomial_validation_rmse_ = math.inf
        candidate_records = []

        def register_candidate(name, formula, indices, degree=None):
            parsed = _parse_formula(formula)
            prediction = evaluate_formula(formula, values[:, indices])
            if not np.isfinite(prediction).all():
                raise ValueError(
                    f"{name} formula is non-finite on the complete training set"
                )
            validation_prediction = prediction[validation_indices]
            rmse = float(
                np.sqrt(np.mean((validation_prediction - validation_target) ** 2))
            )
            record = {
                "name": name,
                "formula": formula,
                "indices": np.asarray(indices, dtype=np.int64),
                "degree": degree,
                "rmse": rmse,
                "complexity": sum(1 for _ in ast.walk(parsed)),
            }
            candidate_records.append(record)
            return record

        fallback_record = register_candidate(
            "fallback", self.fallback_formula_, fallback_selected
        )
        self.fallback_validation_rmse_ = fallback_record["rmse"]

        polynomial_records = []
        for degree in range(1, int(self.polynomial_degree) + 1):
            try:
                polynomial_formula = _polynomial_formula(
                    fallback_training_values,
                    training_target,
                    degree,
                    float(self.ridge_alpha),
                )
                polynomial_records.append(
                    register_candidate(
                        "polynomial",
                        polynomial_formula,
                        polynomial_selected,
                        degree=degree,
                    )
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                continue
        if polynomial_records:
            best_polynomial = min(
                polynomial_records,
                key=lambda record: (
                    record["rmse"],
                    record["complexity"],
                    record["degree"],
                ),
            )
            self.polynomial_formula_ = best_polynomial["formula"]
            self.polynomial_degree_ = int(best_polynomial["degree"])
            self.polynomial_validation_rmse_ = best_polynomial["rmse"]

        engine_training_values = values[training_indices][:, engine_selected]
        gpu_indices = self._gpu_sample_indices(
            engine_training_values.shape[0], seed
        )
        self.n_gpu_samples_ = int(gpu_indices.size)
        gpu_values = engine_training_values[gpu_indices]
        gpu_target = training_target[gpu_indices]
        engine = None
        try:
            with _seeded_engine_runtime(seed, self.generations):
                engine_class = _load_engine_class()
                engine_device = self.device
                if isinstance(engine_device, str):
                    import torch

                    engine_device = torch.device(engine_device)
                engine = engine_class(
                    device=engine_device,
                    pop_size=int(self.pop_size),
                    max_len=int(self.max_len),
                    num_variables=int(engine_selected.size),
                    max_constants=int(self.max_constants),
                    n_islands=int(self.n_islands),
                )
                engine_input = (
                    gpu_values[:, 0]
                    if engine_selected.size == 1
                    else gpu_values
                )
                candidate = engine.run(
                    engine_input,
                    gpu_target,
                    seeds=[],
                    timeout_sec=float(self.max_time),
                    use_log=bool(self.use_log),
                )
                candidate = _normalise_formula(candidate)
                self.candidate_formula_ = candidate
                engine_record = register_candidate(
                    "engine",
                    candidate,
                    engine_selected,
                )
                self.engine_validation_rmse_ = engine_record["rmse"]
        except Exception as exc:
            self.engine_error_ = f"{type(exc).__name__}: {exc}"
        finally:
            engine = None
            _release_cuda_memory()

        best_rmse = min(record["rmse"] for record in candidate_records)
        tolerance = max(1e-12, abs(best_rmse) * 1e-9)
        near_best = [
            record
            for record in candidate_records
            if record["rmse"] <= best_rmse + tolerance
        ]
        priority = {"engine": 0, "fallback": 1, "polynomial": 2}
        winner = min(
            near_best,
            key=lambda record: (
                record["complexity"],
                priority[record["name"]],
                record["degree"] or 0,
            ),
        )

        # The internal holdout chooses the model family and polynomial degree.
        # Once that choice is frozen, refit its numeric coefficients on every
        # outer-training row.  This preserves an untouched SRBench test split
        # without permanently discarding 20% of the data supplied to ``fit``.
        self.fallback_formula_ = _linear_fallback_formula(
            values[:, fallback_selected],
            target,
            self.fallback_strategy,
        )
        refit_formula = winner["formula"]
        if winner["name"] == "fallback":
            refit_formula = self.fallback_formula_
        elif winner["name"] == "polynomial":
            refit_formula = _polynomial_formula(
                values[:, winner["indices"]],
                target,
                int(winner["degree"]),
                float(self.ridge_alpha),
            )
            self.polynomial_formula_ = refit_formula
        elif winner["name"] == "engine":
            # The evolutionary search fixes the expression structure on the
            # inner-training partition.  A final affine calibration is the
            # inexpensive analogue of refitting its constants on all rows.
            raw_prediction = evaluate_formula(
                winner["formula"],
                values[:, winner["indices"]],
            )
            design = np.column_stack(
                [raw_prediction, np.ones(raw_prediction.shape[0], dtype=np.float64)]
            )
            coefficient, intercept = np.linalg.lstsq(
                design,
                target,
                rcond=None,
            )[0]
            calibrated = coefficient * raw_prediction + intercept
            raw_rmse = float(np.sqrt(np.mean((raw_prediction - target) ** 2)))
            calibrated_rmse = float(
                np.sqrt(np.mean((calibrated - target) ** 2))
            )
            improvement_tolerance = max(1e-12, raw_rmse * 1e-9)
            if (
                np.isfinite(coefficient)
                and np.isfinite(intercept)
                and calibrated_rmse < raw_rmse - improvement_tolerance
            ):
                refit_formula = (
                    f"({float(coefficient):.17g})*({winner['formula']})"
                    f"+({float(intercept):.17g})"
                )

        self.formula_ = refit_formula
        self.sympy_formula_ = _formula_with_feature_names(
            refit_formula,
            [feature_names[index] for index in winner["indices"]],
        )
        self.fit_status_ = winner["name"]
        self.selection_reason_ = f"{winner['name']}_validation_rmse"
        self.selected_feature_indices_ = winner["indices"]
        self.selected_feature_names_ = np.asarray(
            [feature_names[index] for index in winner["indices"]], dtype=object
        )
        self.validation_candidates_ = [
            {
                key: record[key]
                for key in ("name", "formula", "degree", "rmse", "complexity")
            }
            for record in candidate_records
        ]
        self.symbolic_complexity_ = sum(
            1 for _ in ast.walk(_parse_formula(refit_formula))
        )
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict with the symbolic model, repairing invalid rows via fallback."""

        check_is_fitted(self, "formula_")
        values = self._coerce_predict_data(X)
        selected_values = values[:, self.selected_feature_indices_]
        try:
            prediction = evaluate_formula(self.formula_, selected_values)
        except (TypeError, ValueError, SyntaxError, OverflowError):
            prediction = np.full(selected_values.shape[0], np.nan, dtype=np.float64)

        invalid = ~np.isfinite(prediction)
        if invalid.any() and self.formula_ != self.fallback_formula_:
            fallback = evaluate_formula(
                self.fallback_formula_,
                values[:, self.fallback_feature_indices_],
            )
            prediction = prediction.copy()
            prediction[invalid] = fallback[invalid]
            invalid = ~np.isfinite(prediction)
        if invalid.any():
            prediction = prediction.copy()
            prediction[invalid] = 0.0
        return prediction

    def _external_feature_names(self, X: Any = None) -> list[str]:
        if X is not None and self._is_dataframe(X):
            columns = [str(column) for column in X.columns]
            if len(columns) == self.n_features_in_:
                return [columns[index] for index in self.selected_feature_indices_]
            if len(columns) == len(self.selected_feature_indices_):
                return columns
        return [str(name) for name in self.selected_feature_names_]

    def to_sympy(self, X: Any = None) -> sympy.Expr:
        """Return the selected model as a SymPy expression."""

        check_is_fitted(self, "formula_")
        return formula_to_sympy(self.formula_, self._external_feature_names(X))

    def to_sympy_string(self, X: Any = None) -> str:
        """Return the selected model using the SRBench string protocol."""

        return str(self.to_sympy(X))
