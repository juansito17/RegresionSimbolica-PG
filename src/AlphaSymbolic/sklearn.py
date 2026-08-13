"""Deprecated compatibility import for the former estimator module."""

from __future__ import annotations

import warnings

warnings.warn(
    "AlphaSymbolic.sklearn is deprecated; use warpsymbolic.api.sklearn.",
    DeprecationWarning,
    stacklevel=2,
)

from warpsymbolic.api.sklearn import (  # noqa: E402
    WarpSymbolicRegressor,
    evaluate_formula,
    formula_to_sympy,
)

AlphaSymbolicRegressor = WarpSymbolicRegressor

__all__ = [
    "AlphaSymbolicRegressor",
    "WarpSymbolicRegressor",
    "evaluate_formula",
    "formula_to_sympy",
]
