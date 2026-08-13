"""Deprecated compatibility import for the former estimator module."""

from .sklearn import *
from .sklearn import AlphaSymbolicRegressor

__all__ = [
    "AlphaSymbolicRegressor",
    "WarpSymbolicRegressor",
    "evaluate_formula",
    "formula_to_sympy",
]
