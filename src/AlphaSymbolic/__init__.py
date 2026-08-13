"""Deprecated compatibility namespace for the former AlphaSymbolic package."""

from __future__ import annotations

import warnings

warnings.warn(
    "AlphaSymbolic is deprecated; import from warpsymbolic instead.",
    DeprecationWarning,
    stacklevel=2,
)

from warpsymbolic import (  # noqa: E402
    GpuUnavailableError,
    TensorGeneticEngine,
    WarpSymbolicRegressor,
    evaluate_formula,
    formula_to_sympy,
)

AlphaSymbolicRegressor = WarpSymbolicRegressor

__all__ = [
    "AlphaSymbolicRegressor",
    "WarpSymbolicRegressor",
    "TensorGeneticEngine",
    "GpuUnavailableError",
    "evaluate_formula",
    "formula_to_sympy",
]
