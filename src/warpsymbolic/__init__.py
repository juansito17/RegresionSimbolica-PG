"""Lean public core for GPU-first WarpSymbolic.

Only the estimator/API, symbolic representation and GPU engine live here.
CLI, UI, benchmark harnesses and experimental search are separate packages.
"""

from .api.estimator import (
    WarpSymbolicRegressor,
    evaluate_formula,
    formula_to_sympy,
)
from .gpu import TensorGeneticEngine
from .gpu.errors import GpuUnavailableError

__all__ = [
    "WarpSymbolicRegressor",
    "TensorGeneticEngine",
    "GpuUnavailableError",
    "evaluate_formula",
    "formula_to_sympy",
]
