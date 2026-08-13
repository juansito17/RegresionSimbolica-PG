import numpy as np
from typing import List, Dict, Tuple, Any

# Standard Symbolic Regression Benchmarks
BENCHMARK_SUITE = [
    {"id": "nguyen-1", "name": "Nguyen-1", "formula": "x^3 + x^2 + x", "level": "Easy"},
    {"id": "nguyen-2", "name": "Nguyen-2", "formula": "x^4 + x^3 + x^2 + x", "level": "Easy"},
    {"id": "nguyen-3", "name": "Nguyen-3", "formula": "x^5 + x^4 + x^3 + x^2 + x", "level": "Easy"},
    {"id": "nguyen-4", "name": "Nguyen-4", "formula": "x^6 + x^5 + x^4 + x^3 + x^2 + x", "level": "Medium"},
    {"id": "nguyen-5", "name": "Nguyen-5", "formula": "sin(x^2) * cos(x) - 1", "level": "Medium"},
    {"id": "nguyen-6", "name": "Nguyen-6", "formula": "sin(x) + sin(x + x^2)", "level": "Medium"},
    {"id": "nguyen-7", "name": "Nguyen-7", "formula": "ln(x + 1) + ln(x^2 + 1)", "level": "Medium"},
    {"id": "nguyen-8", "name": "Nguyen-8", "formula": "sqrt(x)", "level": "Easy"},
    {"id": "keijzer-6", "name": "Keijzer-6", "formula": "sum(1/i for i=1 to x)", "level": "Hard"},
    {"id": "keijzer-7", "name": "Keijzer-7", "formula": "ln(x)", "level": "Easy"},
]

def get_benchmark_data(problem_id: str, n_points: int = 20) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Returns (x, y, formula_str) for a given benchmark problem.
    """
    # Default range
    x = np.linspace(0.1, 5.0, n_points)
    
    if problem_id == "nguyen-1":
        y = x**3 + x**2 + x
        return x, y, "x^3 + x^2 + x"
    elif problem_id == "nguyen-2":
        y = x**4 + x**3 + x**2 + x
        return x, y, "x^4 + x^3 + x^2 + x"
    elif problem_id == "nguyen-3":
        y = x**5 + x**4 + x**3 + x**2 + x
        return x, y, "x^5 + x^4 + x^3 + x^2 + x"
    elif problem_id == "nguyen-4":
        y = x**6 + x**5 + x**4 + x**3 + x**2 + x
        return x, y, "x^6 + x^5 + x^4 + x^3 + x^2 + x"
    elif problem_id == "nguyen-5":
        y = np.sin(x**2) * np.cos(x) - 1
        return x, y, "sin(x^2) * cos(x) - 1"
    elif problem_id == "nguyen-6":
        y = np.sin(x) + np.sin(x + x**2)
        return x, y, "sin(x) + sin(x + x^2)"
    elif problem_id == "nguyen-7":
        y = np.log(x + 1) + np.log(x**2 + 1)
        return x, y, "ln(x + 1) + ln(x^2 + 1)"
    elif problem_id == "nguyen-8":
        y = np.sqrt(x)
        return x, y, "sqrt(x)"
    elif problem_id == "keijzer-6":
        # Sum of harmonic series
        y = np.array([sum(1.0/i for i in range(1, int(val)+1)) if val >= 1 else 0 for val in x])
        return x, y, "harmonic(x)"
    elif problem_id == "keijzer-7":
        y = np.log(x)
        return x, y, "ln(x)"
    
    # Fallback
    return x, x, "x"
