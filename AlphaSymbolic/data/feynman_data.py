"""
Feynman Benchmark Dataset (Subset)
Contains physical equations from the AI Feynman dataset for benchmarking Symbolic Regression.
Format: (Name, Formula_String, Variables_Range)
"""

# Dictionary of formulas
# Keys: Name/ID
# Values: {'formula': string, 'vars': list of variable names, 'ranges': dict of ranges}
# Note: For this system we map variables to x[0], x[1] etc internally, but here we define the target ground truth.
# Since our current model is single-variable (y = f(x)), we will select the 1D or quasi-1D subset first, 
# or fix other variables as constants to test the core relationship.

# IMPORTANT: The current AlphaSymbolic model is trained primarily on single-variable problems (y = f(x)).
# To test it fairly, we will use equations that can be expressed as f(x) or simplify multi-variable ones 
# by fixing all but one variable (ceteris paribus).

FEYNMAN_1D_SUBSET = [
    {
        "id": "I.12.11", 
        "name": "Electric Field from Charge",
        "formula": "C / (x * x)", # q/(4*pi*eps*r^2) -> simplified to C/x^2
        "description": "Inverse square law",
        "target_complexity": 5
    },
    {
        "id": "I.6.20a", 
        "name": "Gaussian Probability", 
        "formula": "exp(-(x * x) / 2)", # e^(-theta^2/2)
        "description": "Gaussian distribution kernel",
        "target_complexity": 7
    },
    {
        "id": "I.15.3x", 
        "name": "Relativistic Mass", 
        "formula": "x / sqrt(1 - (v * v))", # m0 / sqrt(1 - v^2/c^2). Let's treat x as m0 and v as constant? No, let's treat x as v/c.
        "formula_1d": "1 / sqrt(1 - (x * x))", # Gamma factor where x = v/c
        "description": "Relativistic factor",
        "target_complexity": 9
    },
    {
        "id": "I.29.16",
        "name": "Damped Oscillation",
        "formula": "exp(-x) * sin(x)",
        "description": "Damped harmonic oscillator (simplified)",
        "target_complexity": 6
    },
    {
        "id": "II.35.21", 
        "name": "Maxwell Boltzmann", 
        "formula": "x * x * exp(-(x * x))", # v^2 * exp(-mv^2/2kT)
        "description": "Velocity distribution",
        "target_complexity": 8
    },
    {
        "id": "Bonus.1",
        "name": "Growth Decay",
        "formula": "1 - exp(-x)",
        "description": "Charging Capacitor / Population limit",
        "target_complexity": 5
    },
    {
        "id": "Bonus.2",
        "name": "Sigmoid",
        "formula": "1 / (1 + exp(-x))",
        "description": "Logistic function",
        "target_complexity": 7
    },
    {
        "id": "Bonus.3",
        "name": "Sinc Function",
        "formula": "sin(x) / x",
        "description": "Signal processing",
        "target_complexity": 5
    }
]

def get_feynman_problem(problem_id):
    for p in FEYNMAN_1D_SUBSET:
        if p['id'] == problem_id:
            return p
    return None
