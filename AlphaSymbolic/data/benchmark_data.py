import numpy as np

# Standard Benchmark Problems
# Levels: 1 (Easy), 2 (Medium), 3 (Hard)

BENCHMARK_SUITE = [
    # --- Level 1: Polynomials & Basic Arithmetic ---
    {
        'id': 'p1',
        'name': 'Lineal',
        'formula_str': '2.5 * x + 1.0',
        'lambda': lambda x: 2.5 * x + 1.0,
        'domain': (-10, 10),
        'points': 20,
        'level': 1
    },
    {
        'id': 'p2',
        'name': 'Cuadratica Simple',
        'formula_str': 'x * x',
        'lambda': lambda x: x**2,
        'domain': (-5, 5),
        'points': 20,
        'level': 1
    },
    {
        'id': 'p3',
        'name': 'Polinomio Cubico',
        'formula_str': 'x**3 + x**2',
        'lambda': lambda x: x**3 + x**2,
        'domain': (-3, 3),
        'points': 20,
        'level': 1
    },
    
    # --- Level 2: Trigonometric & Transcendental ---
    {
        'id': 'p4',
        'name': 'Seno Basico',
        'formula_str': 'sin(x)',
        'lambda': lambda x: np.sin(x),
        'domain': (-np.pi, np.pi),
        'points': 30,
        'level': 2
    },
    {
        'id': 'p5',
        'name': 'Coseno Desplazado',
        'formula_str': 'cos(x) + 1',
        'lambda': lambda x: np.cos(x) + 1,
        'domain': (-np.pi, np.pi),
        'points': 30,
        'level': 2
    },
    {
        'id': 'p6',
        'name': 'Exponencial Simple',
        'formula_str': 'exp(x)',
        'lambda': lambda x: np.exp(x),
        'domain': (-2, 2), # Small domain to avoid explosion
        'points': 20,
        'level': 2
    },
    
    # --- Level 3: Physics / Complex ---
    {
        'id': 'p7',
        'name': 'Damped Oscillation',
        'formula_str': 'exp(-x) * sin(2*x)',
        'lambda': lambda x: np.exp(-x) * np.sin(2*x),
        'domain': (0, 4),
        'points': 40,
        'level': 3
    },
    {
        'id': 'p8',
        'name': 'Gaussian',
        'formula_str': 'exp(-x**2)',
        'lambda': lambda x: np.exp(-x**2),
        'domain': (-3, 3),
        'points': 30,
        'level': 3
    },
    {
        'id': 'p9',
        'name': 'Nguyen-3 (x^3 + x^2 + x)',
        'formula_str': 'x**3 + x**2 + x',
        'lambda': lambda x: x**3 + x**2 + x,
        'domain': (-2, 2),
        'points': 20,
        'level': 3
    },
    {
        'id': 'p10',
        'name': 'Rational Function',
        'formula_str': 'x / (1 + x**2)',
        'lambda': lambda x: x / (1 + x**2),
        'domain': (-4, 4),
        'points': 30,
        'level': 3
    }
]

def get_benchmark_data(problem_id):
    """Returns (x, y) for a specific problem ID."""
    for p in BENCHMARK_SUITE:
        if p['id'] == problem_id:
            x = np.linspace(p['domain'][0], p['domain'][1], p['points'])
            y = p['lambda'](x)
            return x, y, p
    return None, None, None
