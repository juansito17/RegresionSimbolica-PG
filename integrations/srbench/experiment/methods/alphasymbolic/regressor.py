"""Thin SRBench shim for :class:`AlphaSymbolicRegressor`.

The implementation remains in the installable AlphaSymbolic package so this
file can be copied into either the legacy or current SRBench method layout
without duplicating model logic.
"""

from AlphaSymbolic.sklearn import AlphaSymbolicRegressor


def model(estimator, X=None):
    """Return a SymPy-compatible expression using SRBench column names."""

    return estimator.to_sympy_string(X)


def complexity(estimator):
    """Return the fitted expression-tree node count."""

    return int(estimator.symbolic_complexity_)


est = AlphaSymbolicRegressor(
    pop_size=50_000,
    n_islands=20,
    max_len=48,
    max_constants=10,
    max_gpu_variables=4,
    max_gpu_samples=1024,
    feature_selection="hybrid",
    fallback_strategy="linear",
    polynomial_degree=3,
    max_polynomial_variables=8,
    ridge_alpha=1e-6,
    generations=150,
    max_time=60,
    random_state=0,
    search_mode="adaptive",
    target_transform="auto",
    max_active_variables=8,
)


# Keep SRBench's official outer scaling enabled so AlphaSymbolic receives the
# same prepared inputs as peer methods in the 2025 protocol.
eval_kwargs = {
    "scale_x": True,
    "scale_y": True,
    "use_dataframe": True,
    "max_train_samples": 0,
    "test_params": {
        "pop_size": 2_000,
        "n_islands": 4,
        "max_gpu_samples": 256,
        "polynomial_degree": 2,
        "generations": 5,
        "max_time": 30,
        "search_mode": "adaptive",
    },
}


# Hyperparameters are frozen on the separate development suite.  Official
# datasets therefore cannot feed back into algorithm configuration.
hyper_params = []
