import contextlib

import numpy as np
import pandas as pd
import pytest
import sympy
from sklearn.base import BaseEstimator, RegressorMixin, clone

import warpsymbolic.api.estimator as adapter
from warpsymbolic.api.sklearn import (
    WarpSymbolicRegressor,
    evaluate_formula,
    formula_to_sympy,
)


def _patch_engine(monkeypatch, engine_class):
    monkeypatch.setattr(adapter, "_load_engine_class", lambda: engine_class)
    monkeypatch.setattr(adapter, "_release_cuda_memory", lambda: None)
    monkeypatch.setattr(
        adapter,
        "_seeded_engine_runtime",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )


def test_estimator_is_cloneable_and_initialization_is_lazy(monkeypatch):
    def fail_if_loaded():
        raise AssertionError("constructing or cloning must not import the GPU engine")

    monkeypatch.setattr(adapter, "_load_engine_class", fail_if_loaded)
    estimator = WarpSymbolicRegressor(device="cpu", max_time=12, random_state=17)
    cloned = clone(estimator)

    assert isinstance(estimator, (BaseEstimator, RegressorMixin))
    assert cloned.get_params()["max_time"] == 12
    assert cloned.get_params()["generations"] == 150
    assert cloned.random_state == 17
    assert not hasattr(cloned, "formula_")


def test_fit_selects_gpu_variables_caps_samples_and_maps_dataframe_names(monkeypatch):
    class RecordingEngine:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.run_args = None
            self.__class__.instances.append(self)

        def run(self, X, y, **kwargs):
            self.run_args = (np.asarray(X).copy(), np.asarray(y).copy(), kwargs)
            return "x0 + 2*x1"

    _patch_engine(monkeypatch, RecordingEngine)
    rng = np.random.RandomState(4)
    n_samples = 300
    signal_a = rng.normal(size=n_samples)
    signal_b = rng.normal(size=n_samples)
    frame = pd.DataFrame(
        {
            "noise_a": rng.normal(size=n_samples),
            "signal_a": signal_a,
            "noise_b": rng.normal(size=n_samples),
            "signal_b": signal_b,
            "noise_c": rng.normal(size=n_samples),
        }
    )
    target = signal_a + 2.0 * signal_b
    estimator = WarpSymbolicRegressor(
        device="cpu",
        pop_size=2_000,
        n_islands=4,
        max_gpu_variables=2,
        max_gpu_samples=64,
        feature_selection="correlation",
        max_time=9,
        random_state=123,
    )

    estimator.fit(frame, target)

    engine = RecordingEngine.instances[-1]
    engine_X, engine_y, run_kwargs = engine.run_args
    assert engine.kwargs["num_variables"] == 2
    assert engine_X.shape == (64, 2)
    assert engine_y.shape == (64,)
    assert run_kwargs["timeout_sec"] == 9
    assert estimator.selected_feature_indices_.tolist() == [1, 3]
    assert estimator.selected_feature_names_.tolist() == ["signal_a", "signal_b"]
    assert estimator.fit_status_ == "engine"
    assert estimator.to_sympy_string() == "signal_a + 2*signal_b"

    expected = estimator.predict(frame)
    reordered = estimator.predict(frame[list(reversed(frame.columns))])
    np.testing.assert_allclose(reordered, expected)


def test_engine_failure_uses_fitted_linear_symbolic_fallback(monkeypatch):
    class FailingEngine:
        def __init__(self, **_kwargs):
            pass

        def run(self, *_args, **_kwargs):
            raise RuntimeError("synthetic engine failure")

    _patch_engine(monkeypatch, FailingEngine)
    rng = np.random.RandomState(8)
    X = rng.normal(size=(100, 2))
    y = 1.25 + 2.0 * X[:, 0] - 3.5 * X[:, 1]
    estimator = WarpSymbolicRegressor(
        device="cpu",
        max_gpu_variables=None,
        max_gpu_samples=None,
        random_state=2,
    ).fit(X, y)

    assert estimator.fit_status_ == "fallback"
    assert estimator.engine_error_.startswith("RuntimeError:")
    np.testing.assert_allclose(estimator.predict(X), y, atol=1e-10)
    assert {
        str(symbol)
        for symbol in sympy.sympify(estimator.to_sympy_string()).free_symbols
    } == {"x0", "x1"}


def test_validation_keeps_linear_fallback_when_gpu_candidate_is_worse(monkeypatch):
    class PoorEngine:
        def __init__(self, **_kwargs):
            pass

        def run(self, *_args, **_kwargs):
            return "0"

    _patch_engine(monkeypatch, PoorEngine)
    X = np.linspace(-2.0, 2.0, 100).reshape(-1, 1)
    y = 4.0 * X[:, 0] - 1.0
    estimator = WarpSymbolicRegressor(
        device="cpu",
        max_gpu_variables=1,
        validation_fraction=0.25,
        random_state=42,
    ).fit(X, y)

    assert estimator.candidate_formula_ == "0"
    assert estimator.fit_status_ == "fallback"
    assert estimator.selection_reason_ == "fallback_validation_rmse"
    assert estimator.fallback_validation_rmse_ < estimator.engine_validation_rmse_
    np.testing.assert_allclose(estimator.predict(X), y, atol=1e-10)


def test_polynomial_portfolio_can_win_with_more_variables_than_gpu(monkeypatch):
    class PoorEngine:
        def __init__(self, **_kwargs):
            pass

        def run(self, *_args, **_kwargs):
            return "0"

    _patch_engine(monkeypatch, PoorEngine)
    rng = np.random.RandomState(91)
    frame = pd.DataFrame(
        rng.uniform(-1.0, 1.0, size=(240, 6)),
        columns=[f"feature_{index}" for index in range(6)],
    )
    y = frame["feature_0"].to_numpy() ** 2 + 3.0 * frame["feature_5"].to_numpy()
    estimator = WarpSymbolicRegressor(
        device="cpu",
        max_gpu_variables=2,
        max_polynomial_variables=8,
        polynomial_degree=3,
        random_state=19,
    ).fit(frame, y)

    assert estimator.fit_status_ == "polynomial"
    assert estimator.polynomial_degree_ in {2, 3}
    assert len(estimator.selected_feature_indices_) == 6
    assert {"feature_0", "feature_5"} <= {
        str(symbol) for symbol in estimator.to_sympy().free_symbols
    }
    np.testing.assert_allclose(estimator.predict(frame), y, atol=2e-5)


def test_selected_polynomial_is_refit_on_all_outer_training_rows(monkeypatch):
    class PoorEngine:
        def __init__(self, **_kwargs):
            pass

        def run(self, *_args, **_kwargs):
            return "0"

    _patch_engine(monkeypatch, PoorEngine)
    original_polynomial_formula = adapter._polynomial_formula
    fitted_row_counts = []

    def recording_polynomial_formula(X, y, degree, ridge_alpha):
        fitted_row_counts.append(len(y))
        return original_polynomial_formula(X, y, degree, ridge_alpha)

    monkeypatch.setattr(
        adapter,
        "_polynomial_formula",
        recording_polynomial_formula,
    )
    X = np.linspace(-2.0, 2.0, 100).reshape(-1, 1)
    y = 0.5 + 2.0 * X[:, 0] ** 2
    estimator = WarpSymbolicRegressor(
        device="cpu",
        polynomial_degree=2,
        validation_fraction=0.2,
        max_gpu_variables=1,
        random_state=7,
    ).fit(X, y)

    assert estimator.fit_status_ == "polynomial"
    assert fitted_row_counts[-1] == len(y)
    np.testing.assert_allclose(estimator.predict(X), y, atol=1e-6)


def test_nonfinite_engine_predictions_are_repaired_by_fallback(monkeypatch):
    class ReciprocalEngine:
        def __init__(self, **_kwargs):
            pass

        def run(self, *_args, **_kwargs):
            return "1/x0"

    _patch_engine(monkeypatch, ReciprocalEngine)
    X = np.linspace(0.2, 2.0, 40).reshape(-1, 1)
    estimator = WarpSymbolicRegressor(
        device="cpu",
        max_gpu_variables=1, random_state=0
    ).fit(X, 1.0 / X[:, 0])
    prediction = estimator.predict(np.array([[0.0], [0.5], [1.0]]))

    assert estimator.fit_status_ == "engine"
    assert np.isfinite(prediction).all()
    assert prediction[1] == pytest.approx(2.0)
    assert prediction[2] == pytest.approx(1.0)


def test_dataframe_missing_column_and_nonfinite_values_are_handled(monkeypatch):
    class IdentityEngine:
        def __init__(self, **_kwargs):
            pass

        def run(self, *_args, **_kwargs):
            return "x0"

    _patch_engine(monkeypatch, IdentityEngine)
    frame = pd.DataFrame({"temperature": [1.0, np.nan, 3.0], "unused": [2, 3, 4]})
    estimator = WarpSymbolicRegressor(
        device="cpu",
        max_gpu_variables=1,
        feature_selection="correlation",
        random_state=0,
    ).fit(frame, [1.0, 2.0, 3.0])

    assert np.isfinite(estimator.predict(frame)).all()
    with pytest.raises(ValueError, match="missing fitted columns"):
        estimator.predict(frame[["temperature"]])


def test_safe_formula_language_and_sympy_aliases():
    X = np.array([[1.0, 2.0], [2.0, 3.0]])
    np.testing.assert_allclose(
        evaluate_formula("abs(-x0) + x1^2", X),
        [5.0, 11.0],
    )
    expression = formula_to_sympy(
        "abs(x0) + lgamma(x1) + fact(3)",
        ["velocity", "mass"],
    )
    assert sympy.Symbol("velocity", real=True) in expression.free_symbols
    assert sympy.Symbol("mass", real=True) in expression.free_symbols
    assert "loggamma" in str(expression)

    with pytest.raises(ValueError, match="only direct calls"):
        evaluate_formula("__import__('os').system('echo unsafe')", X)
    with pytest.raises(ValueError, match="unknown name"):
        evaluate_formula("secret + x0", X)
    with pytest.raises(ValueError, match="only 2 variables"):
        evaluate_formula("x2", X)


def test_srbench_shim_exports_the_expected_protocol(monkeypatch):
    class ConstantEngine:
        def __init__(self, **_kwargs):
            pass

        def run(self, *_args, **_kwargs):
            return "2"

    _patch_engine(monkeypatch, ConstantEngine)
    from integrations.srbench.experiment.methods.alphasymbolic import regressor

    fitted = clone(regressor.est).set_params(
        pop_size=100,
        n_islands=2,
        max_time=1,
    )
    frame = pd.DataFrame({"force": [1.0, 2.0, 3.0]})
    fitted.fit(frame, [2.0, 2.0, 2.0])

    assert regressor.model(fitted, frame) == "2"
    assert regressor.complexity(fitted) >= 1
    assert regressor.eval_kwargs["scale_x"] is True
    assert regressor.eval_kwargs["scale_y"] is True
    assert regressor.hyper_params == []
    assert regressor.est.search_mode == "adaptive"
