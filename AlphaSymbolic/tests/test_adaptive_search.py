import time

import numpy as np
import pandas as pd
from sklearn.base import clone

from AlphaSymbolic.sklearn import AlphaSymbolicRegressor


def test_adaptive_is_cloneable_and_exports_audit_state():
    estimator = AlphaSymbolicRegressor(
        search_mode="adaptive", max_time=1.0, polynomial_degree=2, random_state=7
    )
    cloned = clone(estimator)
    X = np.arange(12, dtype=np.float64).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 3.0
    cloned.fit(X, y)
    assert np.allclose(cloned.predict(X), y)
    assert cloned.formula_
    assert cloned.sympy_formula_
    assert cloned.pareto_front_
    assert cloned.search_report_["cv"] == "leave-one-out"
    assert cloned.search_report_["single_expression"] is True
    assert len(cloned.configuration_hash_) == 64


def test_column_names_do_not_change_search_or_predictions():
    values = np.column_stack(
        (np.linspace(-2.0, 2.0, 30), np.linspace(1.0, 4.0, 30))
    )
    y = values[:, 0] * values[:, 1]
    first = AlphaSymbolicRegressor(
        search_mode="adaptive", max_time=1.0, polynomial_degree=2, random_state=3
    ).fit(pd.DataFrame(values, columns=["secret_hint", "answer_name"]), y)
    second = AlphaSymbolicRegressor(
        search_mode="adaptive", max_time=1.0, polynomial_degree=2, random_state=3
    ).fit(pd.DataFrame(values, columns=["a", "b"]), y)
    assert first.formula_ == second.formula_
    assert np.allclose(first.predict(values), second.predict(values))
    assert "secret_hint" not in first.formula_
    assert "secret_hint" in first.sympy_formula_


def test_missing_values_and_irrelevant_variables_are_supported():
    rng = np.random.default_rng(9)
    signal = np.linspace(-1.0, 1.0, 60)
    X = np.column_stack((signal, rng.normal(size=(60, 5))))
    X[4, 0] = np.nan
    y = 4.0 * signal - 1.0
    model = AlphaSymbolicRegressor(
        search_mode="adaptive", max_time=1.0, polynomial_degree=1, random_state=9
    ).fit(X, y)
    prediction = model.predict(X)
    assert prediction.shape == y.shape
    assert np.isfinite(prediction).all()


def test_adaptive_budget_is_capped_at_sixty_seconds():
    X = np.arange(8, dtype=np.float64).reshape(-1, 1)
    y = X[:, 0] + 1.0
    started = time.perf_counter()
    model = AlphaSymbolicRegressor(
        search_mode="adaptive", max_time=600.0, polynomial_degree=1
    ).fit(X, y)
    assert time.perf_counter() - started < 5.0
    assert model.search_budget_sec_ == 60.0


def test_fold_policy_changes_only_with_row_count():
    for rows, expected in ((12, "leave-one-out"), (45, "5-fold"), (510, "3-fold")):
        X = np.linspace(0.0, 1.0, rows).reshape(-1, 1)
        y = 3.0 * X[:, 0]
        model = AlphaSymbolicRegressor(
            search_mode="adaptive", max_time=0.5, polynomial_degree=1
        ).fit(X, y)
        assert model.search_report_["cv"] == expected


def test_configuration_hash_is_frozen_across_repetition_seeds():
    X = np.arange(10, dtype=np.float64).reshape(-1, 1)
    y = X[:, 0] - 2.0
    hashes = {
        AlphaSymbolicRegressor(
            search_mode="adaptive",
            max_time=0.5,
            polynomial_degree=1,
            random_state=seed,
        ).fit(X, y).configuration_hash_
        for seed in (0, 1)
    }
    assert len(hashes) == 1


def test_semantic_and_non_identifier_column_labels_are_export_only():
    X = pd.DataFrame(
        {"true formula hint": np.arange(9, dtype=np.float64), "β column": np.ones(9)}
    )
    y = 5.0 * X.iloc[:, 0].to_numpy()
    model = AlphaSymbolicRegressor(
        search_mode="adaptive", max_time=0.5, polynomial_degree=1
    ).fit(X, y)
    assert np.allclose(model.predict(X), y)
    assert "true formula hint" not in model.formula_
    assert "true formula hint" in model.sympy_formula_
