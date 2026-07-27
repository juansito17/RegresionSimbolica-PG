import importlib

import numpy as np
import pytest


class _FakeModel:
    def to(self, _device):
        return self

    def load_state_dict(self, _state):
        return self

    def eval(self):
        return self


class _DictMCTS:
    def __init__(self, *_args, **_kwargs):
        pass

    def search(self, *_args, **_kwargs):
        return {
            "tokens": ["x0"],
            "formula": "x0",
            "rmse": 0.0,
            "root": object(),
            "pareto_front": [],
        }


def test_search_entrypoints_import_without_optional_data_package():
    basic_search = importlib.import_module("AlphaSymbolic.search.basic_search")
    search_pro = importlib.import_module("AlphaSymbolic.search.search_pro")

    assert callable(basic_search.solve_problem)
    assert callable(search_pro.solve_pro)


def test_basic_search_consumes_mcts_result_dictionary(monkeypatch, tmp_path):
    from AlphaSymbolic.search import basic_search

    monkeypatch.setattr(
        basic_search, "AlphaSymbolicModel", lambda **_kwargs: _FakeModel())
    monkeypatch.setattr(basic_search, "MCTS", _DictMCTS)
    monkeypatch.setattr(
        basic_search, "optimize_constants", lambda *_args, **_kwargs: ({}, 0.0))
    monkeypatch.setattr(
        basic_search, "simplify_tree", lambda tree: tree.get_infix())

    x = np.linspace(-2.0, 2.0, 9)
    result = basic_search.solve_problem(
        x,
        x,
        model_path=str(tmp_path / "missing-model.pth"),
        simulations=1,
    )

    assert result["tokens"] == ["x0"]
    assert result["raw_formula"] == "x0"
    assert result["rmse"] == pytest.approx(0.0)


def test_basic_search_reports_empty_mcts_result(monkeypatch, tmp_path):
    from AlphaSymbolic.search import basic_search

    class EmptyMCTS(_DictMCTS):
        def search(self, *_args, **_kwargs):
            return {"tokens": None, "rmse": float("inf")}

    monkeypatch.setattr(
        basic_search, "AlphaSymbolicModel", lambda **_kwargs: _FakeModel())
    monkeypatch.setattr(basic_search, "MCTS", EmptyMCTS)

    x = np.linspace(-1.0, 1.0, 5)
    with pytest.raises(RuntimeError, match="without a valid formula"):
        basic_search.solve_problem(
            x,
            x,
            model_path=str(tmp_path / "missing-model.pth"),
            simulations=1,
        )


def test_pro_search_mcts_flow_survives_missing_pattern_memory(
        monkeypatch, tmp_path):
    from AlphaSymbolic.search import search_pro

    monkeypatch.setattr(
        search_pro, "AlphaSymbolicModel", lambda **_kwargs: _FakeModel())
    monkeypatch.setattr(search_pro, "MCTS", _DictMCTS)
    monkeypatch.setattr(search_pro, "PatternMemory", None)
    monkeypatch.setattr(
        search_pro,
        "detect_pattern",
        lambda *_args: {
            "type": "linear",
            "confidence": 1.0,
            "suggested_ops": ["+"],
        },
    )
    monkeypatch.setattr(
        search_pro, "optimize_constants", lambda *_args, **_kwargs: ({}, 0.0))
    monkeypatch.setattr(
        search_pro, "simplify_tree", lambda tree: tree.get_infix())

    x = np.linspace(-2.0, 2.0, 9)
    results, pareto = search_pro.solve_pro(
        x,
        x,
        model_path=str(tmp_path / "missing-model.pth"),
        method="mcts",
        mcts_simulations=1,
        use_memory=True,
        verbose=False,
    )

    assert results["best_accuracy"]["formula"] == "x0"
    assert results["best_accuracy"]["rmse"] == pytest.approx(0.0)
    assert pareto.get_best_by_rmse().tokens == ["x0"]
