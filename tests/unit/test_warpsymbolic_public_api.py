"""Public package and deprecation contracts for WarpSymbolic."""

import importlib

import numpy as np
import pytest

from warpsymbolic import GpuUnavailableError, WarpSymbolicRegressor


def test_canonical_public_api_is_gpu_named():
    from warpsymbolic.gpu import TensorGeneticEngine

    assert WarpSymbolicRegressor.__name__ == "WarpSymbolicRegressor"
    assert TensorGeneticEngine.__name__ == "TensorGeneticEngine"


def test_engine_normalizes_string_device():
    import torch

    from warpsymbolic.gpu import TensorGeneticEngine

    engine = TensorGeneticEngine(device="cpu", pop_size=2, max_len=4, n_islands=1)

    assert engine.device == torch.device("cpu")


def test_alpha_public_alias_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        legacy = importlib.import_module("AlphaSymbolic.sklearn")

    assert legacy.AlphaSymbolicRegressor is WarpSymbolicRegressor


def test_production_fit_rejects_missing_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    estimator = WarpSymbolicRegressor(device="cuda", max_time=1)

    with pytest.raises(GpuUnavailableError, match="CUDA-capable GPU"):
        estimator.fit(np.ones((4, 1)), np.ones(4))
