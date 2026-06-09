import numpy as np
import torch

from AlphaSymbolic.ui.live_state import LiveRunState
from AlphaSymbolic.ui import app_gpu_live
from AlphaSymbolic.ui.app_gpu_live import _fill_live_plot_predictions_with_formula


class FakeSimplifier:
    def simplify_batch(self, pop, consts):
        return pop, consts, 0


class FakeVM:
    def eval(self, rpn, x_t, consts):
        n = x_t.shape[-1]
        preds = torch.arange(1, n + 1, dtype=torch.float32).unsqueeze(0)
        sp = torch.ones(1, dtype=torch.long)
        err = torch.zeros(1, dtype=torch.bool)
        return preds, sp, err


class FakeEvaluator:
    def __init__(self):
        self.vm = FakeVM()


class FakeEngine:
    def __init__(self, **kwargs):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.pop_size = kwargs["pop_size"]
        self.n_islands = kwargs["n_islands"]
        self.stop_flag = False
        self.gpu_simplifier = FakeSimplifier()
        self.evaluator = FakeEvaluator()

    def rpn_to_infix(self, _rpn, _consts):
        return "x0 + 1"

    def run(self, _x, _y, _seeds, _timeout_sec, callback):
        rpn = torch.tensor([1, 2, 3], dtype=torch.uint8)
        consts = torch.zeros(4)
        callback(10, 0.0, rpn, consts, True, 0)
        return "x0 + 1"


def test_run_live_gpu_evolution_streams_with_fake_engine(monkeypatch):
    monkeypatch.setattr(app_gpu_live, "ENGINE_CLS", FakeEngine)
    state = LiveRunState()

    outputs = list(
        app_gpu_live.run_live_gpu_evolution(
            "1,2,3",
            "2,3,4",
            10000,
            2,
            4,
            1,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            run_state=state,
            verbose=True,
        )
    )

    assert len(outputs) >= 2
    assert "x0 + 1" in outputs[-1][1]
    assert "Finalizado" in outputs[-1][0]


def test_live_plot_prediction_fallback_fills_nan_points_from_formula():
    x = np.asarray([-5, -4, -3, 0, 1, 2], dtype=float)
    y_pred = np.asarray([np.nan, np.nan, np.nan, 1, 2, 5], dtype=float)

    filled = _fill_live_plot_predictions_with_formula(x, y_pred, "(1 + (x0**2))")

    assert np.all(np.isfinite(filled))
    assert np.allclose(filled, 1 + x**2)
