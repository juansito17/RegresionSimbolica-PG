import threading
import time

import numpy as np
import torch

from AlphaSymbolic.ui.live_state import LiveRunState
from AlphaSymbolic.ui import app_gpu_live
from AlphaSymbolic.ui.app_gpu_live import _fill_live_plot_predictions_with_formula
from warpsymbolic.gpu.config import GpuGlobals


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
    observed_loss_function = None
    observed_use_log = None

    def __init__(self, **kwargs):
        type(self).observed_loss_function = GpuGlobals.LOSS_FUNCTION
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.pop_size = kwargs["pop_size"]
        self.n_islands = kwargs["n_islands"]
        self.stop_flag = False
        self.gpu_simplifier = FakeSimplifier()
        self.evaluator = FakeEvaluator()

    def rpn_to_infix(self, _rpn, _consts):
        return "x0 + 1"

    def run(self, _x, _y, _seeds, _timeout_sec, callback, use_log=None):
        type(self).observed_use_log = use_log
        self.last_run_used_log_transform = bool(use_log)
        rpn = torch.tensor([1, 2, 3], dtype=torch.uint8)
        consts = torch.zeros(4)
        callback(10, 0.0, rpn, consts, True, 0)
        return "exp(x0 + 1)" if use_log else "x0 + 1"


class FakeInvalidCallbackEngine(FakeEngine):
    def rpn_to_infix(self, _rpn, _consts):
        return "Invalid"

    def run(self, _x, _y, _seeds, _timeout_sec, callback, use_log=None):
        self.last_run_used_log_transform = bool(use_log)
        self.last_run_best_rmse = 0.0
        self.last_run_generations = 10
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


def test_final_engine_formula_replaces_invalid_progress_callback(monkeypatch):
    monkeypatch.setattr(app_gpu_live, "ENGINE_CLS", FakeInvalidCallbackEngine)
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

    assert "x0 + 1" in outputs[-1][1]
    assert "Invalid" not in outputs[-1][1]
    assert "Finalizado" in outputs[-1][0]


def test_log_transform_uses_rmse_without_double_log(monkeypatch):
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
            True,
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
            True,
            False,
            False,
            False,
            run_state=state,
            verbose=True,
        )
    )

    assert FakeEngine.observed_loss_function == "RMSE"
    assert FakeEngine.observed_use_log is True
    assert "RMSE log" in outputs[-1][2]
    assert "exp(x0 + 1)" in outputs[-1][1]


class FakeStoppableEngine(FakeEngine):
    started = threading.Event()
    stopped = threading.Event()

    def run(self, _x, _y, _seeds, _timeout_sec, _callback, use_log=None):
        self.last_run_used_log_transform = bool(use_log)
        type(self).started.set()
        while not self.stop_flag:
            time.sleep(0.005)
        type(self).stopped.set()
        return None


class FakeErrorEngine(FakeEngine):
    def run(self, _x, _y, _seeds, _timeout_sec, _callback, use_log=None):
        self.last_run_used_log_transform = bool(use_log)
        raise RuntimeError("fallo controlado")


class FakeOOMEngine:
    def __init__(self, **_kwargs):
        raise RuntimeError("CUDA out of memory")


def _consume_live(state, *, use_log=False, x_str="1,2,3", y_str="2,3,4"):
    return list(
        app_gpu_live.run_live_gpu_evolution(
            x_str,
            y_str,
            10000,
            2,
            4,
            0,
            use_log,
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


def test_input_validation_error_is_terminal_for_live_controls():
    outputs = _consume_live(LiveRunState(), x_str="", y_str="")

    assert "Error: Ingresa valores para X e Y." in outputs[-1][0]
    assert app_gpu_live._live_status_is_terminal(outputs[-1][0])


def test_stop_waits_for_worker_and_never_reports_success(monkeypatch):
    FakeStoppableEngine.started.clear()
    FakeStoppableEngine.stopped.clear()
    monkeypatch.setattr(app_gpu_live, "ENGINE_CLS", FakeStoppableEngine)
    state = LiveRunState()
    result = {}

    consumer = threading.Thread(
        target=lambda: result.setdefault("outputs", _consume_live(state)),
        daemon=True,
    )
    consumer.start()
    assert FakeStoppableEngine.started.wait(2.0)

    state.request_stop()
    consumer.join(2.0)

    assert not consumer.is_alive()
    assert FakeStoppableEngine.stopped.is_set()
    assert "Detenido" in result["outputs"][-1][0]
    assert all("Finalizado" not in output[0] for output in result["outputs"])
    assert state.engine is None
    assert state.thread is None
    assert app_gpu_live.LIVE_ENGINE is None


def test_engine_error_is_terminal_and_runtime_is_cleaned(monkeypatch):
    monkeypatch.setattr(app_gpu_live, "ENGINE_CLS", FakeErrorEngine)
    state = LiveRunState()

    outputs = _consume_live(state)

    assert "Error durante evolución GPU" in outputs[-1][0]
    assert "fallo controlado" in outputs[-1][0]
    assert all("Finalizado" not in output[0] for output in outputs)
    assert state.engine is None
    assert state.thread is None
    assert app_gpu_live.LIVE_ENGINE is None


def test_initialization_failure_restores_globals_and_keeps_engine_attribute(monkeypatch):
    monkeypatch.setattr(app_gpu_live, "ENGINE_CLS", FakeOOMEngine)
    previous_pop_size = GpuGlobals.POP_SIZE
    previous_log_mode = GpuGlobals.USE_LOG_TRANSFORMATION
    state = LiveRunState()

    outputs = _consume_live(state, use_log=not previous_log_mode)

    assert "No se pudo inicializar" in outputs[-1][0]
    assert state.engine is None
    assert hasattr(state, "engine")
    assert app_gpu_live.LIVE_ENGINE is None
    assert GpuGlobals.POP_SIZE == previous_pop_size
    assert GpuGlobals.USE_LOG_TRANSFORMATION == previous_log_mode


def test_closing_stream_early_restores_global_configuration(monkeypatch):
    monkeypatch.setattr(app_gpu_live, "ENGINE_CLS", FakeEngine)
    previous_pop_size = GpuGlobals.POP_SIZE
    previous_log_mode = GpuGlobals.USE_LOG_TRANSFORMATION
    state = LiveRunState()
    stream = app_gpu_live.run_live_gpu_evolution(
        "1,2,3",
        "2,3,4",
        10000,
        2,
        4,
        0,
        not previous_log_mode,
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

    next(stream)
    assert GpuGlobals.POP_SIZE == 10000
    stream.close()

    assert GpuGlobals.POP_SIZE == previous_pop_size
    assert GpuGlobals.USE_LOG_TRANSFORMATION == previous_log_mode
    assert state.engine is None
    assert app_gpu_live.LIVE_ENGINE is None
