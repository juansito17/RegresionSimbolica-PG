import numpy as np
import pytest
import torch

from AlphaSymbolic.core.gpu.config import GpuGlobals
from AlphaSymbolic.core.gpu.engine import TensorGeneticEngine


@pytest.fixture
def minimal_engine_run_config(monkeypatch):
    settings = {
        "GENERATIONS": 2,
        "USE_INITIAL_POP_CACHE": False,
        "USE_STRUCTURAL_SEEDS": False,
        "USE_PATTERN_SEEDS": False,
        "USE_SNIPER": False,
        "USE_NANO_PSO": False,
        "USE_BFGS_OPTIMIZER": False,
        "USE_PARETO_SELECTION": False,
        "USE_PATTERN_MEMORY": False,
        "USE_LIBRARY_LEARNING": False,
        "USE_CUDA_ORCHESTRATOR": True,
        "USE_LOG_TRANSFORMATION": True,
        "CONSTANT_PERTURBATION_RATE": 0.0,
    }
    for name, value in settings.items():
        monkeypatch.setattr(GpuGlobals, name, value)


def test_reset_run_state_clears_dataset_dependent_trackers_and_memories():
    engine = TensorGeneticEngine(
        device=torch.device("cpu"),
        pop_size=8,
        n_islands=1,
        max_len=8,
        num_variables=1,
    )
    engine._gpu_best_rmse = torch.tensor([0.0])
    engine._gpu_best_rpn = torch.ones(engine.max_len, dtype=engine.pop_dtype)
    engine._gpu_best_consts = torch.ones(engine.max_constants)
    engine.best_global_rmse = 0.0
    engine.best_global_rpn = engine._gpu_best_rpn.clone()
    engine._cached_sharing_pen = torch.ones(engine.pop_size)
    engine._pareto_rank_buf.fill_(1)
    engine.mutation_bank = torch.ones(2, engine.max_len, dtype=engine.pop_dtype)
    engine.pattern_memory.patterns_count[0] = 7
    engine.pattern_memory.n_patterns = 1
    engine.library_learner.valid[0] = True
    engine.library_learner.library_count[0] = 3

    engine._reset_run_state()

    assert torch.isinf(engine._gpu_best_rmse).all()
    assert (engine._gpu_best_rpn == 0).all()
    assert (engine._gpu_best_consts == 0).all()
    assert engine.best_global_rmse == float("inf")
    assert engine.best_global_rpn is None
    assert engine._cached_sharing_pen is None
    assert (engine._pareto_rank_buf == 0).all()
    assert engine.mutation_bank is None
    assert engine.pattern_memory.n_patterns == 0
    assert (engine.pattern_memory.patterns_count == 0).all()
    assert not engine.library_learner.valid.any()
    assert (engine.library_learner.library_count == 0).all()
    assert engine.stop_flag is False


def test_requested_log_transform_rejects_nonpositive_targets(
        minimal_engine_run_config):
    engine = TensorGeneticEngine(
        device=torch.device("cpu"),
        pop_size=8,
        n_islands=1,
        max_len=8,
        num_variables=1,
    )

    with pytest.raises(ValueError, match="strictly positive"):
        engine.run([0.0, 1.0, 2.0], [1.0, 0.0, 3.0], use_log=True)

    assert engine.last_run_used_log_transform is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU engine requires CUDA")
def test_reused_engine_isolates_best_honors_log_mode_and_stops(
        minimal_engine_run_config, monkeypatch):
    engine = TensorGeneticEngine(
        device=torch.device("cuda"),
        pop_size=32,
        n_islands=1,
        max_len=16,
        num_variables=1,
    )
    x = torch.linspace(0.25, 2.0, 16, device=engine.device)

    # Explicit False overrides the globally enabled log mode.
    first = engine.run(x, x, seeds=["x0"], timeout_sec=10, use_log=False)
    assert engine.last_run_best_rmse < 1e-6
    assert engine.last_run_used_log_transform is False
    assert not first.startswith("exp(")

    # Reusing the engine must not return the exact best from dataset 1.
    y_quadratic = x * x + 3.0
    second = engine.run(
        x,
        y_quadratic,
        seeds=["(x0 * x0) + 3"],
        timeout_sec=10,
        use_log=False,
    )
    second_pred = engine._eval_formula_safe(second, x.detach().cpu().numpy())
    assert second_pred is not None
    assert np.sqrt(np.mean(
        (second_pred - y_quadratic.detach().cpu().numpy()) ** 2)) < 1e-6
    assert engine.last_run_best_rmse < 1e-6

    # Explicit True is authoritative and controls the inverse transform.
    exponential = engine.run(
        x,
        torch.exp(x),
        seeds=["x0"],
        timeout_sec=10,
        use_log=True,
    )
    assert engine.last_run_used_log_transform is True
    assert exponential.startswith("exp(")

    # Cancellation is observed at the next generation boundary.
    monkeypatch.setattr(GpuGlobals, "GENERATIONS", 1000)
    monkeypatch.setattr(GpuGlobals, "GOOD_ENOUGH_MIN_SECONDS", 999.0)
    noisy_target = torch.sin(x) + 0.0137 * torch.cos(3.0 * x)

    def request_stop(*_args):
        engine.stop_flag = True

    engine.run(
        x,
        noisy_target,
        timeout_sec=30,
        callback=request_stop,
        use_log=False,
    )
    assert engine.last_run_stopped is True
    assert engine.last_run_generations < GpuGlobals.GENERATIONS
    assert engine.last_run_metrics["stopped"] is True
