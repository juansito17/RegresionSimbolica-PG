import os
import sys

import torch


_test_dir = os.path.dirname(os.path.abspath(__file__))
_alpha_symbolic_dir = os.path.dirname(_test_dir)
_project_root = os.path.dirname(_alpha_symbolic_dir)
sys.path.insert(0, _project_root)
sys.path.insert(0, _alpha_symbolic_dir)


def _mock_run_vm(population: torch.Tensor, x: torch.Tensor, constants: torch.Tensor = None, strict_mode: int = 0):
    b = population.shape[0]
    n = x.shape[1]
    total = b * n
    preds = torch.zeros(total, dtype=x.dtype, device=population.device)
    sp = torch.ones(total, dtype=torch.int32, device=population.device)
    err = torch.zeros(total, dtype=torch.uint8, device=population.device)
    return preds, sp, err


def test_evaluate_batch_never_tries_fused_on_cpu_hot_loop():
    from warpsymbolic.gpu.grammar import GPUGrammar
    from warpsymbolic.gpu.evaluation import GPUEvaluator
    from warpsymbolic.gpu.config import GpuGlobals

    grammar = GPUGrammar(num_variables=1, use_globals=False)
    evaluator = GPUEvaluator(grammar, device='cpu', dtype=torch.float64)

    population = torch.ones((4, 5), dtype=torch.uint8)
    x = torch.randn(1, 16, dtype=torch.float64)
    y = torch.randn(16, dtype=torch.float64)
    constants = torch.zeros((4, 3), dtype=torch.float64)

    calls = {'n': 0}

    def failing_fused(*args, **kwargs):
        calls['n'] += 1
        raise RuntimeError("forced fused failure")

    evaluator.vm.eval_fused = failing_fused
    evaluator._run_vm = _mock_run_vm

    original_loss = GpuGlobals.LOSS_FUNCTION
    try:
        GpuGlobals.LOSS_FUNCTION = 'RMSE'
        out1 = evaluator.evaluate_batch(population, x, y, constants)
        out2 = evaluator.evaluate_batch(population, x, y, constants)
    finally:
        GpuGlobals.LOSS_FUNCTION = original_loss

    assert out1.shape == (4,)
    assert out2.shape == (4,)
    assert calls['n'] == 0, (
        "Con tensores CPU no debe intentarse eval_fused; evita overhead por "
        "excepciones en cada generación."
    )


def test_evaluate_batch_skips_fused_on_cpu_inputs():
    from warpsymbolic.gpu.grammar import GPUGrammar
    from warpsymbolic.gpu.evaluation import GPUEvaluator
    from warpsymbolic.gpu.config import GpuGlobals

    grammar = GPUGrammar(num_variables=1, use_globals=False)
    evaluator = GPUEvaluator(grammar, device='cpu', dtype=torch.float64)

    population = torch.ones((3, 4), dtype=torch.uint8)
    x = torch.randn(1, 8, dtype=torch.float64)
    y = torch.randn(8, dtype=torch.float64)
    constants = torch.zeros((3, 2), dtype=torch.float64)

    def should_not_be_called(*args, **kwargs):
        raise AssertionError("eval_fused no debe llamarse con tensores CPU")

    evaluator.vm.eval_fused = should_not_be_called
    evaluator._run_vm = _mock_run_vm

    original_loss = GpuGlobals.LOSS_FUNCTION
    try:
        GpuGlobals.LOSS_FUNCTION = 'RMSE'
        out = evaluator.evaluate_batch(population, x, y, constants)
    finally:
        GpuGlobals.LOSS_FUNCTION = original_loss

    assert out.shape == (3,)
    assert torch.isfinite(out).all()
