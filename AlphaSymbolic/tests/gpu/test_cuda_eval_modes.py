import pytest
import torch

from AlphaSymbolic.core.gpu.cuda_vm import CudaRPNVM, rpn_cuda
from AlphaSymbolic.core.gpu.config import GpuGlobals
from AlphaSymbolic.core.gpu.evaluation import GPUEvaluator
from AlphaSymbolic.core.gpu.grammar import GPUGrammar, PAD_ID
from AlphaSymbolic.core.gpu.optimization import GPUOptimizer


def test_fused_shape_gate_matches_compiled_kernel_limits():
    vm = CudaRPNVM.__new__(CudaRPNVM)
    vm.num_vars = 4
    population = torch.zeros((2, 256), dtype=torch.uint8)
    x = torch.zeros((4, 1024), dtype=torch.float32)
    assert vm.supports_fused_shape(population, x)

    assert not vm.supports_fused_shape(
        population, torch.zeros((5, 1024), dtype=torch.float32))
    assert not vm.supports_fused_shape(
        population, torch.zeros((3, 1024), dtype=torch.float32))
    assert not vm.supports_fused_shape(
        torch.zeros((2, 257), dtype=torch.uint8), x)
    assert not vm.supports_fused_shape(
        population, torch.zeros((4, 1025), dtype=torch.float32))

    vm.num_vars = 5
    assert not vm.supports_fused_shape(population, x)

    vm.num_vars = 4
    assert not vm.supports_fused_shape(
        torch.zeros((0, 256), dtype=torch.uint8), x)
    assert not vm.supports_fused_shape(
        population, torch.zeros((4, 0), dtype=torch.float32))


def test_direct_fused_launcher_rejects_grammar_variable_mismatch():
    vm = CudaRPNVM.__new__(CudaRPNVM)
    vm.num_vars = 2

    with pytest.raises(ValueError, match=r"shape \[2, D\]"):
        vm._launch_fused(
            None, torch.zeros((1, 8)), None, None, None, 0, 0
        )


@pytest.mark.skipif(not torch.cuda.is_available() or rpn_cuda is None,
                    reason="CUDA extension is required")
@pytest.mark.parametrize("samples", [1, 17, 32])
@pytest.mark.parametrize("strict_mode", [0, 1])
def test_block_and_warp_eval_agree(samples, strict_mode):
    grammar = GPUGrammar(num_variables=1)
    vm = CudaRPNVM(grammar, torch.device("cuda"))
    batch, length, constants_count = 512, 64, 8
    pop = torch.full((batch, length), PAD_ID, dtype=torch.uint8, device="cuda")

    x_id = grammar.token_to_id[grammar.active_variables[0]]
    c_id = grammar.token_to_id["C"]
    add_id = grammar.token_to_id["+"]
    mul_id = grammar.token_to_id["*"]
    log_id = grammar.token_to_id["log"]
    exp_id = grammar.token_to_id["exp"]

    # Mix normal, constant-bearing, strict-domain-invalid, and overflow-prone RPN.
    programs = (
        [x_id],
        [x_id, x_id, mul_id],
        [c_id, x_id, add_id],
        [x_id, log_id],
        [x_id, exp_id],
    )
    for row in range(batch):
        program = programs[row % len(programs)]
        pop[row, :len(program)] = torch.tensor(program, dtype=torch.uint8, device="cuda")

    x = torch.linspace(-2.0, 100.0, samples, dtype=torch.float32, device="cuda").reshape(1, -1)
    y = torch.zeros(samples, dtype=torch.float32, device="cuda")
    constants = torch.randn(batch, constants_count, dtype=torch.float32, device="cuda")
    block = torch.empty(batch, dtype=torch.float32, device="cuda")
    warp = torch.empty_like(block)

    vm._launch_fused(pop, x, constants, y, block, strict_mode, 0)
    vm._launch_fused(pop, x, constants, y, warp, strict_mode, 1)
    torch.cuda.synchronize()

    invalid_block = block >= 1e14
    invalid_warp = warp >= 1e14
    assert torch.equal(invalid_block, invalid_warp)
    valid = ~invalid_block
    assert torch.allclose(block[valid], warp[valid], rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available() or rpn_cuda is None,
                    reason="CUDA extension is required")
@pytest.mark.parametrize("samples", [33, 200, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("strict_mode", [0, 1])
def test_fused_and_classic_eval_agree(samples, dtype, strict_mode):
    grammar = GPUGrammar(num_variables=1)
    evaluator = GPUEvaluator(grammar, torch.device("cuda"), dtype=dtype)
    batch, length = 64, 128
    pop = torch.full((batch, length), PAD_ID, dtype=torch.uint8, device="cuda")
    x_id = grammar.token_to_id[grammar.active_variables[0]]
    programs = (
        [x_id],
        [x_id, x_id, grammar.token_to_id["*"]],
        [grammar.token_to_id["C"], x_id, grammar.token_to_id["+"]],
        [x_id, grammar.token_to_id["log"]],
        [x_id, grammar.token_to_id["exp"]],
    )
    for row in range(batch):
        program = programs[row % len(programs)]
        pop[row, :len(program)] = torch.tensor(program, dtype=torch.uint8, device="cuda")
    x = torch.linspace(-2.0, 100.0, samples, dtype=dtype, device="cuda").reshape(1, -1)
    y = torch.zeros(samples, dtype=dtype, device="cuda")
    constants = torch.randn(batch, 8, dtype=dtype, device="cuda")

    old_flag = GpuGlobals.CUDA_FUSED_EVOLVE_SCORE
    try:
        GpuGlobals.CUDA_FUSED_EVOLVE_SCORE = True
        fused = evaluator.evaluate_batch(pop, x, y, constants, strict_mode=strict_mode).clone()
        GpuGlobals.CUDA_FUSED_EVOLVE_SCORE = False
        classic = evaluator.evaluate_batch(pop, x, y, constants, strict_mode=strict_mode).clone()
    finally:
        GpuGlobals.CUDA_FUSED_EVOLVE_SCORE = old_flag

    invalid_fused = fused >= (1e14 if dtype == torch.float32 else 1e99)
    invalid_classic = classic >= (1e14 if dtype == torch.float32 else 1e99)
    assert torch.equal(invalid_fused, invalid_classic)
    valid = ~invalid_fused
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    assert torch.allclose(fused[valid], classic[valid], rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available() or rpn_cuda is None,
                    reason="CUDA extension is required")
def test_more_than_four_variables_uses_classic_path_without_substitution():
    """x4 must be evaluated as x4, never silently replaced with zero."""
    grammar = GPUGrammar(num_variables=5)
    evaluator = GPUEvaluator(grammar, torch.device("cuda"), dtype=torch.float32)
    batch, length, samples = 16, 64, 41
    pop = torch.full((batch, length), PAD_ID, dtype=torch.uint8, device="cuda")
    pop[:, 0] = grammar.token_to_id["x4"]

    x = torch.randn(5, samples, dtype=torch.float32, device="cuda")
    y = x[4].clone()
    constants = torch.empty(batch, 0, dtype=torch.float32, device="cuda")

    def fused_must_not_run(*args, **kwargs):
        raise AssertionError("five-variable workloads must use the classic evaluator")

    evaluator.vm.eval_fused = fused_must_not_run
    old_flag = GpuGlobals.CUDA_FUSED_EVOLVE_SCORE
    try:
        GpuGlobals.CUDA_FUSED_EVOLVE_SCORE = True
        rmse = evaluator.evaluate_batch(pop, x, y, constants)
    finally:
        GpuGlobals.CUDA_FUSED_EVOLVE_SCORE = old_flag

    assert torch.allclose(rmse, torch.zeros_like(rmse), atol=1e-7, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available() or rpn_cuda is None,
                    reason="CUDA extension is required")
def test_direct_fused_call_rejects_more_than_four_variables():
    grammar = GPUGrammar(num_variables=5)
    vm = CudaRPNVM(grammar, torch.device("cuda"))
    pop = torch.full((1, 8), PAD_ID, dtype=torch.uint8, device="cuda")
    x = torch.zeros(5, 4, dtype=torch.float32, device="cuda")
    y = torch.zeros(4, dtype=torch.float32, device="cuda")
    constants = torch.empty(1, 0, dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="at most 4 variables"):
        vm.eval_fused(pop, x, constants, y)

    out = torch.empty(1, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="supports 1\\.\\.4 variables"):
        vm._launch_fused(pop, x, constants, y, out, 0, 0)


@pytest.mark.skipif(not torch.cuda.is_available() or rpn_cuda is None,
                    reason="CUDA extension is required")
def test_fused_pso_dtype_gate_keeps_float64_on_safe_fallback():
    optimizer = GPUOptimizer.__new__(GPUOptimizer)
    optimizer._has_fused_pso = True
    population = torch.zeros((2, 32), dtype=torch.uint8, device="cuda")

    x32 = torch.zeros((1, 8), dtype=torch.float32, device="cuda")
    y32 = torch.zeros(8, dtype=torch.float32, device="cuda")
    c32 = torch.zeros((2, 4), dtype=torch.float32, device="cuda")
    assert optimizer._can_use_fused_pso(population, c32, x32, y32, 16)

    assert not optimizer._can_use_fused_pso(
        population, c32.double(), x32.double(), y32.double(), 16)
