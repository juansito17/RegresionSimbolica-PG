import pytest
import torch

from AlphaSymbolic.core.gpu.cuda_vm import CudaRPNVM, rpn_cuda
from AlphaSymbolic.core.gpu.config import GpuGlobals
from AlphaSymbolic.core.gpu.evaluation import GPUEvaluator
from AlphaSymbolic.core.gpu.grammar import GPUGrammar, PAD_ID


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
