"""Regressions for the 2026-10 GPU engine audit (semantics of evolution operators)."""
import contextlib
import io
import math

import pytest
import torch

from warpsymbolic.gpu.config import GpuGlobals
from warpsymbolic.gpu.cuda_vm import rpn_cuda
from warpsymbolic.gpu.grammar import GPUGrammar, PAD_ID
from warpsymbolic.gpu.evaluation import GPUEvaluator

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or rpn_cuda is None,
    reason="CUDA extension is required",
)


@pytest.fixture
def engine():
    from warpsymbolic.gpu.engine import TensorGeneticEngine

    old_len = GpuGlobals.MAX_FORMULA_LENGTH
    GpuGlobals.MAX_FORMULA_LENGTH = 48
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield TensorGeneticEngine(pop_size=4000, num_variables=1, n_islands=4, max_constants=5)
    finally:
        GpuGlobals.MAX_FORMULA_LENGTH = old_len


def _data():
    x = torch.linspace(-1, 1, 64, device="cuda").unsqueeze(1)
    return x, x[:, 0] ** 2


def test_pow_with_negative_base_and_integer_exponent_is_valid():
    grammar = GPUGrammar(num_variables=1)
    evaluator = GPUEvaluator(grammar, torch.device("cuda"), dtype=torch.float32)
    ids = [grammar.token_to_id[t] for t in ("x0", "3", "pow")]
    pop = torch.full((1, 8), PAD_ID, dtype=torch.uint8, device="cuda")
    pop[0, :3] = torch.tensor(ids, dtype=torch.uint8, device="cuda")
    x = torch.tensor([[-2.0, -0.5, 1.5]], device="cuda")
    y = x[0] ** 3
    for strict in (0, 1):
        rmse = evaluator.evaluate_batch(pop, x, y, None, strict_mode=strict)
        assert float(rmse[0]) < 1e-5, (strict, float(rmse[0]))


def test_transcendentals_are_accurate_for_large_arguments():
    grammar = GPUGrammar(num_variables=1)
    evaluator = GPUEvaluator(grammar, torch.device("cuda"), dtype=torch.float32)
    pop = torch.full((1, 4), PAD_ID, dtype=torch.uint8, device="cuda")
    pop[0, 0] = grammar.token_to_id["x0"]
    pop[0, 1] = grammar.token_to_id["sin"]
    x = torch.tensor([[1000.0, 12345.678]], device="cuda")
    preds, _, _ = evaluator.vm.eval(pop, x, None)
    ref = torch.sin(x[0].double())
    assert torch.allclose(preds[0].double(), ref, rtol=0, atol=1e-6)


def test_euler_constant_is_exact_in_float64():
    grammar = GPUGrammar(num_variables=1)
    evaluator = GPUEvaluator(grammar, torch.device("cuda"), dtype=torch.float64)
    pop = torch.full((1, 4), PAD_ID, dtype=torch.uint8, device="cuda")
    pop[0, 0] = grammar.token_to_id["e"]
    preds, _, _ = evaluator.vm.eval(pop, torch.zeros(1, 1, dtype=torch.float64, device="cuda"), None)
    assert float(preds[0, 0]) == math.e


def test_copies_keep_their_parent_constants(engine):
    """Without crossover or mutation every child is an exact copy of its parent,
    including constants (SBX only blends parents with identical structure)."""
    x, y = _data()
    pop = engine.operators.generate_random_population(4000)
    consts = torch.empty(4000, 5, device="cuda").uniform_(-3, 3)
    fit = torch.rand(4000, device="cuda")
    new_pop, new_c, _ = engine.evolve_generation_cuda(
        pop, consts, fit, None, x, y, None, mutation_rate=0.0, crossover_rate=0.0,
        tournament_size=3, pso_steps=0, pso_particles=2)
    parents = {}
    for i, row in enumerate(pop.cpu().numpy()):
        parents.setdefault(bytes(row), []).append(i)
    checked = 0
    for i, row in enumerate(new_pop.cpu().numpy()):
        matches = parents.get(bytes(row))
        assert matches is not None, "copies must not change structure"
        if len(matches) == 1:  # unique structure: the parent is unambiguous
            checked += 1
            assert torch.equal(new_c[i], consts[matches[0]])
    assert checked > 100


def test_structural_mutation_never_truncates_programs(engine):
    x, y = _data()
    old = GpuGlobals.INIT_MAX_LENGTH
    GpuGlobals.INIT_MAX_LENGTH = 48
    try:
        candidates = engine.operators.generate_random_population(200_000)
        bank = engine.operators.generate_random_population(5000)
    finally:
        GpuGlobals.INIT_MAX_LENGTH = old
    lengths = (candidates != PAD_ID).sum(1)
    pop = candidates[lengths >= 25][:4000].contiguous()
    assert pop.shape[0] == 4000
    consts = torch.zeros(4000, 5, device="cuda")
    fit = torch.rand(4000, device="cuda")
    new_pop, _, _ = engine.evolve_generation_cuda(
        pop, consts, fit, None, x, y, bank, mutation_rate=1.0, crossover_rate=0.0,
        tournament_size=3, pso_steps=0, pso_particles=2)
    assert bool(engine.operators._validate_rpn_batch(new_pop).all())


def test_lexicase_cases_index_the_error_subsample(engine):
    """abs_errors may have fewer columns than the dataset (lexicase sub-sampling)."""
    x = torch.linspace(-1, 1, 1000, device="cuda").unsqueeze(1)
    y = x[:, 0] ** 2
    pop = engine.operators.generate_random_population(4000)
    consts = torch.zeros(4000, 5, device="cuda")
    fit = torch.rand(4000, device="cuda")
    abs_err = torch.rand(4000, 128, device="cuda")
    old = GpuGlobals.USE_LEXICASE_SELECTION
    GpuGlobals.USE_LEXICASE_SELECTION = True
    try:
        new_pop, _, _ = engine.evolve_generation_cuda(
            pop, consts, fit, abs_err, x, y, None, mutation_rate=0.1, crossover_rate=0.5,
            tournament_size=3, pso_steps=0, pso_particles=2)
        torch.cuda.synchronize()
    finally:
        GpuGlobals.USE_LEXICASE_SELECTION = old
    assert new_pop.shape == pop.shape


def test_random_formulas_mostly_use_a_variable(engine):
    pop = engine.operators.generate_random_population(100_000)
    x_id = engine.grammar.token_to_id["x0"]
    with_var = (pop == x_id).any(dim=1).float().mean().item()
    lengths = (pop != PAD_ID).sum(1)
    assert with_var > 0.6
    assert (lengths == 2).float().mean().item() < 0.2
    assert bool(engine.operators._validate_rpn_batch(pop).all())


def test_perturbation_keeps_out_of_range_constants():
    c = torch.tensor([[100.0, -60.0, 3.0]], device="cuda")
    rpn_cuda.constant_perturbation(c, 1.0, 0.01, -25.0, 25.0, 123)
    assert float(c[0, 0]) > 90.0 and float(c[0, 1]) < -50.0


def test_dedup_removes_exact_clones_only(engine):
    pop = engine.operators.generate_random_population(4000)
    pop[1] = pop[0]
    pop[2] = pop[0]
    consts = torch.zeros(4000, 5, device="cuda")
    unique_before = torch.unique(pop, dim=0).shape[0]
    out, _, n_dups = engine.operators.deduplicate_population(pop.clone(), consts.clone())
    n_dups = int(n_dups)
    assert n_dups == 4000 - unique_before
