"""Linear scaling, Levenberg-Marquardt constant optimisation and fitness reuse."""
import contextlib
import io

import numpy as np
import pytest
import torch

from warpsymbolic.gpu.config import GpuGlobals
from warpsymbolic.gpu.cuda_vm import rpn_cuda

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or rpn_cuda is None or not hasattr(rpn_cuda, "lm_optimize"),
    reason="CUDA extension with lm_optimize is required",
)


@pytest.fixture
def engine():
    from warpsymbolic.gpu.engine import TensorGeneticEngine

    with contextlib.redirect_stdout(io.StringIO()):
        eng = TensorGeneticEngine(pop_size=4000, num_variables=2, n_islands=4, max_constants=5)
    yield eng
    eng.evaluator.linear_scaling = False


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.5, 3.0, size=(2, 128)).astype(np.float32)
    y = (3.0 * np.sin(x[0]) * x[1] + 2.0).astype(np.float32)
    return torch.tensor(x, device="cuda"), torch.tensor(y, device="cuda")


def _lstsq_rmse(f, y):
    if np.var(f) <= 1e-10 * float(np.mean(f)) ** 2 + 1e-30:
        res = y - y.mean()
    else:
        A = np.stack([np.ones_like(f), f], 1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ coef
    return float(np.sqrt(np.mean(res ** 2)))


@pytest.mark.parametrize("mode", ["warp", "block"])
def test_scaled_fitness_matches_least_squares(engine, data, mode):
    x, y = data
    vm = engine.evaluator.vm
    torch.manual_seed(1)
    pop = engine.operators.generate_random_population(2000)
    consts = torch.empty(2000, 5, device="cuda").uniform_(-3, 3)
    preds, sp, err = vm.eval(pop, x, consts, strict_mode=1)
    preds = preds.view(2000, -1).double().cpu().numpy()
    valid = ((sp.view(2000, -1) == 1) & (err.view(2000, -1) == 0)).all(1).cpu().numpy()
    valid &= np.isfinite(preds).all(1) & (np.abs(preds) < 1e6).all(1)
    old = GpuGlobals.CUDA_EVAL_MODE
    GpuGlobals.CUDA_EVAL_MODE = mode
    try:
        rmse = vm.eval_fused(pop, x, consts, y, strict_mode=1, scaled=True).double().cpu().numpy()
    finally:
        GpuGlobals.CUDA_EVAL_MODE = old
    y_np = y.double().cpu().numpy()
    scale = float(np.std(y_np))
    worst = max(abs(rmse[i] - _lstsq_rmse(preds[i], y_np)) for i in np.nonzero(valid)[0])
    assert valid.sum() > 500
    assert worst < 1e-5 * scale


def test_scaled_fit_detects_exact_solution_and_reports_coefficients(engine, data):
    x, y = data
    pop, consts = engine.load_population_from_strings(["sin(x0) * x1", "x0 + x1", "C"])
    consts = consts.float()
    consts[2, 0] = 7.0
    rmse, ab = engine.evaluator.vm.eval_fused(pop, x, consts, y, strict_mode=1, scaled=True, return_ab=True)
    assert float(rmse[0]) < 1e-6
    assert abs(float(ab[0, 0]) - 2.0) < 1e-4 and abs(float(ab[0, 1]) - 3.0) < 1e-4
    # A constant prediction is fitted by mean(y): slope 0, RMSE = std(y).
    assert float(ab[2, 1]) == 0.0
    assert abs(float(rmse[2]) - float(y.double().std(unbiased=False))) < 1e-4


def test_fitness_reuse_copies_identical_children_only(engine, data):
    x, y = data
    vm = engine.evaluator.vm
    torch.manual_seed(2)
    pop = engine.operators.generate_random_population(3000)
    consts = torch.empty(3000, 5, device="cuda").uniform_(-3, 3)
    parent = torch.randperm(3000, device="cuda")
    child_pop = pop[parent].clone()
    child_c = consts[parent].clone()
    # One ulp in a constant and a changed program must both force evaluation.
    child_c[5, 0] = torch.nextafter(child_c[5, 0], torch.tensor(1e9, device="cuda"))
    child_pop[6].fill_(child_pop[6, 0])
    sentinel = torch.full((3000,), 12345.0, device="cuda")
    reused = vm.eval_fused(child_pop, x, child_c, y, strict_mode=1, reuse=(parent, pop, consts, sentinel))
    full = vm.eval_fused(child_pop, x, child_c, y, strict_mode=1)
    copied = reused == 12345.0
    assert not bool(copied[5]) and not bool(copied[6])
    assert torch.equal(reused[~copied], full[~copied])
    assert int(copied.sum()) == 2998


def test_levenberg_marquardt_recovers_nonlinear_constants(engine, data):
    x, _ = data
    y = (2.0 * torch.sin(1.5 * x[0]) * x[1] - 0.5).float()
    pop, consts = engine.load_population_from_strings(["C * sin(C * x0) * x1 + C"])
    consts = consts.float()
    consts[0, :3] = torch.tensor([1.0, 1.2, 0.0], device="cuda")
    new_c, rmse = engine.evaluator.vm.lm_optimize(pop, consts, x, y, 30, -25, 25, strict_mode=1)
    assert float(rmse[0]) < 1e-5
    assert np.allclose(new_c[0, :3].cpu().numpy(), [2.0, 1.5, -0.5], atol=1e-3)


@pytest.mark.parametrize("formula,true_c,start_c", [
    ("C * exp(C * x0)", [1.5, -0.7], [1.0, -0.3]),
    ("log(C * x0 + C)", [2.0, 1.0], [1.5, 0.5]),
    ("sqrt(C * x0 + C)", [3.0, 0.5], [2.0, 1.0]),
    ("(x0 + C) / (x1 + C)", [1.0, 2.0], [0.5, 1.5]),
    ("x0 ^ C + C", [1.7, 0.3], [1.2, 0.0]),
    ("C * cos(x0 * C) + C * x1", [1.3, 0.8, -0.4], [1.0, 1.0, 0.0]),
])
def test_levenberg_marquardt_derivatives_per_operator(engine, data, formula, true_c, start_c):
    x, _ = data
    pop, consts = engine.load_population_from_strings([formula])
    consts = consts.float()
    k = len(true_c)
    consts[0, :k] = torch.tensor(true_c, device="cuda")
    y, _, _ = engine.evaluator.vm.eval(pop, x, consts, strict_mode=1)
    y = y.view(-1).float().clone()
    consts[0, :k] = torch.tensor(start_c, device="cuda")
    start = engine.evaluator.vm.eval_fused(pop, x, consts, y, strict_mode=1)
    _, rmse = engine.evaluator.vm.lm_optimize(pop, consts, x, y, 40, -25, 25, strict_mode=1)
    assert float(rmse[0]) < 1e-4 * max(1.0, float(start[0])), (formula, float(start[0]), float(rmse[0]))


def test_levenberg_marquardt_never_worsens_and_matches_evaluator(engine, data):
    x, y = data
    vm = engine.evaluator.vm
    torch.manual_seed(3)
    pop = engine.operators.generate_random_population(4000)
    consts = torch.empty(4000, 5, device="cuda").uniform_(-3, 3)
    for scaled in (False, True):
        base = vm.eval_fused(pop, x, consts, y, strict_mode=1, scaled=scaled)
        new_c, rmse = vm.lm_optimize(pop, consts, x, y, 10, -25, 25, strict_mode=1, scaled=scaled)
        after = vm.eval_fused(pop, x, new_c, y, strict_mode=1, scaled=scaled)
        finite = base < 1e14
        assert not bool(((after > base * (1 + 1e-4) + 1e-6) & finite).any())
        ok = finite & (after < 1e14) & (rmse < 1e14)
        assert torch.allclose(rmse[ok], after[ok], rtol=1e-3, atol=1e-5)
        assert int((after[finite] < base[finite] * 0.999).sum()) > 100


def test_materialized_formula_reproduces_scaled_fitness(engine, data):
    x, y = data
    pop, consts = engine.load_population_from_strings(["sin(x0) * C + x1", "x0 * x1"])
    consts = consts.float()
    consts[0, 0] = 0.7
    engine.evaluator.linear_scaling = True
    try:
        scaled = engine.evaluator.evaluate_batch(pop, x, y, consts, strict_mode=1)
        mats = [engine._materialize_scaling(pop[i], consts[i], x, y) for i in range(2)]
    finally:
        engine.evaluator.linear_scaling = False
    for i, (rpn, c) in enumerate(mats):
        plain = engine.evaluator.evaluate_batch(rpn.unsqueeze(0), x, y, c.unsqueeze(0), strict_mode=1)
        assert abs(float(plain[0]) - float(scaled[i])) <= 1e-5 * float(y.std()) + 1e-6
        assert engine.rpn_to_infix(rpn, c) != "Invalid"


def test_scaled_run_returns_explicit_formula_and_resets_state():
    from warpsymbolic.gpu.engine import TensorGeneticEngine

    old = (GpuGlobals.USE_LINEAR_SCALING, GpuGlobals.USE_INITIAL_POP_CACHE)
    GpuGlobals.USE_LINEAR_SCALING = True
    GpuGlobals.USE_INITIAL_POP_CACHE = False
    try:
        torch.manual_seed(4)
        x = np.linspace(-2, 2, 64, dtype=np.float32)
        y = (5.0 * x ** 2 - 3.0).astype(np.float32)
        with contextlib.redirect_stdout(io.StringIO()):
            eng = TensorGeneticEngine(pop_size=20000, num_variables=1, n_islands=4, max_constants=5)
            formula = eng.run(x, y, timeout_sec=4)
        assert formula
        assert eng.evaluator.linear_scaling is False
        xt = torch.tensor(x, device="cuda").unsqueeze(0)
        yt = torch.tensor(y, device="cuda")
        res = eng.evaluator.validate_strict(eng.best_global_rpn.unsqueeze(0), xt, yt,
                                            eng.best_global_consts.unsqueeze(0).to(eng.dtype))
        assert float(res["rmse"][0]) < 1e-3, formula
    finally:
        GpuGlobals.USE_LINEAR_SCALING, GpuGlobals.USE_INITIAL_POP_CACHE = old


def test_adaptive_linear_scaling_switches_on_and_returns_valid_formula():
    from warpsymbolic.gpu.engine import TensorGeneticEngine

    old = (GpuGlobals.USE_LINEAR_SCALING, GpuGlobals.LINEAR_SCALING_TIME_FRACTION,
           GpuGlobals.USE_INITIAL_POP_CACHE)
    GpuGlobals.USE_LINEAR_SCALING = "adaptive"
    GpuGlobals.LINEAR_SCALING_TIME_FRACTION = 0.3
    GpuGlobals.USE_INITIAL_POP_CACHE = False
    try:
        torch.manual_seed(6)
        rng = np.random.default_rng(6)
        x = rng.uniform(0.3, 4.0, size=(128, 2)).astype(np.float32)
        y = (np.exp(-(x[:, 0] - 1) ** 2) / (1.2 + (x[:, 1] - 2.5) ** 2)).astype(np.float32)
        with contextlib.redirect_stdout(io.StringIO()):
            eng = TensorGeneticEngine(pop_size=20000, num_variables=2, n_islands=4, max_constants=5)
            formula = eng.run(x, y, timeout_sec=3)
        assert formula
        assert eng.last_run_metrics["linear_scaling_generation"] is not None
        assert eng.evaluator.linear_scaling is False
        xt = torch.tensor(x.T.copy(), device="cuda")
        yt = torch.tensor(y, device="cuda")
        res = eng.evaluator.validate_strict(eng.best_global_rpn.unsqueeze(0), xt, yt,
                                            eng.best_global_consts.unsqueeze(0).to(eng.dtype))
        assert abs(float(res["rmse"][0]) - eng.last_run_best_rmse) <= 1e-4 + 1e-3 * eng.last_run_best_rmse
    finally:
        (GpuGlobals.USE_LINEAR_SCALING, GpuGlobals.LINEAR_SCALING_TIME_FRACTION,
         GpuGlobals.USE_INITIAL_POP_CACHE) = old


def test_adaptive_scaling_waits_for_time_and_spares_near_exact_runs(engine):
    should = engine._should_enable_adaptive_scaling
    assert not should(1.0, 10, 5, 0, best_rmse=0.5, y_std=1.0)       # too early
    assert should(2.5, 10, 50, 0, best_rmse=0.5, y_std=1.0)          # 20 % of the budget
    assert not should(2.5, 10, 50, 0, best_rmse=5e-5, y_std=1.0)     # converging to an exact fit
    assert should(0.1, 10, 50, 1000, best_rmse=0.5, y_std=1.0)       # long stagnation


@pytest.mark.parametrize("value,expected", [
    (True, "on"), (False, "off"), ("adaptive", "adaptive"), ("ADAPTIVE", "adaptive"), ("off", "off"),
])
def test_linear_scaling_mode_parsing(value, expected):
    from warpsymbolic.gpu.engine import TensorGeneticEngine

    old = GpuGlobals.USE_LINEAR_SCALING
    GpuGlobals.USE_LINEAR_SCALING = value
    try:
        assert TensorGeneticEngine._linear_scaling_mode() == expected
    finally:
        GpuGlobals.USE_LINEAR_SCALING = old


def test_sympy_budget_skips_repeated_structures(engine):
    engine._sympy_inloop_spent = 0.0
    engine._sympy_inloop_last_struct = None
    rpn = torch.arange(10, dtype=torch.uint8, device="cuda")
    assert engine._sympy_inloop_allowed(rpn, elapsed=1.0)
    assert not engine._sympy_inloop_allowed(rpn, elapsed=1.0)
    other = rpn.flip(0)
    engine._sympy_inloop_spent = 10.0
    assert not engine._sympy_inloop_allowed(other, elapsed=1.0)
    engine._sympy_inloop_spent = 0.0
    assert engine._sympy_inloop_allowed(other, elapsed=1.0)


def test_structural_duplicate_mask_keeps_one_representative(engine):
    pop = engine.operators.generate_random_population(100)
    pop = torch.cat([pop, pop[:30], pop[:30]])
    mask = engine.operators.structural_duplicate_mask(pop)
    uniq = torch.unique(pop, dim=0).shape[0]
    assert int((~mask).sum()) == uniq


def test_dedup_mutate_mode_replaces_duplicates_with_valid_programs(engine, data):
    x, y = data
    old = (GpuGlobals.DEDUP_REPLACEMENT, GpuGlobals.PREVENT_DUPLICATES)
    GpuGlobals.DEDUP_REPLACEMENT = "mutate"
    GpuGlobals.PREVENT_DUPLICATES = True
    try:
        base = engine.operators.generate_random_population(200)
        pop = torch.cat([base, base]).contiguous()
        consts = torch.empty(400, 5, device="cuda").uniform_(-3, 3)
        pop, consts, n = engine.operators.deduplicate_population(pop, consts)
        assert n > 0
        rmse = engine.evaluator.vm.eval_fused(pop, x, consts.float(), y, strict_mode=0)
        assert int((rmse >= 1e14).sum()) <= int(0.05 * 400)
    finally:
        GpuGlobals.DEDUP_REPLACEMENT, GpuGlobals.PREVENT_DUPLICATES = old


def test_migration_moves_best_individuals_with_their_fitness(engine):
    torch.manual_seed(5)
    pop = engine.operators.generate_random_population(4000)
    consts = torch.empty(4000, 5, device="cuda").uniform_(-3, 3)
    fit = torch.rand(4000, device="cuda")
    island = engine.island_size
    mig = min(GpuGlobals.MIGRATION_SIZE, island // 2)
    best0 = torch.topk(fit[:island], mig, largest=False).indices
    best_rows = pop[best0].clone()
    best_fit = fit[best0].clone()
    engine._migration_count = 0  # next call uses the circular topology
    engine.migrate_islands(pop, consts, fit, carry_fitness=True)
    dest = slice(island, 2 * island)
    for row, f in zip(best_rows, best_fit):
        hits = (pop[dest] == row).all(dim=1) & (fit[dest] == f)
        assert bool(hits.any())


def test_string_loading_keeps_placeholder_and_literal_constants_aligned(engine):
    pop, consts = engine.load_population_from_strings(["C * sin(x0) + 2.5"])
    assert float(consts[0, 0]) == 0.0
    assert float(consts[0, 1]) == 2.5
