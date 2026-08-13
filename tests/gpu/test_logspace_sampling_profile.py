import torch

from warpsymbolic.gpu import TensorGeneticEngine
from warpsymbolic.gpu.config import GpuGlobals
from warpsymbolic.gpu.grammar import GPUGrammar
from warpsymbolic.gpu.operators import GPUOperators


def _count_token(tensor: torch.Tensor, token_id: int) -> int:
    return int((tensor.cpu() == token_id).sum().item())


def test_logspace_sampling_profile_biases_sampling_without_changing_grammar():
    old_var_weight = getattr(GpuGlobals, "LOGSPACE_VARIABLE_TERMINAL_WEIGHT", 4)
    old_const_weight = getattr(GpuGlobals, "LOGSPACE_FREE_CONST_TERMINAL_WEIGHT", 4)
    old_terminal_prob = getattr(GpuGlobals, "LOGSPACE_TERMINAL_PROB", 0.32)
    try:
        GpuGlobals.LOGSPACE_VARIABLE_TERMINAL_WEIGHT = 3
        GpuGlobals.LOGSPACE_FREE_CONST_TERMINAL_WEIGHT = 5
        GpuGlobals.LOGSPACE_TERMINAL_PROB = 0.31

        grammar = GPUGrammar(num_variables=1)
        ops = GPUOperators(grammar, device=torch.device("cpu"), pop_size=8, max_len=16, num_variables=1)

        id_x0 = grammar.token_to_id["x0"]
        id_c = grammar.token_to_id["C"]
        id_exp = grammar.token_to_id.get("exp")
        id_log = grammar.token_to_id.get("log")
        id_neg = grammar.token_to_id.get("neg")
        id_mul = grammar.token_to_id.get("*")
        id_div = grammar.token_to_id.get("/")
        id_pow = grammar.token_to_id.get("pow")
        id_mod = grammar.token_to_id.get("%")

        full_terminals = ops.terminal_ids.clone()
        full_unary = ops.arity_1_ids.clone()
        full_weights = ops._sampling_category_weights()
        version_before = getattr(ops, "_sampling_profile_version", 0)

        ops.set_sampling_profile("log_algebraic")
        log_weights = ops._sampling_category_weights()

        assert _count_token(ops.terminal_ids, id_x0) == 3
        assert _count_token(ops.terminal_ids, id_c) == 5
        assert id_neg is not None and _count_token(ops.arity_1_ids, id_neg) >= 1
        if id_exp is not None:
            assert _count_token(ops.arity_1_ids, id_exp) == 0
        if id_log is not None:
            assert _count_token(ops.arity_1_ids, id_log) == 0
        assert id_mul is not None and _count_token(ops.arity_2_ids, id_mul) >= 4
        if id_div is not None:
            assert _count_token(ops.arity_2_ids, id_div) == 0
        if id_pow is not None:
            assert _count_token(ops.arity_2_ids, id_pow) == 0
        if id_mod is not None:
            assert _count_token(ops.arity_2_ids, id_mod) == 0

        # The grammar/evaluator vocabulary remains full; only sampling pools change.
        assert "exp" in grammar.token_to_id
        assert "log" in grammar.token_to_id
        assert getattr(ops, "_sampling_profile_version", 0) > version_before
        assert log_weights[0] == 0.31
        assert log_weights[1] < full_weights[1]
        assert log_weights[2] > log_weights[1]

        ops.set_sampling_profile("log_algebraic_rich")
        if id_div is not None:
            assert _count_token(ops.arity_2_ids, id_div) > 0
        if id_pow is not None:
            assert _count_token(ops.arity_2_ids, id_pow) > 0

        ops.set_sampling_profile("full")

        assert torch.equal(ops.terminal_ids, full_terminals)
        assert torch.equal(ops.arity_1_ids, full_unary)
    finally:
        GpuGlobals.LOGSPACE_VARIABLE_TERMINAL_WEIGHT = old_var_weight
        GpuGlobals.LOGSPACE_FREE_CONST_TERMINAL_WEIGHT = old_const_weight
        GpuGlobals.LOGSPACE_TERMINAL_PROB = old_terminal_prob


def test_transform_probe_prefers_simpler_representation():
    engine = TensorGeneticEngine(num_variables=1, pop_size=16, n_islands=1, max_len=16)
    x = torch.linspace(-2.5, 2.5, 31, device=engine.device, dtype=engine.dtype)
    y_poly = x * x + 2.0 * x + 6.0
    y_exp = torch.exp(0.055 * x * x * x - 0.18 * x * x + 0.62 * x + 1.2)

    assert engine._one_dim_poly_r2(x, y_poly, degree=3) > 0.999999
    assert engine._one_dim_poly_r2(x, y_poly, degree=3) > engine._one_dim_poly_r2(x, torch.log(y_poly), degree=3)
    assert engine._one_dim_poly_r2(x, torch.log(y_exp), degree=3) > 0.999999
    assert engine._one_dim_poly_r2(x, torch.log(y_exp), degree=3) > engine._one_dim_poly_r2(x, y_exp, degree=3)
