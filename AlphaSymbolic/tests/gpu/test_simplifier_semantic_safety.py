import torch

from AlphaSymbolic.core.gpu.gpu_simplifier import GPUSymbolicSimplifier
from AlphaSymbolic.core.gpu.grammar import GPUGrammar, PAD_ID


def _row(grammar, tokens, length=16):
    ids = [grammar.token_to_id[token] for token in tokens]
    return torch.tensor(
        [ids + [PAD_ID] * (length - len(ids))],
        dtype=grammar.dtype,
    )


def test_parametric_formula_is_not_rewritten_without_constant_remapping():
    grammar = GPUGrammar(num_variables=1, use_globals=False)
    simplifier = GPUSymbolicSimplifier(grammar, torch.device("cpu"))
    population = _row(grammar, ["C", "x0", "+"])
    constants = torch.tensor([[7.5, -3.0]], dtype=torch.float64)

    simplified, simplified_constants, n_changed = simplifier.simplify_batch(
        population, constants
    )

    assert n_changed == 0
    assert torch.equal(simplified, population)
    assert torch.equal(simplified_constants, constants)


def test_log_exp_chain_is_not_assumed_globally_invertible():
    grammar = GPUGrammar(num_variables=1, use_globals=False)
    simplifier = GPUSymbolicSimplifier(grammar, torch.device("cpu"))
    population = _row(grammar, ["x0", "log", "exp"])

    simplified, _, _ = simplifier.simplify_batch(population)
    active = simplified[0][simplified[0] != PAD_ID].tolist()

    assert active == [
        grammar.token_to_id["x0"],
        grammar.token_to_id["log"],
        grammar.token_to_id["exp"],
    ]
