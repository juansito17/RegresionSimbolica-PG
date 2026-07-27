from collections import Counter

import torch

from AlphaSymbolic.core.gpu.config import GpuGlobals
from AlphaSymbolic.core.gpu.grammar import GPUGrammar
from AlphaSymbolic.core.gpu.operators import GPUOperators


def _counts_by_name(operators, ids):
    return Counter(operators.grammar.id_to_token[int(token)] for token in ids.cpu())


def test_full_profile_applies_per_binary_operator_weights():
    grammar = GPUGrammar(num_variables=1, use_globals=True)
    operators = GPUOperators(grammar, torch.device("cpu"), pop_size=32)
    counts = _counts_by_name(operators, operators.arity_2_ids)

    # The configured + prior is twice the pow prior (0.20 vs 0.10).
    if "+" in counts and "pow" in counts:
        assert counts["+"] == 2 * counts["pow"]

    # Division's configured mass is distinct from multiplication's; this
    # catches the former uniform-within-arity implementation.
    if "/" in counts and "*" in counts:
        assert counts["*"] > counts["/"]


def test_gamma_family_shares_one_configured_probability_mass():
    grammar = GPUGrammar(num_variables=1, use_globals=True)
    operators = GPUOperators(grammar, torch.device("cpu"), pop_size=32)
    counts = _counts_by_name(operators, operators.arity_1_ids)

    enabled = [name for name in ("gamma", "lgamma") if name in counts]
    if len(enabled) == 2:
        gamma_index = 13
        expected_family_slots = round(GpuGlobals.OPERATOR_WEIGHTS[gamma_index] * 200)
        assert counts["gamma"] + counts["lgamma"] == expected_family_slots


def test_uniform_profile_restores_one_slot_per_enabled_operator():
    grammar = GPUGrammar(num_variables=3, use_globals=True)
    operators = GPUOperators(grammar, torch.device("cpu"), pop_size=32)

    operators.set_sampling_profile("full_uniform")

    unary_counts = _counts_by_name(operators, operators.arity_1_ids)
    binary_counts = _counts_by_name(operators, operators.arity_2_ids)
    assert set(unary_counts.values()) == {1}
    assert set(binary_counts.values()) == {1}


def test_runtime_operator_toggle_receives_nonzero_sampling_mass(monkeypatch):
    # The web UI changes these booleans after config.py has already been
    # imported. The old positional list kept sin at zero forever.
    monkeypatch.setattr(GpuGlobals, "USE_OP_SIN", True)
    grammar = GPUGrammar(num_variables=1, use_globals=True)
    operators = GPUOperators(grammar, torch.device("cpu"), pop_size=32)
    counts = _counts_by_name(operators, operators.arity_1_ids)

    assert counts["sin"] > 0


def test_gamma_and_lgamma_web_toggles_are_independent(monkeypatch):
    monkeypatch.setattr(GpuGlobals, "USE_OP_GAMMA", False)
    monkeypatch.setattr(GpuGlobals, "USE_OP_LGAMMA", True)

    grammar = GPUGrammar(num_variables=1, use_globals=True)

    assert "gamma" not in grammar.operators
    assert "lgamma" in grammar.operators
