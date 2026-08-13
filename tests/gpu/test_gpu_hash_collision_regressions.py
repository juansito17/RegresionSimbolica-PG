import os
import sys

import torch


_test_dir = os.path.dirname(os.path.abspath(__file__))
_alpha_symbolic_dir = os.path.dirname(_test_dir)
_project_root = os.path.dirname(_alpha_symbolic_dir)
sys.path.insert(0, _project_root)
sys.path.insert(0, _alpha_symbolic_dir)


def _build_minimal_population(grammar, max_len: int):
    pad = 0
    x0 = grammar.token_to_id['x0']
    c = grammar.token_to_id['C']
    plus = grammar.token_to_id['+']
    mul = grammar.token_to_id['*']

    pop = torch.full((3, max_len), pad, dtype=torch.uint8)
    pop[0, :3] = torch.tensor([x0, x0, plus], dtype=torch.uint8)
    pop[1, :3] = torch.tensor([x0, c, plus], dtype=torch.uint8)
    pop[2, :3] = torch.tensor([x0, x0, mul], dtype=torch.uint8)
    return pop


def test_dedup_fallback_does_not_drop_unique_individuals_on_hash_collision():
    from warpsymbolic.gpu.grammar import GPUGrammar
    from warpsymbolic.gpu.operators import GPUOperators

    grammar = GPUGrammar(num_variables=1, use_globals=False)
    ops = GPUOperators(grammar, device='cpu', pop_size=3, max_len=8, num_variables=1, dtype=torch.float64)

    population = _build_minimal_population(grammar, max_len=8)
    constants = torch.zeros((3, 4), dtype=torch.float64)

    # Fuerza colisión de hash para todas las fórmulas en el fallback CPU.
    ops.dedup_weights = torch.zeros(population.shape[1], dtype=torch.long)

    pop_after, const_after, n_dups = ops.deduplicate_population(population.clone(), constants.clone())

    assert n_dups == 0, (
        "La deduplicación no debe marcar duplicados solo por colisión de hash cuando "
        "las estructuras RPN son distintas."
    )
    assert torch.equal(pop_after, population)
    assert torch.equal(const_after, constants)


def test_pattern_memory_hash_collision_keeps_distinct_patterns():
    from warpsymbolic.gpu.grammar import GPUGrammar
    from warpsymbolic.gpu.operators import GPUOperators
    from warpsymbolic.gpu.pattern_memory import PatternMemory

    grammar = GPUGrammar(num_variables=1, use_globals=False)
    ops = GPUOperators(grammar, device='cpu', pop_size=8, max_len=8, num_variables=1, dtype=torch.float64)
    memory = PatternMemory(device=torch.device('cpu'), operators=ops, max_patterns=8, dtype=torch.float64)

    x0 = grammar.token_to_id['x0']
    c = grammar.token_to_id['C']
    plus = grammar.token_to_id['+']
    mul = grammar.token_to_id['*']

    p1 = torch.tensor([[x0, x0, plus] + [0] * (memory.max_pattern_len - 3)], dtype=torch.uint8)
    p2 = torch.tensor([[x0, c, mul] + [0] * (memory.max_pattern_len - 3)], dtype=torch.uint8)

    # Primera actualización: almacena p1.
    memory._update_storage(
        p1,
        torch.tensor([12345], dtype=torch.long),
        torch.tensor([0.1], dtype=torch.float64),
        torch.tensor([3], dtype=torch.long),
    )

    # Segunda actualización: p2 tiene mismo hash pero estructura distinta.
    memory._update_storage(
        p2,
        torch.tensor([12345], dtype=torch.long),
        torch.tensor([0.2], dtype=torch.float64),
        torch.tensor([3], dtype=torch.long),
    )

    assert memory.n_patterns == 2, (
        "PatternMemory no debe fusionar patrones estructuralmente distintos "
        "solo porque comparten hash."
    )
    assert torch.equal(memory.patterns_tensor[0], p1[0])
    assert torch.equal(memory.patterns_tensor[1], p2[0])