import os
import sys

import torch


_test_dir = os.path.dirname(os.path.abspath(__file__))
_alpha_symbolic_dir = os.path.dirname(_test_dir)
_project_root = os.path.dirname(_alpha_symbolic_dir)
sys.path.insert(0, _project_root)
sys.path.insert(0, _alpha_symbolic_dir)


def test_library_update_learns_when_block_len_exceeds_formula_len():
    from warpsymbolic.gpu.config import GpuGlobals
    from warpsymbolic.gpu.grammar import GPUGrammar
    from warpsymbolic.gpu.library_learning import LibraryLearner

    original_max_block_len = GpuGlobals.LIBRARY_MAX_BLOCK_LEN
    try:
        GpuGlobals.LIBRARY_MAX_BLOCK_LEN = 8

        grammar = GPUGrammar(num_variables=1, use_globals=False)
        learner = LibraryLearner(
            grammar=grammar,
            pop_size=16,
            max_len=4,
            device="cpu",
            dtype=torch.float32,
            capacity=64,
        )

        x0_id = grammar.token_to_id["x0"]
        population = torch.full((16, 4), x0_id, dtype=torch.int32)
        fitness = torch.linspace(0.1, 1.6, 16, dtype=torch.float32)

        learner.update(population, fitness)

        assert learner.size > 0, (
            "LibraryLearner did not store patterns when max_block_len exceeds formula length. "
            "This indicates update failed silently instead of adapting block length."
        )

        sampled = learner.sample(k=4)
        assert sampled is not None and sampled.shape[1] == learner.max_block_len
    finally:
        GpuGlobals.LIBRARY_MAX_BLOCK_LEN = original_max_block_len
