import os
import sys
from unittest.mock import patch

import torch


_test_dir = os.path.dirname(os.path.abspath(__file__))
_alpha_symbolic_dir = os.path.dirname(_test_dir)
_project_root = os.path.dirname(_alpha_symbolic_dir)
sys.path.insert(0, _project_root)
sys.path.insert(0, _alpha_symbolic_dir)


class _MockEvaluator:
    def __init__(self, n_cases: int):
        self.n_cases = n_cases

    def evaluate_batch_full(self, sub_pop, x, y_target, sub_c, strict_mode=0, force_f32=False):
        n = sub_pop.shape[0]
        return torch.ones((n, self.n_cases), dtype=torch.float32)


def test_epsilon_lexicase_tie_break_is_not_positional():
    from warpsymbolic.gpu.engine import TensorGeneticEngine

    torch.manual_seed(1234)

    engine = TensorGeneticEngine(device='cpu', pop_size=8, max_len=6, n_islands=1, num_variables=1)
    engine.evaluator = _MockEvaluator(n_cases=4)

    population = torch.zeros((8, 6), dtype=torch.uint8)
    constants = torch.zeros((8, 3), dtype=torch.float64)
    x = torch.zeros((1, 4), dtype=torch.float64)
    y = torch.zeros(4, dtype=torch.float64)

    n_parents = 300
    fixed_rand_idx = torch.stack(
        [
            torch.zeros(n_parents, dtype=torch.long),
            torch.ones(n_parents, dtype=torch.long),
        ],
        dim=1,
    )

    with patch('torch.randint', return_value=fixed_rand_idx):
        winners = engine.epsilon_lexicase_selection(
            population=population,
            n_parents=n_parents,
            x=x,
            y_target=y,
            constants=constants,
            tour_size=2,
        )

    unique = set(winners.tolist())
    assert unique == {0, 1}, (
        "En empate total de lexicase, los ganadores deben elegirse al azar entre "
        "los candidatos activos, no sesgados por posición."
    )
