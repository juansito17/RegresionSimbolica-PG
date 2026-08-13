import os
import sys

import torch


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_ALPHASYMBOLIC_DIR = os.path.join(_REPO_ROOT, "src")
if _ALPHASYMBOLIC_DIR not in sys.path:
    sys.path.insert(0, _ALPHASYMBOLIC_DIR)
    sys.path.insert(0, _REPO_ROOT)

from warpsymbolic.gpu.pareto import ParetoOptimizer


def test_nonfinite_objectives_do_not_enter_first_front():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pareto = ParetoOptimizer(device=device)

    fitness = torch.tensor([0.10, float("nan"), 0.20, float("inf")], device=device, dtype=torch.float32)
    complexity = torch.tensor([2.0, 2.0, 3.0, 1.0], device=device, dtype=torch.float32)

    fronts, ranks = pareto.non_dominated_sort(fitness, complexity)
    first_front = set(fronts[0].tolist())

    assert 0 in first_front
    assert 1 not in first_front
    assert int(ranks[1].item()) > int(ranks[0].item())


def test_select_avoids_nan_when_finite_candidates_exist():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pareto = ParetoOptimizer(device=device)

    population = torch.arange(4, device=device, dtype=torch.long).unsqueeze(1)
    fitness = torch.tensor([0.10, float("nan"), 0.20, 0.30], device=device, dtype=torch.float32)
    complexity = torch.tensor([2.0, 2.0, 3.0, 4.0], device=device, dtype=torch.float32)

    selected = pareto.select(population, fitness, complexity, n_select=2)
    selected_ids = set(selected.tolist())

    assert 1 not in selected_ids
