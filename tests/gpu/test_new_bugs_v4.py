import sys
import os
import unittest
import torch

# Setup paths
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_ALPHA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ALPHA_ROOT not in sys.path:
    sys.path.insert(0, _ALPHA_ROOT)

from warpsymbolic.gpu.grammar import GPUGrammar
from warpsymbolic.gpu.engine import TensorGeneticEngine
from warpsymbolic.gpu.config import GpuGlobals

def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class TestN10LexicaseNaNBug(unittest.TestCase):
    def setUp(self):
        self.device = _get_device()
        self.engine = TensorGeneticEngine(device=self.device, pop_size=10, max_len=10, n_islands=1)

    def test_lexicase_nan_bias(self):
        # We simulate epsilon_lexicase_selection with a candidate error matrix containing NaNs
        n_parents = 2
        tour_size = 3
        N_cases = 5
        
        # Create dummy population and constants
        B = n_parents * tour_size
        population = torch.zeros((B, 10), dtype=torch.long, device=self.device)
        constants = torch.zeros((B, 5), dtype=torch.float64, device=self.device)
        x = torch.randn(1, N_cases, device=self.device, dtype=torch.float64)
        y_target = torch.randn(N_cases, device=self.device, dtype=torch.float64)
        
        # Mock evaluator to return controlled errors with NaNs
        class MockEvaluator:
            def evaluate_batch_full(self, pop, x, y, c, strict_mode=0):
                # We return errs where the FIRST candidate has NaN errors
                errs = torch.zeros((pop.shape[0], y.shape[0]), device=x.device, dtype=torch.float32)
                errs[0, :] = float('nan') # Candidate 0 is NaN
                errs[1, :] = 0.5 # Candidate 1 is decent
                errs[2, :] = 0.1 # Candidate 2 is the BEST
                
                # For the second tournament (indices 3, 4, 5)
                errs[3, :] = 0.5 # Candidate 3 is decent
                errs[4, :] = float('nan') # Candidate 4 is NaN
                errs[5, :] = 0.1 # Candidate 5 is the BEST
                return errs
                
        self.engine.evaluator = MockEvaluator()
        
        # Force a predictable tournament index generation
        original_randint = torch.randint
        def mock_randint(low, high, size, device=None):
            return torch.tensor([[0, 1, 2], [3, 4, 5]], device=device)
        
        import builtins
        torch.randint = mock_randint
        try:
            winners = self.engine.epsilon_lexicase_selection(population, n_parents, x, y_target, constants, tour_size=3)
        finally:
            torch.randint = original_randint
            
        # The bug previously meant Lexicase would return candidate 0 when NaNs were present.
        # Now it should pick candidate 2 and 5 (the ones with 0.1 error).
        self.assertEqual(winners[0].item(), 2, "Bug N10: Lexicase selected the wrong candidate due to NaN.")
        self.assertEqual(winners[1].item(), 5, "Bug N10: Lexicase selected the wrong candidate due to NaN.")

class TestN11TournamentNaNBug(unittest.TestCase):
    def setUp(self):
        self.device = _get_device()
        self.engine = TensorGeneticEngine(device=self.device, pop_size=6, max_len=10, n_islands=1)

    def test_tournament_nan_takeover(self):
        # Fitness array with NaNs
        # Tournament size 3. Population 6.
        # Tour 1: [NaN, 0.5, 0.1] -> Should pick 0.1 (index 2)
        # Tour 2: [0.5, NaN, 0.1] -> Should pick 0.1 (index 5)
        # Actually flat_fitness will be: [NaN, 0.5, 0.1, 0.5, NaN, 0.1]
        
        fitness = torch.tensor([float('nan'), 0.5, 0.1, 0.5, float('nan'), 0.1], device=self.device)
        population = torch.zeros((6, 10), dtype=torch.long, device=self.device)
        
        # Force a predictable tournament index generation for the test by mocking torch.randint
        original_randint = torch.randint
        def mock_randint(low, high, size, device):
            # Return indices exactly [0, 1, 2] for first parent, [3, 4, 5] for second
            return torch.tensor([[[0, 1, 2], [3, 4, 5], [0,1,2], [3,4,5], [0,1,2], [3,4,5]]], device=device)
        
        import builtins
        torch.randint = mock_randint
        try:
            winners = self.engine.tournament_selection_island(population, fitness, n_islands=1, tournament_size=3)
            # If bug exists, `torch.min` on [NaN, 0.5, 0.1] might pick the NaN (index 0).
            # Let's see what PyTorch actually does.
            # Usually torch.min([NaN, 0.5, 0.1]) -> NaN at index 0
            has_nan_winner = any(torch.isnan(fitness[w]) for w in winners)
            self.assertFalse(has_nan_winner, "Bug N11: Tournament selection picked a NaN individual as the winner.")
        finally:
            torch.randint = original_randint

if __name__ == "__main__":
    unittest.main()
