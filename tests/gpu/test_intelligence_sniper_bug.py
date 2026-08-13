import sys
import os
import unittest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from warpsymbolic.gpu.grammar import GPUGrammar
from warpsymbolic.gpu.operators import GPUOperators
from warpsymbolic.gpu.evaluation import GPUEvaluator
from warpsymbolic.gpu.optimization import GPUOptimizer
from warpsymbolic.gpu.engine import TensorGeneticEngine
from warpsymbolic.gpu.config import GpuGlobals

class TestSniperBugs(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.grammar = GPUGrammar(num_variables=1)
        self.ops = GPUOperators(self.grammar, self.device, pop_size=2, max_len=20)
        self.evaluator = GPUEvaluator(self.grammar, self.device)
        self.optimizer = GPUOptimizer(self.evaluator, self.ops, self.device)
        
        self.engine = TensorGeneticEngine(device=self.device, pop_size=2, n_islands=1)
        # Enable Sniper
        GpuGlobals.USE_SNIPER = True
        

    def test_sniper_multivar_flatten_bug(self):
        """
        Bug Hypothesis: Sniper.run() blindly flattens x_data, so if x_data is [N, Vars] where Vars > 1, 
        it gets flattened to [N*Vars], causing a dimension mismatch in lstsq against y_t of size N.
        """
        from warpsymbolic.gpu.sniper import Sniper
        sniper = Sniper(self.device)
        
        # 2 variables
        x_t = torch.rand(10, 2, device=self.device)
        # target depends only on x0 to make it solvable if it were split correctly
        y_t = x_t[:, 0] * 3.5 + 1.2
        
        try:
            res = sniper.run(x_t, y_t)
            # Should not crash internally. If it does, sniper.run catches it and returns None.
            # But wait! If it returns None due to internal exception, that means it failed to find a simple linear pattern
            # when it clearly should have.
            self.assertIsNotNone(res, "Sniper returned None for a perfect linear match. Likely crashed internally due to flatten().")
        except Exception as e:
            self.fail(f"Sniper.run crashed: {e}")

if __name__ == '__main__':
    unittest.main()
