import sys
import os
import unittest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from warpsymbolic.gpu.grammar import GPUGrammar
from warpsymbolic.gpu.operators import GPUOperators
from warpsymbolic.gpu.optimization import GPUOptimizer
from warpsymbolic.gpu.config import GpuGlobals

PAD_ID = 0

class TestGPUCoreIntelligenceBugs(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.grammar = GPUGrammar(num_variables=1)
        self.ops = GPUOperators(self.grammar, self.device, pop_size=10, max_len=20)
        self.pop_dtype = self.ops.pop_dtype
        
    def test_depth_fair_sample_all_invalid(self):
        B, L = 2, 20
        parents = torch.full((B, L), PAD_ID, dtype=self.pop_dtype, device=self.device)
        x0_id = self.grammar.token_to_id.get('x0', 1)
        parents[:, 0] = x0_id
        
        valid_mask = torch.zeros((B, L), dtype=torch.bool, device=self.device)
        
        try:
            end_idx = self.ops._depth_fair_sample(parents, valid_mask)
            self.assertTrue(end_idx.shape == (B,))
        except Exception as e:
            self.fail(f"_depth_fair_sample crashed with all-False valid_mask: {e}")

    def test_crossover_all_invalid_mask(self):
        GpuGlobals.DEPTH_FAIR_CROSSOVER = False
        B, L = 4, 10
        parents = torch.full((B, L), PAD_ID, dtype=self.pop_dtype, device=self.device)
        constants = torch.zeros((B, 1), dtype=self.ops.dtype, device=self.device)
        
        x0_id = self.grammar.token_to_id.get('x0', 1)
        parents[:, 0] = x0_id
        
        try:
            new_parents, new_consts = self.ops.crossover_population(parents, constants, crossover_rate=1.0)
            self.assertEqual(new_parents.shape, (B, L))
        except Exception as e:
            self.fail(f"crossover_population crashed with trivial trees: {e}")

    def test_subtree_mutation_max_len_overflow(self):
        B, L = 2, 8
        self.ops = GPUOperators(self.grammar, self.device, pop_size=B, max_len=L)
        parents = torch.full((B, L), PAD_ID, dtype=self.ops.pop_dtype, device=self.device)
        constants = torch.zeros((B, 1), dtype=self.ops.dtype, device=self.device)
        
        # Fill it up to L. e.g. x0 x0 + x0 + x0 + 
        x0_id = self.grammar.token_to_id.get('x0', 1)
        plus_id = self.grammar.token_to_id.get('+', 2)
        
        seq = [x0_id, x0_id, plus_id, x0_id, plus_id, x0_id, plus_id, PAD_ID]
        parents[:, :] = torch.tensor(seq, dtype=self.ops.pop_dtype, device=self.device)
        
        new_parents, new_consts = self.ops.subtree_mutation(parents, constants, mutation_rate=1.0)
        self.assertEqual(new_parents.shape, (B, L))
        is_valid = self.ops._validate_rpn_batch(new_parents)
        self.assertTrue(is_valid.all(), "Subtree mutation produced invalid RPN for full trees.")

    def test_nano_pso_nan_velocity(self):
        """
        Test if nano_pso handles extreme values in constants which could cause NaNs
        in velocity updating, thus ruining the constant.
        """
        B, K = 2, 2
        population = torch.full((B, 10), PAD_ID, dtype=self.pop_dtype, device=self.device)
        constants = torch.tensor([[1e30, -1e30], [0.0, 0.0]], dtype=torch.float32, device=self.device)
        x = torch.zeros((10, 1), dtype=torch.float32, device=self.device)
        y = torch.zeros((10,), dtype=torch.float32, device=self.device)
        
        from warpsymbolic.gpu.evaluation import GPUEvaluator
        evaluator = GPUEvaluator(self.grammar, self.device, dtype=torch.float32)
        optimizer = GPUOptimizer(evaluator, self.ops, self.device, dtype=torch.float32)
        
        try:
            best_c, best_f = optimizer.nano_pso(population, constants, x, y, steps=10)
            self.assertFalse(torch.isnan(best_c).any(), "nano_pso returned NaN for extreme starting velocities.")
        except Exception as e:
            self.fail(f"nano_pso crashed: {e}")

if __name__ == '__main__':
    unittest.main()
