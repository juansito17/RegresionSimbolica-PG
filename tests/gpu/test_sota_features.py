import torch
import unittest
import sys
import os

# Add project root to path
# We need to find the folder containing 'core'
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"DEBUG: Project Root: {project_root}")
repo_root = os.path.dirname(project_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from warpsymbolic.gpu.config import GpuGlobals
except ImportError:
    # If running from inside WarpSymbolic, maybe core is top level?
    sys.path.insert(0, os.path.join(project_root, 'WarpSymbolic'))
    from warpsymbolic.gpu.config import GpuGlobals

from warpsymbolic.gpu.grammar import GPUGrammar
from warpsymbolic.gpu.operators import GPUOperators
from warpsymbolic.gpu.library_learning import LibraryLearner
from warpsymbolic.gpu.pareto import ParetoOptimizer

class TestSOTAFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_available():
            cls.device = torch.device('cuda')
        else:
            print("WARNING: CUDA not available, testing on CPU (performance metrics invalid)")
            cls.device = torch.device('cpu')
        
        cls.pop_size = 1000
        cls.max_len = 30
        # GPUGrammar init takes (num_variables, use_globals)
        cls.grammar = GPUGrammar(num_variables=1)
        cls.ops = GPUOperators(cls.grammar, cls.device, cls.pop_size, cls.max_len)
        
        # Mock population
        cls.pop = cls.ops.generate_random_population(cls.pop_size)
        cls.fitness = torch.rand(cls.pop_size, device=cls.device)

    def test_depth_fair_crossover(self):
        """Verify Depth-Fair sampling doesn't crash and returns valid indices"""
        print("\n[Test] Depth-Fair Crossover")
        valid_mask = (self.pop != 0) # simplified mask
        
        # 1. Test helper directly
        try:
            indices = self.ops._depth_fair_sample(self.pop, valid_mask)
            self.assertEqual(indices.shape[0], self.pop_size)
            print("  - _depth_fair_sample: OK")
        except Exception as e:
            self.fail(f"_depth_fair_sample failed: {e}")

        # 2. Test integration
        GpuGlobals.DEPTH_FAIR_CROSSOVER = True
        try:
            offspring = self.ops.crossover_population(self.pop.clone(), 1.0)
            self.assertEqual(offspring.shape, self.pop.shape)
            print("  - crossover_population (Depth-Fair): OK")
        except Exception as e:
            self.fail(f"crossover_population failed: {e}")

    def test_library_learning(self):
        """Verify Library Learning update and sample"""
        print("\n[Test] Library Learning")
        lib = LibraryLearner(self.grammar, self.pop_size, self.max_len, self.device)
        
        # 1. Update
        try:
            lib.update(self.pop, self.fitness)
            print(f"  - update: OK (size={lib.size})")
        except Exception as e:
            self.fail(f"Library update failed: {e}")
            
        # 2. Sample
        try:
            blocks = lib.sample(k=10)
            if lib.size > 0:
                self.assertIsNotNone(blocks)
                self.assertEqual(blocks.shape[0], min(10, lib.size))
            print("  - sample: OK")
        except Exception as e:
            self.fail(f"Library sample failed: {e}")

    def test_alps_logic(self):
        """Verify ALPS age tensor operations (GPU)"""
        print("\n[Test] ALPS Logic")
        ages = torch.zeros(self.pop_size, dtype=torch.long, device=self.device)
        
        # 1. Increment
        ages.add_(1)
        self.assertTrue((ages == 1).all())
        
        # 2. Mock Reseed (Layer 0)
        # Sort by age (all equal now) -> take top 10
        reseed_idx = torch.topk(ages, 10, largest=False).indices
        ages[reseed_idx] = 0
        self.assertTrue((ages[reseed_idx] == 0).all())
        print("  - ALPS tensor ops: OK")

    def test_pareto_optimizer(self):
        """Verify Pareto Sort (NSGA-II) runs on GPU"""
        print("\n[Test] Pareto Optimizer")
        pareto = ParetoOptimizer(self.device)
        
        # Mock objectives: minimize f1, maximize f2 (minimize -f2)
        f1 = torch.rand(100, device=self.device)
        f2 = torch.rand(100, device=self.device)
        
        try:
            ranks, crowding = pareto.compute_ranks_and_crowding(f1, f2)
            self.assertEqual(ranks.shape[0], 100)
            self.assertEqual(crowding.shape[0], 100)
            print("  - compute_ranks_and_crowding: OK")
        except Exception as e:
            self.fail(f"Pareto sort failed: {e}")

if __name__ == '__main__':
    unittest.main()
