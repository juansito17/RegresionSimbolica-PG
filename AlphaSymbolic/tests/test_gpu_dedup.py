"""
Test population deduplication in GPU engine.
"""
import unittest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.gpu import TensorGeneticEngine
from core.gpu.config import GpuGlobals
from core.gpu.engine import PAD_ID


class TestDeduplication(unittest.TestCase):
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.engine = TensorGeneticEngine(device=self.device, pop_size=100, n_islands=1, num_variables=1)
    
    def test_detects_duplicates(self):
        """Test that duplicate detection works."""
        print("\nTesting duplicate detection...")
        
        # Create population with known duplicates
        pop = self.engine.initialize_population()
        constants = torch.randn(100, self.engine.max_constants, device=self.device, dtype=torch.float64)
        
        # Make first 10 individuals duplicates of index 0
        for i in range(1, 10):
            pop[i] = pop[0].clone()
        
        # Run deduplication
        new_pop, new_consts, n_dups = self.engine.deduplicate_population(pop, constants)
        
        print(f"Detected and replaced {n_dups} duplicates")
        
        # Should have detected at least 9 duplicates (indices 1-9 are copies of 0)
        # Note: initial population may have pre-existing duplicates from tiling
        self.assertGreaterEqual(n_dups, 9, f"Expected at least 9 duplicates, found {n_dups}")
        
    def test_replaces_with_different(self):
        """Test that duplicates are replaced with different individuals."""
        print("\nTesting duplicate replacement...")
        
        pop = self.engine.initialize_population()
        constants = torch.randn(100, self.engine.max_constants, device=self.device, dtype=torch.float64)
        
        # Create duplicates
        original = pop[0].clone()
        for i in range(1, 5):
            pop[i] = original.clone()
        
        new_pop, _, n_dups = self.engine.deduplicate_population(pop, constants)
        
        # Check that replaced individuals are different from original
        different_count = 0
        for i in range(1, 5):
            if not (new_pop[i] == original).all().item():
                different_count += 1
        
        print(f"{different_count}/{4} duplicates were replaced with different individuals")
        self.assertGreater(different_count, 0, "At least some duplicates should be replaced with different individuals")
        
    def test_no_duplicates_unchanged(self):
        """Test that population without duplicates is unchanged."""
        print("\nTesting no-duplicate case...")
        
        # Use very small population to manually create unique individuals
        engine = TensorGeneticEngine(device=self.device, pop_size=10, n_islands=1, num_variables=1)
        
        # Create unique individuals
        formulas = ["x0", "(x0 + 1)", "(x0 * 2)", "(x0 + 2)", "(x0 * 3)", 
                    "(x0 + 3)", "(x0 * 1)", "(x0 - 1)", "(x0 - 2)", "(x0 / 2)"]
        pop = engine.infix_to_rpn(formulas)
        constants = torch.randn(10, engine.max_constants, device=self.device, dtype=torch.float64)
        
        new_pop, _, n_dups = engine.deduplicate_population(pop, constants)
        
        print(f"No-duplicate population: {n_dups} duplicates found")
        self.assertEqual(n_dups, 0, "No duplicates should be found")
        
    def test_config_flag_respected(self):
        """Test that PREVENT_DUPLICATES=False disables deduplication."""
        print("\nTesting config flag...")
        
        # Temporarily disable
        original_value = GpuGlobals.PREVENT_DUPLICATES
        GpuGlobals.PREVENT_DUPLICATES = False
        
        pop = self.engine.initialize_population()
        constants = torch.randn(100, self.engine.max_constants, device=self.device, dtype=torch.float64)
        
        # Create duplicates
        for i in range(1, 10):
            pop[i] = pop[0].clone()
        
        new_pop, _, n_dups = self.engine.deduplicate_population(pop, constants)
        
        # Restore
        GpuGlobals.PREVENT_DUPLICATES = original_value
        
        self.assertEqual(n_dups, 0, "Deduplication should be disabled when flag is False")
        print("Config flag correctly disables deduplication")


if __name__ == '__main__':
    unittest.main(verbosity=2)
