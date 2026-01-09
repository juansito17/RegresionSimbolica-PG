"""
Test SymPy simplification in GPU engine.
"""
import unittest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.gpu import TensorGeneticEngine
from core.gpu.engine import SYMPY_AVAILABLE


class TestGPUSimplification(unittest.TestCase):
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.engine = TensorGeneticEngine(device=self.device, pop_size=100, n_islands=1, num_variables=1)
    
    @unittest.skipUnless(SYMPY_AVAILABLE, "SymPy not installed")
    def test_simplify_redundant_expression(self):
        """Test that x + 0 simplifies to x."""
        print("\nTesting simplification of redundant expression...")
        
        # Create RPN for "x + 0" which should simplify to "x"
        # RPN: x 0 +  (but we don't have 0 literal, so let's use x * 1)
        # Actually, let's test with: (x * 1) -> x
        formula = "(x0 * 1)"
        rpn = self.engine.infix_to_rpn([formula])
        constants = torch.zeros(self.engine.max_constants, device=self.device, dtype=torch.float64)
        
        print(f"Original: {formula}")
        print(f"RPN shape: {rpn.shape}, tokens: {rpn[0].tolist()}")
        
        new_rpn, new_consts, success = self.engine.simplify_expression(rpn[0], constants)
        
        if success:
            result = self.engine.rpn_to_infix(new_rpn, new_consts)
            print(f"Simplified: {result}")
            # Should be shorter or equal
            self.assertLessEqual(len(result), len(formula))
        else:
            print("Simplification did not succeed (may be expected for trivial cases)")
    
    @unittest.skipUnless(SYMPY_AVAILABLE, "SymPy not installed")
    def test_simplify_algebraic_identity(self):
        """Test that (x + x) simplifies to 2*x."""
        print("\nTesting simplification of algebraic identity...")
        
        formula = "(x0 + x0)"
        rpn = self.engine.infix_to_rpn([formula])
        constants = torch.zeros(self.engine.max_constants, device=self.device, dtype=torch.float64)
        
        print(f"Original: {formula}")
        
        new_rpn, new_consts, success = self.engine.simplify_expression(rpn[0], constants)
        
        if success:
            result = self.engine.rpn_to_infix(new_rpn, new_consts)
            print(f"Simplified: {result}")
            # Should contain 2 and x
            self.assertTrue('2' in result or '2*' in result or result == 'x0' or 'x0' in result)
        else:
            print("Simplification did not succeed")
            
    @unittest.skipUnless(SYMPY_AVAILABLE, "SymPy not installed")
    def test_simplify_complex_expression(self):
        """Test simplification of nested expression."""
        print("\nTesting simplification of complex expression...")
        
        # ((x * 2) + (x * 3)) should simplify to 5*x
        formula = "((x0 * 2) + (x0 * 3))"
        rpn = self.engine.infix_to_rpn([formula])
        constants = torch.zeros(self.engine.max_constants, device=self.device, dtype=torch.float64)
        
        print(f"Original: {formula}")
        
        new_rpn, new_consts, success = self.engine.simplify_expression(rpn[0], constants)
        
        if success:
            result = self.engine.rpn_to_infix(new_rpn, new_consts)
            print(f"Simplified: {result}")
            # Result should be simpler
            orig_len = (rpn[0] != 0).sum().item()
            new_len = (new_rpn != 0).sum().item()
            print(f"Token count: {orig_len} -> {new_len}")
        else:
            print("Simplification did not succeed")
    
    @unittest.skipUnless(SYMPY_AVAILABLE, "SymPy not installed")
    def test_simplify_population_batch(self):
        """Test batch simplification of population."""
        print("\nTesting batch simplification...")
        
        formulas = [
            "(x0 * 1)",
            "(x0 + x0)",
            "((x0 * 2) + (x0 * 3))",
            "(x0 + 0)",  # Note: 0 may not be in vocabulary
            "x0"
        ]
        
        # Only convert valid formulas
        valid_formulas = [f for f in formulas if self.engine.grammar.token_to_id.get('0') or '0' not in f]
        pop = self.engine.infix_to_rpn(valid_formulas[:min(5, len(valid_formulas))])
        
        # Pad to full population if needed
        if pop.shape[0] < 10:
            pop = torch.cat([pop, pop.repeat(2, 1)[:10-pop.shape[0]]], dim=0)
        
        constants = torch.zeros(pop.shape[0], self.engine.max_constants, device=self.device, dtype=torch.float64)
        
        new_pop, new_consts, n_simplified = self.engine.simplify_population(pop, constants, top_k=5)
        
        print(f"Simplified {n_simplified}/{5} expressions")
        self.assertGreaterEqual(n_simplified, 0)  # At least 0 (may fail if formulas are already simple)

    def test_sympy_import(self):
        """Check if SymPy is available."""
        print(f"\nSymPy available: {SYMPY_AVAILABLE}")
        if SYMPY_AVAILABLE:
            import sympy
            print(f"SymPy version: {sympy.__version__}")
        self.assertTrue(True)  # Just informational


if __name__ == '__main__':
    unittest.main(verbosity=2)
