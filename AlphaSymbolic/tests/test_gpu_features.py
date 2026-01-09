
import unittest
import torch
import numpy as np
import sys
import os

# Add root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.gpu_engine import TensorGeneticEngine

class TestGPUFeatures(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device('cuda')
        self.engine = TensorGeneticEngine(device=self.device, pop_size=1000, n_islands=1)

    def test_sniper_linear(self):
        """Test 'The Sniper' detects linear patterns instantly."""
        print("\nTesting Sniper (Linear)...")
        x = np.linspace(-5, 5, 20)
        y = 2.5 * x + 1.0
        
        # Should return instantly (< 0.1s)
        # We pass a tiny timeout to ensure it doesn't run evolution
        result = self.engine.run(x, y, seeds=[], timeout_sec=0.5)
        
        print(f"Sniper Result: {result}")
        # Expected: ((2.5000 * x) + 1.0000) or similar
        self.assertIsNotNone(result)
        self.assertIn("2.5000 * x", result)
        self.assertIn("1.0000", result)

    def test_sniper_geometric(self):
        """Test 'The Sniper' detects geometric patterns."""
        print("\nTesting Sniper (Geometric)...")
        x = np.linspace(0, 2, 20)
        # y = 3 * e^(2x) -> log(y) = log(3) + 2x = 1.0986 + 2x
        y = 3.0 * np.exp(2.0 * x)
        
        result = self.engine.run(x, y, seeds=[], timeout_sec=0.5)
        
        print(f"Sniper Result: {result}")
        # Expected: exp((2.0000 * x) + 1.0986)
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("exp("))
        self.assertIn("2.0000 * x", result)

    def test_dynamic_adaptation_no_crash(self):
        """Test that the engine runs with dynamic adaptation variables without crashing."""
        print("\nTesting Dynamic Adaptation (Smoke Test)...")
        x = np.linspace(-5, 5, 20)
        y = x**2 - 5 # Simple quadratic, needs evolution
        
        # Run for 2 seconds
        result = self.engine.run(x, y, seeds=[], timeout_sec=2.0)
        print(f"evolution Result: {result}")
        # Just ensure it didn't crash
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
