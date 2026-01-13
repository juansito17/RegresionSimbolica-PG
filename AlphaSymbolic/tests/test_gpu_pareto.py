"""
Test Pareto optimization in GPU engine.
"""
import unittest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.gpu.pareto import ParetoOptimizer
from core.gpu import TensorGeneticEngine


class TestParetoOptimizer(unittest.TestCase):
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = ParetoOptimizer(self.device, max_front_size=50)
    
    def test_dominates(self):
        """Test domination check."""
        print("\nTesting domination logic...")
        
        # A dominates B if A is better in all objectives
        self.assertTrue(self.optimizer.dominates((1.0, 5), (2.0, 10)))  # A better in both
        self.assertTrue(self.optimizer.dominates((1.0, 5), (1.0, 10)))  # A same fit, better complexity
        self.assertTrue(self.optimizer.dominates((1.0, 5), (2.0, 5)))   # A better fit, same complexity
        
        self.assertFalse(self.optimizer.dominates((2.0, 5), (1.0, 10)))  # Trade-off, no dominance
        self.assertFalse(self.optimizer.dominates((1.0, 5), (1.0, 5)))   # Same, no dominance
        
        print("Dominance logic correct")
    
    def test_non_dominated_sort(self):
        """Test non-dominated sorting."""
        print("\nTesting non-dominated sorting...")
        
        # Create test case with known Pareto fronts
        # Front 1: (1, 10), (5, 5), (10, 1) - non-dominated
        # Front 2: (2, 11), (6, 6) - dominated by front 1
        
        fitness = torch.tensor([1.0, 5.0, 10.0, 2.0, 6.0], device=self.device)
        complexity = torch.tensor([10.0, 5.0, 1.0, 11.0, 6.0], device=self.device)
        
        fronts = self.optimizer.non_dominated_sort(fitness, complexity)
        
        print(f"Fronts: {fronts}")
        
        # First front should contain indices 0, 1, 2
        self.assertEqual(len(fronts), 2, "Should have 2 fronts")
        self.assertEqual(set(fronts[0]), {0, 1, 2}, "First front should be {0, 1, 2}")
        
        print("Non-dominated sorting correct")
    
    def test_crowding_distance(self):
        """Test crowding distance calculation."""
        print("\nTesting crowding distance...")
        
        fitness = torch.tensor([1.0, 5.0, 10.0], device=self.device)
        complexity = torch.tensor([10.0, 5.0, 1.0], device=self.device)
        
        front = [0, 1, 2]
        distances = self.optimizer.crowding_distance(front, fitness, complexity)
        
        print(f"Crowding distances: {distances}")
        
        # Boundary points should have infinite distance
        self.assertEqual(distances[0].item(), float('inf'))
        self.assertEqual(distances[2].item(), float('inf'))
        
        # Interior point should have finite distance
        self.assertTrue(distances[1].item() < float('inf'))
        self.assertTrue(distances[1].item() > 0)
        
        print("Crowding distance correct")
    
    def test_select(self):
        """Test selection using NSGA-II."""
        print("\nTesting NSGA-II selection...")
        
        engine = TensorGeneticEngine(device=self.device, pop_size=100, n_islands=1)
        population = engine.initialize_population()
        
        fitness = torch.rand(100, device=self.device)
        complexity = (population != 0).sum(dim=1).float()
        
        # Select 20 individuals
        selected = self.optimizer.select(population, fitness, complexity, 20)
        
        print(f"Selected {len(selected)} individuals")
        
        self.assertEqual(len(selected), 20)
        
        # All indices should be valid
        self.assertTrue((selected >= 0).all())
        self.assertTrue((selected < 100).all())
        
        print("NSGA-II selection works")
    
    def test_pareto_front(self):
        """Test getting the Pareto front."""
        print("\nTesting Pareto front extraction...")
        
        fitness = torch.tensor([1.0, 5.0, 10.0, 2.0, 6.0], device=self.device)
        complexity = torch.tensor([10.0, 5.0, 1.0, 11.0, 6.0], device=self.device)
        
        front = self.optimizer.get_pareto_front(fitness, complexity)
        
        print(f"Pareto front: {front}")
        
        self.assertEqual(set(front), {0, 1, 2})
        
        print("Pareto front extraction correct")


if __name__ == '__main__':
    unittest.main(verbosity=2)
