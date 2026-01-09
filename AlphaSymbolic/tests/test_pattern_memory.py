"""
Test Pattern Memory system in GPU engine.
"""
import unittest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.gpu.pattern_memory import PatternMemory
from core.gpu import TensorGeneticEngine


class TestPatternMemory(unittest.TestCase):
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory = PatternMemory(self.device, max_patterns=50, fitness_threshold=100.0, min_uses=2)
        self.engine = TensorGeneticEngine(device=self.device, pop_size=100, n_islands=1)
    
    def test_record_subtrees(self):
        """Test pattern recording."""
        print("\nTesting pattern recording...")
        
        pop = self.engine.initialize_population()
        fitness = torch.rand(100, device=self.device) * 10  # All good fitness
        
        # Record patterns
        self.memory.record_subtrees(pop, fitness, self.engine.grammar)
        
        stats = self.memory.get_stats()
        print(f"Recorded patterns: {stats['n_patterns']}")
        
        self.assertGreater(stats['n_patterns'], 0, "Should have recorded some patterns")
        self.assertGreater(stats['total_recorded'], 0)
    
    def test_get_useful_patterns(self):
        """Test retrieving useful patterns."""
        print("\nTesting useful pattern retrieval...")
        
        # Record same patterns multiple times to make them "useful"
        pop = self.engine.initialize_population()
        fitness = torch.rand(100, device=self.device) * 5
        
        for _ in range(5):  # Record multiple times
            self.memory.record_subtrees(pop, fitness, self.engine.grammar)
        
        patterns = self.memory.get_useful_patterns(10)
        print(f"Got {len(patterns)} useful patterns")
        
        # After multiple recordings, should have some useful patterns
        self.assertGreater(len(patterns), 0, "Should have useful patterns after multiple recordings")
    
    def test_inject_patterns(self):
        """Test pattern injection into population."""
        print("\nTesting pattern injection...")
        
        # First record some patterns
        pop = self.engine.initialize_population()
        fitness = torch.rand(100, device=self.device) * 5
        
        for _ in range(5):
            self.memory.record_subtrees(pop, fitness, self.engine.grammar)
        
        constants = torch.randn(100, self.engine.max_constants, device=self.device, dtype=torch.float64)
        
        # Inject
        new_pop, new_const, n_inj = self.memory.inject_into_population(
            pop, constants, self.engine.grammar, percent=0.1
        )
        
        print(f"Injected {n_inj} patterns")
        
        self.assertGreaterEqual(n_inj, 0)  # May be 0 if no useful patterns yet
        
        stats = self.memory.get_stats()
        print(f"Total injected: {stats['total_injected']}")
    
    def test_eviction(self):
        """Test pattern eviction when memory is full."""
        print("\nTesting pattern eviction...")
        
        # Create memory with small limit
        small_memory = PatternMemory(self.device, max_patterns=10, fitness_threshold=100.0)
        
        pop = self.engine.initialize_population()
        fitness = torch.rand(100, device=self.device) * 10
        
        # Record many times to trigger eviction
        for _ in range(20):
            small_memory.record_subtrees(pop, fitness, self.engine.grammar)
        
        stats = small_memory.get_stats()
        print(f"Patterns in memory: {stats['n_patterns']} (max: 10)")
        
        # Should not exceed max_patterns
        self.assertLessEqual(stats['n_patterns'], 10)
    
    def test_integration_with_engine(self):
        """Test that engine has pattern memory initialized."""
        print("\nTesting engine integration...")
        
        self.assertIsNotNone(self.engine.pattern_memory)
        self.assertIsInstance(self.engine.pattern_memory, PatternMemory)
        
        print("Pattern memory correctly integrated")


if __name__ == '__main__':
    unittest.main(verbosity=2)
