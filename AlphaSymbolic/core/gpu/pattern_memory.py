"""
Pattern Memory System for GPU GP Engine.

Stores successful subtrees/patterns and injects them into the population
to accelerate convergence by reusing proven building blocks.
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class PatternMemory:
    """
    Memory system that stores successful formula patterns (subtrees).
    
    Patterns with good fitness scores are recorded and can be injected
    into the population to share successful building blocks.
    """
    
    def __init__(self, device: torch.device, max_patterns: int = 100, 
                 fitness_threshold: float = 10.0, min_uses: int = 3):
        """
        Args:
            device: Torch device
            max_patterns: Maximum number of patterns to store
            fitness_threshold: Only record patterns from individuals with fitness below this
            min_uses: Minimum uses before a pattern is considered "useful"
        """
        self.device = device
        self.max_patterns = max_patterns
        self.fitness_threshold = fitness_threshold
        self.min_uses = min_uses
        
        # Pattern storage: hash -> (pattern_rpn, count, best_fitness)
        self.patterns: Dict[tuple, Tuple[List[int], int, float]] = {}
        
        # Usage stats
        self.total_recorded = 0
        self.total_injected = 0
    
    def record_subtrees(self, population: torch.Tensor, fitness: torch.Tensor, 
                        grammar, min_size: int = 3, max_size: int = 10):
        """
        Extract and record successful subtrees from the population.
        
        Args:
            population: [PopSize, L] RPN tensors
            fitness: [PopSize] RMSE values
            grammar: GPUGrammar for subtree extraction
            min_size: Minimum subtree size to record
            max_size: Maximum subtree size to record
        """
        pop_cpu = population.cpu().numpy()
        fit_cpu = fitness.cpu().numpy()
        
        # Only look at individuals with good fitness
        good_mask = fit_cpu < self.fitness_threshold
        good_indices = np.where(good_mask)[0]
        
        for idx in good_indices[:50]:  # Limit to prevent slowdown
            rpn = pop_cpu[idx]
            fit = fit_cpu[idx]
            
            # Find all subtrees
            subtrees = self._extract_subtrees(rpn, grammar, min_size, max_size)
            
            for subtree in subtrees:
                self._record_pattern(subtree, fit)
    
    def _extract_subtrees(self, rpn: np.ndarray, grammar, min_size: int, max_size: int) -> List[List[int]]:
        """
        Extract all valid subtrees from an RPN expression.
        """
        subtrees = []
        
        # Find non-pad length
        non_pad = rpn[rpn != 0]
        if len(non_pad) < min_size:
            return subtrees
        
        # Try each position as potential subtree root
        for root_idx in range(len(non_pad)):
            span = grammar.get_subtree_span(non_pad.tolist(), root_idx)
            if span[0] == -1:
                continue
            
            start, end = span
            size = end - start + 1
            
            if min_size <= size <= max_size:
                subtree = non_pad[start:end+1].tolist()
                subtrees.append(subtree)
        
        return subtrees
    
    def _record_pattern(self, pattern: List[int], fitness: float):
        """
        Record a pattern in memory.
        """
        key = tuple(pattern)
        
        if key in self.patterns:
            rpn, count, best_fit = self.patterns[key]
            self.patterns[key] = (rpn, count + 1, min(best_fit, fitness))
        else:
            if len(self.patterns) >= self.max_patterns:
                # Evict least used pattern
                self._evict_least_useful()
            
            self.patterns[key] = (pattern, 1, fitness)
            self.total_recorded += 1
    
    def _evict_least_useful(self):
        """
        Remove the least useful pattern (lowest count, highest fitness).
        """
        if not self.patterns:
            return
        
        # Score: higher is worse (low count, high fitness)
        def score(item):
            key, (rpn, count, best_fit) = item
            return -count + best_fit / 100.0
        
        worst_key = max(self.patterns.items(), key=score)[0]
        del self.patterns[worst_key]
    
    def get_useful_patterns(self, n: int = 10) -> List[List[int]]:
        """
        Get the top N most useful patterns.
        
        Args:
            n: Number of patterns to return
            
        Returns:
            List of RPN patterns (as lists of token IDs)
        """
        # Filter by min_uses
        useful = [(k, v) for k, v in self.patterns.items() if v[1] >= self.min_uses]
        
        # Sort by usefulness: high count, low fitness
        useful.sort(key=lambda x: (-x[1][1], x[1][2]))
        
        return [list(k) for k, v in useful[:n]]
    
    def inject_into_population(self, population: torch.Tensor, constants: torch.Tensor,
                                grammar, percent: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Inject useful patterns into the population by replacing some individuals.
        
        Args:
            population: [PopSize, L] RPN tensors
            constants: [PopSize, MaxC] constants
            grammar: GPUGrammar for token lookup
            percent: Fraction of population to replace
            
        Returns:
            (new_population, new_constants, n_injected)
        """
        patterns = self.get_useful_patterns(20)
        if not patterns:
            return population, constants, 0
        
        pop_size, max_len = population.shape
        n_inject = max(1, int(pop_size * percent))
        n_inject = min(n_inject, len(patterns) * 2)  # Don't inject more than we have variety
        
        pop_out = population.clone()
        const_out = constants.clone()
        
        # Inject at random positions (avoid elites at front)
        inject_start = int(pop_size * 0.1)  # Skip first 10% (elites)
        inject_positions = torch.randint(inject_start, pop_size, (n_inject,))
        
        for i, pos in enumerate(inject_positions):
            pattern = patterns[i % len(patterns)]
            
            # Pad pattern to max_len
            padded = pattern + [0] * (max_len - len(pattern))
            padded = padded[:max_len]
            
            pop_out[pos] = torch.tensor(padded, device=self.device, dtype=population.dtype)
            
            # Random constants for the pattern
            const_out[pos] = torch.randn_like(const_out[pos]) * 0.5
        
        self.total_injected += n_inject
        return pop_out, const_out, n_inject
    
    def get_stats(self) -> Dict:
        """
        Get pattern memory statistics.
        """
        return {
            'n_patterns': len(self.patterns),
            'total_recorded': self.total_recorded,
            'total_injected': self.total_injected,
            'useful_count': sum(1 for v in self.patterns.values() if v[1] >= self.min_uses)
        }
