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
    
    def __init__(self, device: torch.device, operators, max_patterns: int = 100, 
                 fitness_threshold: float = 10.0, min_uses: int = 3):
        """
        Args:
            device: Torch device
            max_patterns: Maximum number of patterns to store
            fitness_threshold: Only record patterns from individuals with fitness below this
            min_uses: Minimum uses before a pattern is considered "useful"
        """
        self.device = device
        self.operators = operators
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
        Populate pattern memory on GPU.
        Uses operators._get_subtree_ranges to identify valid structures.
        """
        # 1. Filter Good Candidates (GPU)
        good_mask = fitness < self.fitness_threshold
        if not good_mask.any(): return
        
        # Limit processing to a batch to avoid stalling GPU
        indices = torch.nonzero(good_mask).squeeze(1)
        if indices.numel() > 50:
             # Random sub-sample
             perm = torch.randperm(indices.numel(), device=self.device)
             indices = indices[perm[:50]]
             
        good_pop = population[indices] # [K, L]
        good_fit = fitness[indices]
        
        # 2. Get Subtree Ranges (GPU)
        # starts_mat: [K, L]. Value j at [i, k] means subtree ending at k starts at j.
        # -1 if invalid.
        starts_mat = self.operators._get_subtree_ranges(good_pop)
        
        # 3. Filter by Size (GPU)
        B, L = good_pop.shape
        grid = torch.arange(L, device=self.device).unsqueeze(0).expand(B, L)
        
        # Size = End - Start + 1
        # End is 'grid' (column index)
        sizes = grid - starts_mat + 1
        
        valid_size_mask = (starts_mat != -1) & (sizes >= min_size) & (sizes <= max_size)
        
        if not valid_size_mask.any(): return
        
        # 4. Extract and Record (CPU loop only for insertion into Dict, but data stays on Tensor until list conversion)
        #   Ideally we would stay on GPU but Dictionary requires hashing which is easier on CPU for sparse patterns.
        #   However, we extracted 'indices' of valid subtrees. 
        #   Let's process the valid ones.
        
        # Get coordinates of valid subtrees
        coords = torch.nonzero(valid_size_mask) # [N_patterns, 2] (row, col_end)
        
        # Limit total patterns to extract per step
        if coords.shape[0] > 100:
             coords = coords[:100]
             
        # Pull to CPU only the necessary loose/small lists for storage
        # This is strictly cheaper than moving whole Population
        
        for k in range(coords.shape[0]):
             row_idx = coords[k, 0].item()
             end_idx = coords[k, 1].item()
             
             start_idx = starts_mat[row_idx, end_idx].item()
             
             # Extract pattern
             # We need to act on the specific individual 'good_pop[row_idx]'
             pattern_tensor = good_pop[row_idx, start_idx : end_idx+1]
             
             # Convert to list for hashing
             pattern_list = pattern_tensor.tolist()
             fit_val = good_fit[row_idx].item()
             
             self._record_pattern(pattern_list, fit_val)

    # _extract_subtrees REMOVED as it is replaced by GPU logic above.

    
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
