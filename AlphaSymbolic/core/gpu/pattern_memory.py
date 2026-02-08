"""
Pattern Memory System for GPU GP Engine - GPU-Native Version.

Stores successful subtrees/patterns using GPU tensor storage with hash-based
indexing. No CPU loops or .tolist() calls.
"""
import torch
from typing import Tuple


class PatternMemory:
    """
    GPU-Native memory system that stores successful formula patterns (subtrees).
    
    Uses hash-based tensor storage instead of Python dicts for maximum speed.
    """
    
    def __init__(self, device: torch.device, operators, max_patterns: int = 100, 
                 fitness_threshold: float = 10.0, min_uses: int = 3, dtype=None):
        """
        Args:
            device: Torch device
            operators: GPUOperators instance
            max_patterns: Maximum number of patterns to store
            min_uses: Minimum uses before a pattern is considered "useful"
            dtype: Floating point dtype for fitness (default float64)
        """
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float64
        self.operators = operators
        self.max_patterns = max_patterns
        self.fitness_threshold = fitness_threshold
        self.min_uses = min_uses
        
        # GPU-Native Pattern Storage
        # patterns_tensor: [max_patterns, max_pattern_len] - stored RPN patterns
        # patterns_hash: [max_patterns] - hash for each pattern
        # patterns_count: [max_patterns] - usage count
        # patterns_fitness: [max_patterns] - best fitness seen
        # patterns_len: [max_patterns] - actual length of each pattern
        
        self.max_pattern_len = 15  # Max subtree length to store
        self.patterns_tensor = torch.zeros(max_patterns, self.max_pattern_len, 
                                           dtype=torch.long, device=device)
        self.patterns_hash = torch.zeros(max_patterns, dtype=torch.long, device=device)
        self.patterns_count = torch.zeros(max_patterns, dtype=torch.long, device=device)
        self.patterns_fitness = torch.full((max_patterns,), float('inf'), 
                                           dtype=self.dtype, device=device)
        self.patterns_len = torch.zeros(max_patterns, dtype=torch.long, device=device)
        self.n_patterns = 0  # Current number of stored patterns
        
        # Hash weights for pattern hashing
        self.hash_weights = torch.randint(-2**30, 2**30, (self.max_pattern_len,), 
                                          dtype=torch.long, device=device)
        
        # Usage stats
        self.total_recorded = 0
        self.total_injected = 0
    
    def _compute_pattern_hash(self, patterns: torch.Tensor) -> torch.Tensor:
        """
        Compute hash for each pattern in batch. Pure GPU operation.
        
        Args:
            patterns: [N, L] tensor of pattern tokens
            
        Returns:
            [N] tensor of hashes
        """
        L = patterns.shape[1]
        weights = self.hash_weights[:L]
        return (patterns * weights).sum(dim=1)
    
    def record_subtrees(self, population: torch.Tensor, fitness: torch.Tensor, 
                        grammar, min_size: int = 3, max_size: int = 10):
        """
        Record successful subtrees from the population. GPU-Native implementation.
        No Python loops over individual patterns.
        """
        # 1. Filter Good Candidates (GPU)
        good_mask = fitness < self.fitness_threshold
        if not good_mask.any(): 
            return
        
        # Limit processing to a batch to avoid stalling GPU
        indices = torch.nonzero(good_mask).squeeze(1)
        if indices.numel() > 50:
            perm = torch.randperm(indices.numel(), device=self.device)
            indices = indices[perm[:50]]
        
        good_pop = population[indices]  # [K, L]
        good_fit = fitness[indices]
        
        # 2. Get Subtree Ranges (GPU) - [K, L] matrix of start indices
        starts_mat = self.operators._get_subtree_ranges(good_pop)
        
        # 3. Filter by Size (GPU)
        K, L = good_pop.shape
        grid = torch.arange(L, device=self.device).unsqueeze(0).expand(K, L)
        sizes = grid - starts_mat + 1
        
        # Valid subtrees: correct size range and valid start
        valid_mask = (starts_mat >= 0) & (sizes >= min_size) & (sizes <= max_size)
        valid_mask = valid_mask & (sizes <= self.max_pattern_len)
        
        if not valid_mask.any(): 
            return
        
        # 4. Extract patterns vectorized
        # Get coordinates of valid subtrees
        coords = torch.nonzero(valid_mask)  # [N_patterns, 2] (row, col_end)
        
        if coords.shape[0] == 0:
            return
            
        # Limit to avoid memory issues
        if coords.shape[0] > 100:
            perm = torch.randperm(coords.shape[0], device=self.device)
            coords = coords[perm[:100]]
        
        N_valid = coords.shape[0]
        
        # Extract pattern tensors using advanced indexing
        row_indices = coords[:, 0]
        end_indices = coords[:, 1]
        start_indices = starts_mat[row_indices, end_indices]
        pattern_lengths = end_indices - start_indices + 1
        
        # Create padded pattern tensor
        extracted_patterns = torch.zeros(N_valid, self.max_pattern_len, 
                                         dtype=torch.long, device=self.device)
        
        # Fully vectorized extraction using advanced indexing (no Python loop)
        # Build a [N_valid, max_pattern_len] index matrix
        offsets = torch.arange(self.max_pattern_len, device=self.device).unsqueeze(0)  # [1, MPL]
        src_indices = start_indices.unsqueeze(1) + offsets  # [N_valid, MPL]
        in_range_mask = (offsets < pattern_lengths.unsqueeze(1)) & (src_indices < L)  # [N_valid, MPL]
        src_indices_clamped = src_indices.clamp(0, L - 1)
        # Gather all values at once
        all_values = good_pop[row_indices.unsqueeze(1).expand_as(src_indices_clamped), src_indices_clamped]
        extracted_patterns = torch.where(in_range_mask, all_values, torch.zeros_like(all_values))
        
        # 5. Compute hashes for extracted patterns
        pattern_hashes = self._compute_pattern_hash(extracted_patterns)
        pattern_fitnesses = good_fit[row_indices]
        
        # 6. Update pattern storage (GPU operations)
        self._update_storage(extracted_patterns, pattern_hashes, pattern_fitnesses, pattern_lengths)
        
        self.total_recorded += N_valid
    
    def _update_storage(self, patterns: torch.Tensor, hashes: torch.Tensor, 
                        fitnesses: torch.Tensor, lengths: torch.Tensor):
        """
        Update the pattern storage with new patterns. FULLY VECTORIZED GPU-native.
        No Python loops - pure tensor operations.
        """
        N = patterns.shape[0]
        if N == 0:
            return
        
        # 1. Find which hashes already exist in storage (vectorized comparison)
        if self.n_patterns > 0:
            # Compare all new hashes against all existing hashes
            # existing_hashes: [n_patterns], new_hashes: [N]
            existing_hashes = self.patterns_hash[:self.n_patterns]  # [P]
            
            # Match matrix: [N, P] - True where new[i] matches existing[j]
            match_matrix = (hashes.unsqueeze(1) == existing_hashes.unsqueeze(0))  # [N, P]
            
            # For each new pattern, find if it has a match
            has_match = match_matrix.any(dim=1)  # [N]
            match_idx = match_matrix.long().argmax(dim=1)  # [N] - index of first match (or 0 if none)
            
            # Update counts for matching patterns (GPU scatter_add)
            if has_match.any():
                matching_indices = match_idx[has_match]  # Indices in existing storage
                ones = torch.ones(matching_indices.shape[0], dtype=torch.long, device=self.device)
                self.patterns_count.scatter_add_(0, matching_indices, ones)
                
                # Update fitness if new is better (GPU minimum)
                matching_fitnesses = fitnesses[has_match]
                current_fitnesses = self.patterns_fitness[matching_indices]
                better_mask = matching_fitnesses < current_fitnesses
                
                if better_mask.any():
                    update_indices = matching_indices[better_mask]
                    new_fit_vals = matching_fitnesses[better_mask]
                    self.patterns_fitness[update_indices] = new_fit_vals
            
            # Get new patterns that need to be added
            new_mask = ~has_match
        else:
            new_mask = torch.ones(N, dtype=torch.bool, device=self.device)
        
        # 2. Add new patterns to storage
        new_count = new_mask.sum().item()
        if new_count == 0:
            return
        
        new_patterns = patterns[new_mask]
        new_hashes = hashes[new_mask]
        new_fitnesses = fitnesses[new_mask]
        new_lengths = lengths[new_mask]
        
        # Determine how many we can add vs need to evict
        available_slots = self.max_patterns - self.n_patterns
        
        if new_count <= available_slots:
            # Simple case: just append
            start_idx = self.n_patterns
            end_idx = start_idx + new_count
            
            self.patterns_tensor[start_idx:end_idx] = new_patterns
            self.patterns_hash[start_idx:end_idx] = new_hashes
            self.patterns_count[start_idx:end_idx] = 1
            self.patterns_fitness[start_idx:end_idx] = new_fitnesses
            self.patterns_len[start_idx:end_idx] = new_lengths
            
            self.n_patterns = end_idx
        else:
            # Need to evict some patterns
            # Strategy: Keep highest-count, lowest-fitness patterns
            
            # Fill remaining slots first
            if available_slots > 0:
                start_idx = self.n_patterns
                end_idx = start_idx + available_slots
                
                self.patterns_tensor[start_idx:end_idx] = new_patterns[:available_slots]
                self.patterns_hash[start_idx:end_idx] = new_hashes[:available_slots]
                self.patterns_count[start_idx:end_idx] = 1
                self.patterns_fitness[start_idx:end_idx] = new_fitnesses[:available_slots]
                self.patterns_len[start_idx:end_idx] = new_lengths[:available_slots]
                
                self.n_patterns = self.max_patterns
                
                new_patterns = new_patterns[available_slots:]
                new_hashes = new_hashes[available_slots:]
                new_fitnesses = new_fitnesses[available_slots:]
                new_lengths = new_lengths[available_slots:]
            
            # Evict worst patterns for remaining new ones
            n_to_evict = new_patterns.shape[0]
            if n_to_evict > 0:
                # Score: higher is worse (more likely to evict)
                # Low count + high fitness = bad pattern
                evict_scores = -self.patterns_count.float() + self.patterns_fitness / 100.0
                
                # Get indices of worst patterns
                _, worst_indices = torch.topk(evict_scores[:self.n_patterns], 
                                              min(n_to_evict, self.n_patterns))
                
                n_replace = min(len(worst_indices), new_patterns.shape[0])
                replace_indices = worst_indices[:n_replace]
                
                self.patterns_tensor[replace_indices] = new_patterns[:n_replace]
                self.patterns_hash[replace_indices] = new_hashes[:n_replace]
                self.patterns_count[replace_indices] = 1
                self.patterns_fitness[replace_indices] = new_fitnesses[:n_replace]
                self.patterns_len[replace_indices] = new_lengths[:n_replace]
    
    def get_useful_patterns(self, n: int = 10) -> torch.Tensor:
        """
        Get the top N most useful patterns as a GPU tensor.
        
        Returns:
            [N, max_pattern_len] tensor of patterns
        """
        if self.n_patterns == 0:
            return torch.zeros(0, self.max_pattern_len, dtype=torch.long, device=self.device)
        
        # Filter by min_uses
        useful_mask = self.patterns_count[:self.n_patterns] >= self.min_uses
        
        if not useful_mask.any():
            return torch.zeros(0, self.max_pattern_len, dtype=torch.long, device=self.device)
        
        useful_indices = useful_mask.nonzero().squeeze(1)
        
        # Score: higher count, lower fitness = better
        scores = self.patterns_count[useful_indices].float() - self.patterns_fitness[useful_indices] / 1000.0
        
        # Get top N
        n_available = min(n, useful_indices.shape[0])
        _, top_local_idx = torch.topk(scores, n_available)
        top_indices = useful_indices[top_local_idx]
        
        return self.patterns_tensor[top_indices]
    
    def inject_into_population(self, population: torch.Tensor, constants: torch.Tensor,
                                grammar, percent: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Inject useful patterns into the population. GPU-native.
        LEGACY: Returns (pop, const, count). Use inject_into_population_inplace instead.
        """
        inject_positions = self.inject_into_population_inplace(population, constants, grammar, percent)
        n_injected = inject_positions.numel() if inject_positions is not None else 0
        return population, constants, n_injected
    
    def inject_into_population_inplace(self, population: torch.Tensor, constants: torch.Tensor,
                                        grammar, percent: float = 0.05):
        """
        Inject useful patterns into the population IN-PLACE. No clones.
        Returns: inject_positions tensor (or None if nothing injected).
        """
        patterns = self.get_useful_patterns(20)
        if patterns.shape[0] == 0:
            return None
        
        pop_size, max_len = population.shape
        n_inject = max(1, int(pop_size * percent))
        n_inject = min(n_inject, patterns.shape[0] * 2)
        
        # Inject at random positions (avoid elites at front)
        inject_start = int(pop_size * 0.1)
        inject_positions = torch.randint(inject_start, pop_size, (n_inject,), device=self.device)
        
        # Cycle through available patterns
        pattern_indices = torch.arange(n_inject, device=self.device) % patterns.shape[0]
        selected_patterns = patterns[pattern_indices]  # [n_inject, max_pattern_len]
        
        # Pad patterns to population max_len
        if selected_patterns.shape[1] < max_len:
            padding = torch.zeros(n_inject, max_len - selected_patterns.shape[1], 
                                  dtype=torch.long, device=self.device)
            selected_patterns = torch.cat([selected_patterns, padding], dim=1)
        else:
            selected_patterns = selected_patterns[:, :max_len]
        
        # Inject in-place (no clone)
        population[inject_positions] = selected_patterns
        constants[inject_positions] = torch.randn(n_inject, constants.shape[1],
                                                   device=self.device, dtype=constants.dtype) * 0.5
        
        self.total_injected += n_inject
        return inject_positions
    
    def get_stats(self) -> dict:
        """Get pattern memory statistics."""
        useful_count = (self.patterns_count[:self.n_patterns] >= self.min_uses).sum().item()
        return {
            'n_patterns': self.n_patterns,
            'total_recorded': self.total_recorded,
            'total_injected': self.total_injected,
            'useful_count': useful_count
        }
