"""
Pareto Optimization (NSGA-II) for GPU GP Engine.

Implements multi-objective optimization balancing:
- Objective 1: Error (RMSE) - minimize
- Objective 2: Complexity (tree size) - minimize
"""
import torch
import numpy as np
from typing import List, Tuple


class ParetoOptimizer:
    """
    NSGA-II style Pareto optimizer for symbolic regression.
    
    Objectives:
        - fitness: RMSE (lower is better)
        - complexity: number of tokens (lower is better)
    """
    
    def __init__(self, device: torch.device, max_front_size: int = 50, dtype=None):
        self.device = device
        self.max_front_size = max_front_size
        self.dtype = dtype if dtype is not None else torch.float64
    
    def dominates(self, obj_a: Tuple[float, float], obj_b: Tuple[float, float]) -> bool:
        """
        Check if solution A dominates solution B (both objectives <= and at least one <).
        """
        a_fit, a_comp = obj_a
        b_fit, b_comp = obj_b
        
        # A dominates B if A is <= B in all objectives and < in at least one
        at_least_one_better = (a_fit < b_fit) or (a_comp < b_comp)
        not_worse = (a_fit <= b_fit) and (a_comp <= b_comp)
        
        return not_worse and at_least_one_better
    
    
    def non_dominated_sort(self, fitness: torch.Tensor, complexity: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Memory-efficient GPU Non-Dominated Sort (block-based).
        Avoids O(N^2) memory by computing domination on the fly.
        """
        n = fitness.shape[0]
        device = self.device
        
        # 1. Compute Domination Counts (how many dominate me)
        # We do this in blocks to avoid allocating [N, N] matrix
        domination_counts = torch.zeros(n, dtype=torch.long, device=device)
        
        # Block size for N^2 comparisons
        # RTX 3050 has 4GB. Safe matrix size ~5000x5000 bools (25MB).
        # We can do [Block, N]. Block=5000.
        block_size = 2048
        
        # Pre-broadcast tensors
        # We need to compare every i with every j
        # Inner loop compares a Block of 'i' against ALL 'j'
        
        fit_flat = fitness
        comp_flat = complexity
        
        for i_start in range(0, n, block_size):
            i_end = min(i_start + block_size, n)
            i_idx = torch.arange(i_start, i_end, device=device)
            bn = i_end - i_start
            
            # [Block, 1]
            b_fit = fit_flat[i_idx].unsqueeze(1)
            b_comp = comp_flat[i_idx].unsqueeze(1)
            
            # Compare against ALL [1, N]
            # [1, N]
            all_fit = fit_flat.unsqueeze(0)
            all_comp = comp_flat.unsqueeze(0)
            
            # Domination logic: J dominates I?
            # J_fit <= I_fit AND J_comp <= I_comp AND (J_fit < I_fit OR J_comp < I_comp)
            # Here 'I' is the block (rows). 'J' is all (cols).
            # We want to sum over J (cols) for each I.
            
            # j_not_worse_i = (all_fit <= b_fit) & (all_comp <= b_comp)
            # j_better_i = (all_fit < b_fit) | (all_comp < b_comp)
            # j_dominates_i = j_not_worse_i & j_better_i [Block, N]
            
            # Chunking the 'j' (cols) loop also if N is huge?
            # 2048 * 100,000 = 200M bools = 200MB. This fits easily.
            
            worse_or_equal = (all_fit <= b_fit) & (all_comp <= b_comp)
            strict_better = (all_fit < b_fit) | (all_comp < b_comp)
            dominated_by = (worse_or_equal & strict_better) # [Block, N]
            
            # Sum rows -> count for each i in block
            domination_counts[i_idx] += dominated_by.sum(dim=1).long()
            
        # 2. Peel Fronts
        ranks = torch.zeros(n, dtype=torch.long, device=device)
        active_mask = torch.ones(n, dtype=torch.bool, device=device)
        fronts = []
        current_rank = 0
        
        while active_mask.any():
            # Identify current front
            # Those active with count 0
            current_front_mask = (domination_counts == 0) & active_mask
            
            if not current_front_mask.any():
                break # Should ensure progress
                
            front_indices = torch.nonzero(current_front_mask).squeeze(1)
            fronts.append(front_indices)
            ranks[current_front_mask] = current_rank
            
            # Remove from active
            active_mask[current_front_mask] = False
            
            # Decrement counts of those dominated by THIS front
            # We must compute: For remaining J, does Front I dominate J?
            # Sub-problem: [FrontSize, RemainingSize]
            
            # Remaining indices
            remaining_mask = active_mask # Boolean [N]
            if not remaining_mask.any():
                break
                
            # To apply updates efficiently:
            # We iterate the FRONT in blocks (since front can be large)
            # And compare against ALL active (or just all, simpler indexing)
            # And subtract from domination_counts
            
            f_size = front_indices.shape[0]
            
            for f_start in range(0, f_size, block_size):
                f_end = min(f_start + block_size, f_size)
                f_batch = front_indices[f_start:f_end] # [Batch]
                
                # [Batch, 1]
                b_fit = fit_flat[f_batch].unsqueeze(1)
                b_comp = comp_flat[f_batch].unsqueeze(1)
                
                # Compare against ALL (masking later is faster than fancy indexing?)
                # [1, N]
                all_fit = fit_flat.unsqueeze(0)
                all_comp = comp_flat.unsqueeze(0)
                
                # I (Front) dominates J (All)?
                # I_fit <= J_fit ...
                
                i_dominates_j = (b_fit <= all_fit) & (b_comp <= all_comp) & \
                                ((b_fit < all_fit) | (b_comp < all_comp)) # [Batch, N]
                                
                # Sum columns -> how many in this batch dominate each J
                decrement_vec = i_dominates_j.sum(dim=0) # [N]
                
                domination_counts -= decrement_vec.long()
                
            current_rank += 1
            
        return fronts, ranks

    def compute_ranks_and_crowding(self, fitness: torch.Tensor, complexity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Rank (0-based) and Crowding Distance for the entire population.
        Used for Tournament Selection.
        High Crowding Distance is good (for same rank).
        """
        fronts, ranks = self.non_dominated_sort(fitness, complexity)
        n = fitness.shape[0]
        crowding = torch.zeros(n, dtype=self.dtype, device=self.device)
        
        for front in fronts:
             # Compute crowding for each front
             # If front size < 2, inf
             k = front.shape[0]
             if k < 2:
                 crowding[front] = float('inf')
                 continue
                 
             # Gather values
             f_vals = fitness[front]
             c_vals = complexity[front]
             
             # Sub-crowding
             dists = torch.zeros(k, dtype=self.dtype, device=self.device)
             
             # Objective 1: Fitness
             sorted_idx = torch.argsort(f_vals)
             dists[sorted_idx[0]] = float('inf')
             dists[sorted_idx[-1]] = float('inf')
             r = f_vals[sorted_idx[-1]] - f_vals[sorted_idx[0]]
             if r > 1e-9:
                 dists[sorted_idx[1:-1]] += (f_vals[sorted_idx[2:]] - f_vals[sorted_idx[:-2]]) / r
                 
             # Objective 2: Complexity
             sorted_idx = torch.argsort(c_vals)
             dists[sorted_idx[0]] = float('inf')
             dists[sorted_idx[-1]] = float('inf')
             r = c_vals[sorted_idx[-1]] - c_vals[sorted_idx[0]]
             if r > 1e-9:
                 dists[sorted_idx[1:-1]] += (c_vals[sorted_idx[2:]] - c_vals[sorted_idx[:-2]]) / r
                 
             crowding[front] = dists
             
        return ranks, crowding
    
    def select(self, population: torch.Tensor, fitness: torch.Tensor, complexity: torch.Tensor, n_select: int) -> torch.Tensor:
        """
        Select n_select individuals using NSGA-II selection (Rank + Crowding).
        """
        fronts, ranks = self.non_dominated_sort(fitness, complexity)
        # We need crowding for filtering the last front
        # But we can use compute_ranks_and_crowding if we want full metrics
        # For selection we just iterate fronts
        
        selected_indices = []
        count = 0
        
        for front in fronts:
            k = front.shape[0]
            if count + k <= n_select:
                selected_indices.append(front)
                count += k
            else:
                # Last front, sort by crowding
                rem = n_select - count
                
                # Compute crowding ONLY for this front (faster)
                f_vals = fitness[front]
                c_vals = complexity[front]
                
                dists = torch.zeros(k, dtype=self.dtype, device=self.device)
                
                # Fit
                idx = torch.argsort(f_vals)
                dists[idx[0]] = float('inf'); dists[idx[-1]] = float('inf')
                r = f_vals[idx[-1]] - f_vals[idx[0]]
                if r > 1e-9: dists[idx[1:-1]] += (f_vals[idx[2:]] - f_vals[idx[:-2]]) / r
                
                # Comp
                idx = torch.argsort(c_vals)
                dists[idx[0]] = float('inf'); dists[idx[-1]] = float('inf')
                r = c_vals[idx[-1]] - c_vals[idx[0]]
                if r > 1e-9: dists[idx[1:-1]] += (c_vals[idx[2:]] - c_vals[idx[:-2]]) / r
                
                _, best_local = torch.topk(dists, rem, largest=True) # Max crowding
                selected_indices.append(front[best_local])
                count += rem
                break
                
        return torch.cat(selected_indices)

    def get_pareto_front(self, fitness: torch.Tensor, complexity: torch.Tensor) -> List[int]:
        fronts, _ = self.non_dominated_sort(fitness, complexity)
        if not fronts: return []
        front = fronts[0]
        
        if front.shape[0] > self.max_front_size:
            # Prune by crowding
             f_vals = fitness[front]
             c_vals = complexity[front]
             k = front.shape[0]
             dists = torch.zeros(k, dtype=self.dtype, device=self.device)
             
             idx = torch.argsort(f_vals)
             dists[idx[0]] = float('inf'); dists[idx[-1]] = float('inf')
             if (f_vals[idx[-1]]-f_vals[idx[0]]) > 1e-9: dists[idx[1:-1]] += (f_vals[idx[2:]]-f_vals[idx[:-2]])/(f_vals[idx[-1]]-f_vals[idx[0]])

             idx = torch.argsort(c_vals)
             dists[idx[0]] = float('inf'); dists[idx[-1]] = float('inf') 
             if (c_vals[idx[-1]]-c_vals[idx[0]]) > 1e-9: dists[idx[1:-1]] += (c_vals[idx[2:]]-c_vals[idx[:-2]])/(c_vals[idx[-1]]-c_vals[idx[0]])
             
             _, top_idx = torch.topk(dists, self.max_front_size)
             front = front[top_idx]
             
        return front.tolist()

