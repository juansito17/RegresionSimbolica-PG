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
    
    def __init__(self, device: torch.device, max_front_size: int = 50):
        self.device = device
        self.max_front_size = max_front_size
    
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
        Perform vectorized non-dominated sorting on the population on GPU.
        Returns:
            fronts: List[Tensor], where each tensor contains indices of a front.
            ranks: [N] tensor with rank of each individual (0 = front 1).
        """
        n = fitness.shape[0]
        
        # 1. Compute Domination Matrix [N, N]
        # A dominates B?
        # A_fit <= B_fit AND A_comp <= B_comp AND (A_fit < B_fit OR A_comp < B_comp)
        
        fit = fitness.unsqueeze(1) # [N, 1]
        comp = complexity.unsqueeze(1) # [N, 1]
        
        fit_T = fitness.unsqueeze(0) # [1, N]
        comp_T = complexity.unsqueeze(0) # [1, N]
        
        # Broadcasting
        not_worse = (fit <= fit_T) & (comp <= comp_T)
        better = (fit < fit_T) | (comp < comp_T)
        dominates = not_worse & better # [N, N]. dominates[i, j] is True if i dominates j
        
        # 2. Compute Domination Counts (how many dominate me)
        # sum over i (rows) for each j (col)
        domination_counts = dominates.sum(dim=0).long() # [N]
        
        # 3. Iteratively peel fronts
        ranks = torch.zeros(n, dtype=torch.long, device=self.device)
        active_mask = torch.ones(n, dtype=torch.bool, device=self.device)
        fronts = []
        current_rank = 0
        
        while active_mask.any():
            # Current front: domination_count == 0 AND active
            current_front_mask = (domination_counts == 0) & active_mask
            
            if not current_front_mask.any():
                # Should not happen in DAG, but safety break
                break
                
            front_indices = torch.nonzero(current_front_mask).squeeze(1)
            fronts.append(front_indices)
            ranks[current_front_mask] = current_rank
            
            # Remove current front
            active_mask[current_front_mask] = False
            
            # Update counts
            # For every i in current_front, if i dominates j, decrement count[j]
            # dominates[current_front_mask, :] is [F, N]
            # We want to subtract active rows from counts
            # But we can just use matrix multiplication or sum?
            # Reduce active domination matrix
            # domination_counts -= dominates[current_front_mask].sum(dim=0)
            
            # Optimization: dominates matrix allows quick update
            # count_j_new = count_j_old - sum_{i in front} (i dominates j)
            decrement = dominates[current_front_mask].sum(dim=0)
            domination_counts = domination_counts - decrement
            
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
        crowding = torch.zeros(n, dtype=torch.float64, device=self.device)
        
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
             dists = torch.zeros(k, dtype=torch.float64, device=self.device)
             
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
                
                dists = torch.zeros(k, dtype=torch.float64, device=self.device)
                
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
             dists = torch.zeros(k, dtype=torch.float64, device=self.device)
             
             idx = torch.argsort(f_vals)
             dists[idx[0]] = float('inf'); dists[idx[-1]] = float('inf')
             if (f_vals[idx[-1]]-f_vals[idx[0]]) > 1e-9: dists[idx[1:-1]] += (f_vals[idx[2:]]-f_vals[idx[:-2]])/(f_vals[idx[-1]]-f_vals[idx[0]])

             idx = torch.argsort(c_vals)
             dists[idx[0]] = float('inf'); dists[idx[-1]] = float('inf') 
             if (c_vals[idx[-1]]-c_vals[idx[0]]) > 1e-9: dists[idx[1:-1]] += (c_vals[idx[2:]]-c_vals[idx[:-2]])/(c_vals[idx[-1]]-c_vals[idx[0]])
             
             _, top_idx = torch.topk(dists, self.max_front_size)
             front = front[top_idx]
             
        return front.tolist()

