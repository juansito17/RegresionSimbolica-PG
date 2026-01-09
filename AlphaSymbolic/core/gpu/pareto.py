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
    
    def non_dominated_sort(self, fitness: torch.Tensor, complexity: torch.Tensor) -> List[List[int]]:
        """
        Perform non-dominated sorting on the population.
        
        Args:
            fitness: [PopSize] RMSE values
            complexity: [PopSize] tree sizes
            
        Returns:
            List of fronts, where each front is a list of indices
        """
        n = fitness.shape[0]
        fitness_cpu = fitness.cpu().numpy()
        complexity_cpu = complexity.cpu().numpy()
        
        # For each individual, count how many dominate it
        domination_count = np.zeros(n, dtype=np.int32)
        dominated_by = [[] for _ in range(n)]  # Who each individual dominates
        
        for i in range(n):
            for j in range(i + 1, n):
                obj_i = (fitness_cpu[i], complexity_cpu[i])
                obj_j = (fitness_cpu[j], complexity_cpu[j])
                
                if self.dominates(obj_i, obj_j):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif self.dominates(obj_j, obj_i):
                    dominated_by[j].append(i)
                    domination_count[i] += 1
        
        # Build fronts
        fronts = []
        current_front = []
        
        # First front: individuals with domination_count = 0
        for i in range(n):
            if domination_count[i] == 0:
                current_front.append(i)
        
        while current_front:
            fronts.append(current_front)
            next_front = []
            
            for i in current_front:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            current_front = next_front
        
        return fronts
    
    def crowding_distance(self, front: List[int], fitness: torch.Tensor, complexity: torch.Tensor) -> torch.Tensor:
        """
        Calculate crowding distance for individuals in a front.
        
        Args:
            front: List of indices in this front
            fitness: [PopSize] RMSE values
            complexity: [PopSize] tree sizes
            
        Returns:
            [len(front)] crowding distances
        """
        n = len(front)
        if n <= 2:
            return torch.full((n,), float('inf'), device=self.device)
        
        distances = torch.zeros(n, device=self.device, dtype=torch.float64)
        
        # For each objective
        for obj_vals in [fitness, complexity]:
            # Get values for this front
            front_vals = obj_vals[front].cpu().numpy()
            
            # Sort by objective
            sorted_idx = np.argsort(front_vals)
            
            # Boundary points get infinite distance
            distances[sorted_idx[0]] = float('inf')
            distances[sorted_idx[-1]] = float('inf')
            
            # Normalize by range
            obj_range = front_vals[sorted_idx[-1]] - front_vals[sorted_idx[0]]
            if obj_range < 1e-9:
                continue
            
            # Calculate crowding distance for interior points
            for i in range(1, n - 1):
                distances[sorted_idx[i]] += (front_vals[sorted_idx[i + 1]] - front_vals[sorted_idx[i - 1]]) / obj_range
        
        return distances
    
    def select(self, population: torch.Tensor, fitness: torch.Tensor, complexity: torch.Tensor, n_select: int) -> torch.Tensor:
        """
        Select n_select individuals using NSGA-II selection.
        
        Args:
            population: [PopSize, L] RPN tensors
            fitness: [PopSize] RMSE values
            complexity: [PopSize] tree sizes
            n_select: Number of individuals to select
            
        Returns:
            [n_select] tensor of selected indices
        """
        # Non-dominated sorting
        fronts = self.non_dominated_sort(fitness, complexity)
        
        selected = []
        
        for front in fronts:
            if len(selected) + len(front) <= n_select:
                # Add entire front
                selected.extend(front)
            else:
                # Need to select subset using crowding distance
                remaining = n_select - len(selected)
                
                # Calculate crowding distance
                distances = self.crowding_distance(front, fitness, complexity)
                
                # Select by highest crowding distance
                _, sorted_idx = torch.sort(distances, descending=True)
                for i in range(remaining):
                    selected.append(front[sorted_idx[i].item()])
                
                break
        
        return torch.tensor(selected, device=self.device, dtype=torch.long)
    
    def get_pareto_front(self, fitness: torch.Tensor, complexity: torch.Tensor) -> List[int]:
        """
        Get indices of individuals in the Pareto front.
        
        Args:
            fitness: [PopSize] RMSE values
            complexity: [PopSize] tree sizes
            
        Returns:
            List of indices in the Pareto front
        """
        fronts = self.non_dominated_sort(fitness, complexity)
        
        if not fronts:
            return []
        
        front = fronts[0]
        
        # Limit size
        if len(front) > self.max_front_size:
            distances = self.crowding_distance(front, fitness, complexity)
            _, sorted_idx = torch.sort(distances, descending=True)
            front = [front[sorted_idx[i].item()] for i in range(self.max_front_size)]
        
        return front
