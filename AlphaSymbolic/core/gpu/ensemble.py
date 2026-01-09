"""
Ensemble / Coevolution Support for GPU GP Engine.

Provides utilities to:
- Run multiple GP engines in parallel
- Combine results from multiple runs
- Share best individuals between runs (coevolution)
"""
import torch
import numpy as np
from typing import List, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class EnsembleRunner:
    """
    Runs multiple GP engines and combines their results.
    
    Supports:
    - Parallel execution of multiple runs
    - Hall of Fame aggregation across runs
    - Best solution selection with Pareto consideration
    """
    
    def __init__(self, engine_factory: Callable, n_runs: int = 5, 
                 share_best: bool = True, share_interval: int = 100):
        """
        Args:
            engine_factory: Function that creates a TensorGeneticEngine instance
            n_runs: Number of parallel runs
            share_best: Whether to share best solutions between runs
            share_interval: Generations between sharing best solutions
        """
        self.engine_factory = engine_factory
        self.n_runs = n_runs
        self.share_best = share_best
        self.share_interval = share_interval
        
        # Hall of Fame: list of (formula_str, rmse, complexity)
        self.hall_of_fame: List[Tuple[str, float, int]] = []
        self.max_hof_size = 20
        
    def run_single(self, engine, x_values, y_targets, seeds, timeout_sec, run_id) -> Tuple[Optional[str], float, int]:
        """
        Run a single GP engine.
        
        Returns:
            (best_formula, best_rmse, run_id)
        """
        try:
            result = engine.run(x_values, y_targets, seeds, timeout_sec=timeout_sec)
            
            # Get fitness from last evaluation
            if hasattr(engine, 'best_rmse'):
                rmse = engine.best_rmse
            else:
                rmse = float('inf')
            
            return (result, rmse, run_id)
        except Exception as e:
            print(f"[Ensemble] Run {run_id} failed: {e}")
            return (None, float('inf'), run_id)
    
    def run_ensemble(self, x_values: List[float], y_targets: List[float], 
                     seeds: List[str] = None, timeout_sec: float = 10,
                     callback: Callable = None) -> str:
        """
        Run ensemble of GP engines and return best result.
        
        Args:
            x_values: Input data
            y_targets: Target data
            seeds: Optional seed formulas
            timeout_sec: Timeout per run
            callback: Optional progress callback
            
        Returns:
            Best formula found across all runs
        """
        if seeds is None:
            seeds = []
        
        # Create engines
        engines = [self.engine_factory() for _ in range(self.n_runs)]
        
        results = []
        best_formula = None
        best_rmse = float('inf')
        
        # Run sequentially (parallel would require careful GPU memory management)
        for i, engine in enumerate(engines):
            if callback:
                callback(f"Running engine {i+1}/{self.n_runs}")
            
            # Use shared seeds from Hall of Fame
            shared_seeds = seeds.copy()
            if self.share_best and self.hall_of_fame:
                top_hof = [f for f, _, _ in self.hall_of_fame[:5]]
                shared_seeds.extend(top_hof)
            
            result, rmse, _ = self.run_single(engine, x_values, y_targets, 
                                              shared_seeds, timeout_sec, i)
            
            if result:
                results.append((result, rmse))
                
                # Update Hall of Fame
                complexity = len(result) if result else 0
                self._add_to_hof(result, rmse, complexity)
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_formula = result
        
        if callback:
            callback(f"Ensemble complete. Best RMSE: {best_rmse:.6f}")
        
        return best_formula
    
    def _add_to_hof(self, formula: str, rmse: float, complexity: int):
        """Add formula to Hall of Fame if it's good enough."""
        if formula is None:
            return
        
        # Check if already in HoF
        for existing_formula, _, _ in self.hall_of_fame:
            if existing_formula == formula:
                return
        
        # Add
        self.hall_of_fame.append((formula, rmse, complexity))
        
        # Sort by (rmse, complexity) - lexicographic
        self.hall_of_fame.sort(key=lambda x: (x[1], x[2]))
        
        # Trim to max size
        if len(self.hall_of_fame) > self.max_hof_size:
            self.hall_of_fame = self.hall_of_fame[:self.max_hof_size]
    
    def get_pareto_front(self) -> List[Tuple[str, float, int]]:
        """
        Get Pareto-optimal solutions from Hall of Fame.
        
        Returns:
            List of (formula, rmse, complexity) tuples on the Pareto front
        """
        if not self.hall_of_fame:
            return []
        
        pareto = []
        for formula, rmse, complexity in self.hall_of_fame:
            is_dominated = False
            for other_formula, other_rmse, other_complexity in self.hall_of_fame:
                if other_formula == formula:
                    continue
                # Check if other dominates this
                if other_rmse <= rmse and other_complexity <= complexity:
                    if other_rmse < rmse or other_complexity < complexity:
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto.append((formula, rmse, complexity))
        
        return pareto
    
    def get_best(self) -> Optional[str]:
        """Get best formula from Hall of Fame."""
        if not self.hall_of_fame:
            return None
        return self.hall_of_fame[0][0]


def create_ensemble_runner(device=None, pop_size=1000, n_runs=5):
    """
    Factory function to create an EnsembleRunner.
    
    Args:
        device: Torch device
        pop_size: Population size per run
        n_runs: Number of runs
        
    Returns:
        EnsembleRunner instance
    """
    from . import TensorGeneticEngine
    
    def factory():
        return TensorGeneticEngine(device=device, pop_size=pop_size, n_islands=4)
    
    return EnsembleRunner(factory, n_runs=n_runs)
