
import torch
from typing import Tuple
from .config import GpuGlobals
from .evaluation import GPUEvaluator
from .operators import GPUOperators

class GPUOptimizer:
    def __init__(self, evaluator: GPUEvaluator, operators: GPUOperators, device):
        self.evaluator = evaluator
        self.operators = operators
        self.device = device

    def optimize_constants(self, population: torch.Tensor, constants: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor, steps=10, lr=0.1):
        """
        Refine constants using Gradient Descent (Adam).
        Returns: (best_constants, best_rmse)
        """
        optimized_consts = constants.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([optimized_consts], lr=lr)
        
        best_mse = torch.full((population.shape[0],), float('inf'), device=self.device, dtype=torch.float64)
        best_consts = constants.clone().detach() 
        
        for _ in range(steps):
            optimizer.zero_grad()
            
            # Use differentiable evaluation (returns Loss/MSE)
            loss_per_ind = self.evaluator.evaluate_differentiable(population, optimized_consts, x, y_target)
            
            # Track best
            current_mse = loss_per_ind.detach()
            # Handle potential NaNs in output
            valid = ~torch.isnan(current_mse)
            
            improved = (current_mse < best_mse) & valid
            if improved.any():
                best_mse[improved] = current_mse[improved]
                best_consts[improved] = optimized_consts[improved].detach()
                
            loss = loss_per_ind[valid].sum()
            
            if not loss.requires_grad: 
                break
                
            loss.backward()
            optimizer.step()
            
        return best_consts, torch.sqrt(best_mse)

    def local_search(self, population: torch.Tensor, constants: torch.Tensor, 
                     x: torch.Tensor, y: torch.Tensor, 
                     top_k: int = 10, attempts: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hill climbing: try single-token mutations on top individuals, keep improvements.
        """
        if attempts is None:
            attempts = GpuGlobals.LOCAL_SEARCH_ATTEMPTS
        
        pop_out = population.clone()
        const_out = constants.clone()
        
        fitness = self.evaluator.evaluate_batch(population, x, y, constants)
        _, top_idx = torch.topk(fitness, min(top_k, len(fitness)), largest=False)
        
        for idx in top_idx:
            idx = idx.item()
            current_rpn = population[idx:idx+1]
            current_const = constants[idx:idx+1]
            current_fit = fitness[idx].item()
            
            best_rpn = current_rpn.clone()
            best_const = current_const.clone()
            best_fit = current_fit
            
            for _ in range(attempts):
                # Mutate
                mutant = self.operators.mutate_population(current_rpn, mutation_rate=0.15)
                mutant_fit = self.evaluator.evaluate_batch(mutant, x, y, current_const)[0].item()
                
                if mutant_fit < best_fit:
                    best_rpn = mutant.clone()
                    best_fit = mutant_fit
            
            if best_fit < current_fit:
                pop_out[idx] = best_rpn[0]
                opt_const, _ = self.optimize_constants(best_rpn, best_const, x, y, steps=5)
                const_out[idx] = opt_const[0]
        
        return pop_out, const_out
