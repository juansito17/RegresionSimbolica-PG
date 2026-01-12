
import torch
import numpy as np
from typing import Tuple
from core.grammar import ExpressionTree
from .grammar import PAD_ID, GPUGrammar
from .config import GpuGlobals

class GPUEvaluator:
    def __init__(self, grammar: GPUGrammar, device, max_stack=10):
        self.grammar = grammar
        self.device = device
        self.MAX_STACK = max_stack
        
        # New Native VM
        from .cuda_vm import CudaRPNVM
        self.vm = CudaRPNVM(grammar, device)

    def _run_vm(self, population: torch.Tensor, x: torch.Tensor, constants: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Internal VM interpreter using Native CUDA Extension.
        """
        # Ensure x is [Vars, Samples]
        # Logic in evaluate_batch handles this, but let's double check or leave it to VM which checks.
        
        # Native Call
        return self.vm.eval(population, x, constants)
                    


    def evaluate_batch(self, population: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor, constants: torch.Tensor = None) -> torch.Tensor:
        """
        Evaluates the RPN population on the GPU over multiple samples.
        x: [Vars, Samples]
        y_target: [Samples]
        constants: [PopSize, K]
        Returns: RMSE per individual [PopSize]
        """
        B_pop, L = population.shape
        
        # Determine number of samples
        # x is [Vars, Samples]
        if x.dim() == 1:
            # Single variable, N samples? Or 1 sample D vars?
            # Assume 1 var, N samples if 1D?
            # But grammar usually expects [Vars, N].
            # Let's force [Vars, N] in engine/caller.
            x = x.unsqueeze(0)
            
        # Robust Shape Detection
        # Expectation: x can be [Vars, Samples] (Legacy) or [Samples, Vars] (Standard)
        # We need internally: [Vars, Samples] so that x.T becomes [Samples, Vars] for VM
        
        if x.dim() == 2:
            if x.shape[1] == y_target.shape[0] and x.shape[0] != y_target.shape[0]:
                # Matches [Vars, Samples] - Do nothing
                pass
            elif x.shape[0] == y_target.shape[0]:
                # Matches [Samples, Vars] -> Transpose
                x = x.T
        
        N_vars, N_samples = x.shape
        
        # We need to run B_pop * N_samples executions
        
        # 1. Expand Population: Virtual (Chunking handles it)
        # We process in chunks to avoid OOM
        
        # We process in chunks to avoid OOM
        # Optimized for RTX 3050 (4GB) with small dataset (25 samples)
        # 200,000 individuals * 25 samples * 8 bytes ~ 40MB per buffer
        max_chunk_inds = 200000
        
        # Output buffer for RMSE only
        all_rmse = []
        
        # Pre-process Target
        y_target_chunk = y_target.flatten().unsqueeze(0) # [1, N]
        
        # Transpose X for Cupy VM: [N, Vars] -> NO, CUDA VM expects [Vars, N]
        # cupy_vm expects x as [Samples, Features] and expands population against it
        # But CUDA VM expects [Vars, Samples] for coalescing.
        # x is [Vars, Samples] here.
        x_for_vm = x 
        
        for i in range(0, B_pop, max_chunk_inds):
            end_i = min(B_pop, i + max_chunk_inds)
            
            # Sub-batch of population
            sub_pop = population[i:end_i]
            sub_c = constants[i:end_i] if constants is not None else None
            
            # Run VM
            f_preds, sp, err = self._run_vm(sub_pop, x_for_vm, sub_c)
            
            current_B = sub_pop.shape[0]
            
            # Process Validity within chunk
            # f_preds is flattened [current_B * N]
            is_valid = (sp == 1) & (~err)
            f_preds = torch.where(is_valid & ~torch.isnan(f_preds) & ~torch.isinf(f_preds), 
                                  f_preds, 
                                  torch.tensor(1e300, device=self.device, dtype=torch.float64))
            
            # Reshape to [current_B, N]
            preds_mat = f_preds.view(current_B, N_samples)
            
            # Compare (Broadcasting y_target [1, N])
            if GpuGlobals.LOSS_FUNCTION == 'RMSLE':
                # RMSLE: log(pred + 1) - log(target + 1)
                # We use abs() to handle negative predictions gracefully, though ideally models shouldn't produce them for count data.
                # Clamp to avoid log(0)
                log_pred = torch.log(torch.abs(preds_mat) + 1.0)
                log_target = torch.log(torch.abs(y_target_chunk) + 1.0)
                
                diff = log_pred - log_target
                sq_diff = diff**2
                mse = torch.mean(sq_diff, dim=1) # [current_B]
                
                # RMSE of Log Errors = RMSLE
                metric_score = torch.sqrt(torch.where(torch.isnan(mse) | torch.isinf(mse), 
                                              torch.tensor(1e150, device=self.device, dtype=torch.float64), 
                                              mse))
            else:
                # Standard RMSE
                diff = preds_mat - y_target_chunk
                sq_diff = diff**2
                mse = torch.mean(sq_diff, dim=1) # [current_B]
                
                metric_score = torch.sqrt(torch.where(torch.isnan(mse) | torch.isinf(mse), 
                                              torch.tensor(1e150, device=self.device, dtype=torch.float64), 
                                              mse))
                                          
            all_rmse.append(metric_score)
             
            # Cleanup
            del sub_pop, sub_c, f_preds, sp, err, preds_mat, diff, sq_diff, mse, metric_score
            # torch.cuda.empty_cache() 
            
        final_rmse = torch.cat(all_rmse)
        
        return final_rmse

    def evaluate_differentiable(self, population: torch.Tensor, constants: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """
        Evaluates population with Autograd enabled to return Loss [PopSize].
        Supports backprop to constants.
        """
        y_flat = y_target.flatten()
        n_samples = y_flat.shape[0]
        
        # Robust Shape Detection
        # Standardize x to [Vars, Samples]
        if x.dim() == 1:
            x = x.unsqueeze(0) # [1, Samples] (Assuming input was Samples)
            # Check if it was actually [Vars] (1 sample)? Rare. 
            # Usually optimization passes x as [Vars, Samples] or [Samples, Vars]
            if x.shape[1] != n_samples and x.shape[0] == n_samples:
                x = x.T

        if x.dim() == 2:
            if x.shape[1] == n_samples:
                # [Vars, Samples] - Correct
                pass
            elif x.shape[0] == n_samples:
                # [Samples, Vars] - Transpose to [Vars, Samples]
                x = x.T
        
        N_samples = x.shape[1]
        
        # _run_vm expects [Vars, Samples]
        final_preds, sp, has_error = self._run_vm(population, x, constants)
        is_valid = (sp == 1) & (~has_error)
        
        # Reshape to [B, N_samples]
        valid_matrix = is_valid.view(population.shape[0], N_samples)
        preds = final_preds.view(population.shape[0], N_samples)
        target = y_target.flatten().unsqueeze(0).expand_as(preds)
        
        if GpuGlobals.LOSS_FUNCTION == 'RMSLE':
            # RMSLE Differentiable
            log_pred = torch.log(torch.abs(preds) + 1.0)
            log_target = torch.log(torch.abs(target) + 1.0)
            
            diff = log_pred - log_target
            sq_err = diff**2
            
            # Masking
            masked_sq_err = torch.where(valid_matrix, sq_err, torch.tensor(0.0, device=self.device, dtype=torch.float64))
            loss = masked_sq_err.mean(dim=1)
            return loss
            
        else:
            # MSE
            sq_err = (preds - target)**2
            sq_err = torch.clamp(sq_err, max=1e10)
            
            masked_sq_err = torch.where(valid_matrix, sq_err, torch.tensor(0.0, device=self.device, dtype=torch.float64))
            loss = masked_sq_err.mean(dim=1)
            return loss
    
    def evaluate_batch_full(self, population: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor, constants: torch.Tensor = None) -> torch.Tensor:
        B, L = population.shape
        y_flat = y_target.flatten()
        n_samples = y_flat.shape[0]
        
        # Robust Shape Detection
        # Standardize x to [Vars, Samples]
        if x.dim() == 1:
            x = x.unsqueeze(0) 

        if x.dim() == 2:
            if x.shape[1] == n_samples:
                # [Vars, Samples] - Correct
                pass
            elif x.shape[0] == n_samples:
                # [Samples, Vars] - Transpose to [Vars, Samples]
                x = x.T
        
        D = x.shape[1] # Samples
        
        # VM expects [Vars, Samples]
        x_for_vm = x
        
        # Logic mirrors evaluate_batch but returns full [B, D] errors
        max_chunk_inds = 100000 # Smaller chunk size for full matrix (D=25 -> 2.5M elems per chunk)
        
        all_abs_errors = []
        
        # Pre-process Target [1, D]
        target_matrix_chunk = y_target.flatten().unsqueeze(0) 

        for i in range(0, B, max_chunk_inds):
            end_i = min(B, i + max_chunk_inds)
            
            sub_pop = population[i:end_i]
            sub_c = constants[i:end_i] if constants is not None else None
            current_B = sub_pop.shape[0]
            
            # Run VM
            final_preds, sp, has_error = self._run_vm(sub_pop, x_for_vm, sub_c)
            is_valid = (sp == 1) & (~has_error)
            
            final_preds = torch.where(is_valid & ~torch.isnan(final_preds) & ~torch.isinf(final_preds), 
                                      final_preds, 
                                      torch.tensor(1e300, device=self.device, dtype=torch.float64))
            
            # Reshape to [current_B, D]
            preds_matrix = final_preds.view(current_B, D)
            
            # Broadcast Target
            # target is [1, D], preds is [cur_B, D] => Broadcast works automatically
            abs_err = torch.abs(preds_matrix - target_matrix_chunk)
            
            abs_err = torch.where(torch.isnan(abs_err) | torch.isinf(abs_err), 
                                  torch.tensor(1e300, device=self.device, dtype=torch.float64), 
                                  abs_err)
            
            all_abs_errors.append(abs_err)
            
            del sub_pop, sub_c, final_preds, sp, has_error, preds_matrix, abs_err
            # torch.cuda.empty_cache() # Optional speed vs memory trade-off

        return torch.cat(all_abs_errors, dim=0)
