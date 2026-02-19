
import torch
from typing import Tuple
from core.grammar import ExpressionTree
from .grammar import PAD_ID, GPUGrammar
from .config import GpuGlobals

class GPUEvaluator:
    def __init__(self, grammar: GPUGrammar, device, max_stack=64, dtype=torch.float64):
        self.grammar = grammar
        self.device = device
        self.MAX_STACK = max_stack
        self.dtype = dtype
        
        # Max Float safe value
        # FIX BUG-6: 1e30 is too close to float32 max (~3.4e38) and can overflow
        # when squared or used in operations. Use 1e20 which is safe for:
        # - sqrt(1e20) = 1e10 (OK)
        # - mean([1e20]) = 1e20 (OK)
        # - 1e20^2 = 1e40 (would overflow, but we avoid squaring penalty values)
        # For float64, 1e100 is safe.
        self.INF_VAL = 1e20 if dtype == torch.float32 else 1e100
        
        # P1-4: Pre-create cached scalar tensor to avoid repeated allocations
        self._inf_tensor = torch.tensor(self.INF_VAL, device=self.device, dtype=self.dtype)
        
        # New Native VM
        from .cuda_vm import CudaRPNVM
        self.vm = CudaRPNVM(grammar, device)

    def _run_vm(self, population: torch.Tensor, x: torch.Tensor, constants: torch.Tensor = None, strict_mode: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Internal VM interpreter using Native CUDA Extension.
        """
        # Ensure x is [Vars, Samples]
        # Logic in evaluate_batch handles this, but let's double check or leave it to VM which checks.
        
        # Native Call
        return self.vm.eval(population, x, constants, strict_mode=strict_mode)
                    


    def evaluate_batch(self, population: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor, constants: torch.Tensor = None, strict_mode: int = 0) -> torch.Tensor:
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
        
        # FIX N1: Usar num_variables de la gramática como referencia cuando
        # la matriz es cuadrada (num_vars == num_samples)
        num_vars_grammar = len(self.grammar.active_variables)
        
        if x.dim() == 2:
            # Caso 1: Segunda dimensión coincide con num_variables y primera NO
            # Claramente es [Samples, Vars] -> necesita transposición
            if x.shape[1] == num_vars_grammar and x.shape[0] != num_vars_grammar:
                x = x.T.contiguous()
            # Caso 2: Primera dimensión coincide con num_variables y segunda NO
            # Claramente es [Vars, Samples] -> correcto
            elif x.shape[0] == num_vars_grammar and x.shape[1] != num_vars_grammar:
                pass  # Formato correcto
            # Caso 3: Matriz cuadrada donde AMBAS dimensiones coinciden con num_variables
            # Usar y_target como referencia: samples = len(y_target)
            elif x.shape[0] == num_vars_grammar and x.shape[1] == num_vars_grammar:
                # Matriz cuadrada: ambas dimensiones son num_vars
                # Verificar si el número de samples coincide con y_target
                n_samples = y_target.shape[0]
                if x.shape[1] == n_samples:
                    # [Vars, Samples] - samples en segunda dimensión coinciden
                    pass
                elif x.shape[0] == n_samples:
                    # [Samples, Vars] - samples en primera dimensión
                    x = x.T.contiguous()
                # Si ambas coinciden (extremadamente raro: num_vars == num_samples == len(y))
                # Asumimos [Vars, Samples] por convención
            # Fallback: lógica anterior para casos donde num_variables no coincide
            elif x.shape[1] == y_target.shape[0] and x.shape[0] != y_target.shape[0]:
                # Matches [Vars, Samples] - Do nothing
                pass
            elif x.shape[0] == y_target.shape[0] and x.shape[0] != x.shape[1]:
                # Matches [Samples, Vars] -> Transpose
                x = x.T.contiguous()
        
        N_vars, N_samples = x.shape
        
        # We need to run B_pop * N_samples executions
        
        # 1. Expand Population: Virtual (Chunking handles it)
        # We process in chunks to avoid OOM
        
        # We process in chunks to avoid OOM
        # Optimized for RTX 3050 (4GB) with small dataset (25 samples)
        # 200,000 individuals * 25 samples * 8 bytes ~ 40MB per buffer
        max_chunk_inds = 1000000
        
        # Pre-allocate output buffer (avoid torch.cat at the end)
        final_rmse = torch.full((B_pop,), self.INF_VAL, device=self.device, dtype=self.dtype)
        
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
            f_preds, sp, err = self._run_vm(sub_pop, x_for_vm, sub_c, strict_mode=strict_mode)
            
            current_B = sub_pop.shape[0]
            
            # Process Validity within chunk: (Stack OK) AND (No Kernel Error)
            is_valid = (sp == 1) & (err == 0)
            
            # Penalize: Replace invalid, NaNs or Infs with INF_VAL
            # This must be done BEFORE calculating diff to avoid INF/NaN leakage
            f_preds = torch.where(is_valid & ~torch.isnan(f_preds) & ~torch.isinf(f_preds), 
                                  f_preds, 
                                  self._inf_tensor)
            
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
                                              self._inf_tensor, 
                                              mse))
            else:
                # Standard RMSE
                diff = preds_mat - y_target_chunk
                sq_diff = diff**2
                mse = torch.mean(sq_diff, dim=1) # [current_B]
                
                metric_score = torch.sqrt(torch.where(torch.isnan(mse) | torch.isinf(mse), 
                                              self._inf_tensor, 
                                              mse))
                                          
            # Write directly to pre-allocated buffer
            final_rmse[i:end_i] = metric_score
             
            # Cleanup
            del sub_pop, sub_c, f_preds, sp, err, preds_mat, diff, sq_diff, mse, metric_score
            # torch.cuda.empty_cache() 
            
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
                x = x.T.contiguous()
        
        N_samples = x.shape[1]
        
        # _run_vm expects [Vars, Samples]
        final_preds, sp, has_error = self._run_vm(population, x, constants)
        
        # Robust Validity: (Stack OK) AND (No Kernel Error) AND (No NaNs) AND (No Infs)
        # We check per sample first
        sample_valid = (sp == 1) & (has_error == 0) & (~torch.isnan(final_preds)) & (~torch.isinf(final_preds))
        
        # Reshape to [B, N_samples]
        valid_matrix = sample_valid.view(population.shape[0], N_samples)
        preds = final_preds.view(population.shape[0], N_samples)
        target = y_target.flatten().unsqueeze(0).expand_as(preds)
        
        if GpuGlobals.LOSS_FUNCTION == 'RMSLE':
            # RMSLE Differentiable
            log_pred = torch.log(torch.abs(preds) + 1.0)
            log_target = torch.log(torch.abs(target) + 1.0)
            
            diff = log_pred - log_target
            sq_err = diff**2
            
            # Masking: Use high penalty for invalid cases
            # PENALTY must be significantly larger than any possible real target error
            masked_sq_err = torch.where(valid_matrix, sq_err, self._inf_tensor)
            loss = masked_sq_err.mean(dim=1)
            return loss, preds
            
        else:
            # MSE
            sq_err = (preds - target)**2
            sq_err = torch.clamp(sq_err, max=1e10)
            
            # Masking: Use INF_VAL penalty
            masked_sq_err = torch.where(valid_matrix, sq_err, self._inf_tensor)
            loss = masked_sq_err.mean(dim=1)
            return loss, preds
    
    def evaluate_batch_full(self, population: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor, constants: torch.Tensor = None, strict_mode: int = 0) -> torch.Tensor:
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
                x = x.T.contiguous()
        
        D = x.shape[1] # Samples
        
        # VM expects [Vars, Samples]
        x_for_vm = x
        
        # Logic mirrors evaluate_batch but returns full [B, D] errors
        max_chunk_inds = 100000 # Smaller chunk size for full matrix (D=25 -> 2.5M elems per chunk)
        
        # Pre-allocate output buffer (avoid torch.cat at the end)
        all_abs_errors = torch.full((B, D), self.INF_VAL, device=self.device, dtype=self.dtype)
        
        # Pre-process Target [1, D]
        target_matrix_chunk = y_target.flatten().unsqueeze(0) 

        for i in range(0, B, max_chunk_inds):
            end_i = min(B, i + max_chunk_inds)
            
            sub_pop = population[i:end_i]
            sub_c = constants[i:end_i] if constants is not None else None
            current_B = sub_pop.shape[0]
            
            # Run VM
            final_preds, sp, has_error = self._run_vm(sub_pop, x_for_vm, sub_c, strict_mode=strict_mode)

            # Validity: (Stack OK) AND (No Kernel Error)
            is_valid = (sp == 1) & (has_error == 0)
            
            # Penalize
            final_preds = torch.where(is_valid & ~torch.isnan(final_preds) & ~torch.isinf(final_preds), 
                                      final_preds, 
                                      self._inf_tensor)
            
            # Reshape to [current_B, D]
            preds_matrix = final_preds.view(current_B, D)
            
            # Broadcast Target
            # target is [1, D], preds is [cur_B, D] => Broadcast works automatically
            abs_err = torch.abs(preds_matrix - target_matrix_chunk)
            
            abs_err = torch.where(torch.isnan(abs_err) | torch.isinf(abs_err), 
                                  self._inf_tensor, 
                                  abs_err)
            
            # Write directly to pre-allocated buffer
            all_abs_errors[i:end_i] = abs_err
            
            del sub_pop, sub_c, final_preds, sp, has_error, preds_matrix, abs_err
            # torch.cuda.empty_cache() # Optional speed vs memory trade-off

        return all_abs_errors

    def validate_strict(self, population: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor, constants: torch.Tensor = None) -> dict:
        """
        Validates formulas using STRICT math (no protected operators).
        Runs on GPU with strict_mode=1 — domain errors (log(neg), sqrt(neg), etc.)
        are flagged via the kernel's error output.
        
        Returns dict per individual:
            'rmse': RMSE (inf if any domain error)
            'n_errors': number of data points with domain errors
            'n_total': total data points
            'is_valid': True if n_errors == 0
        """
        B, L = population.shape
        
        # Robust Shape Detection (Sync with evaluate_batch)
        num_vars_grammar = len(self.grammar.active_variables)
        if x.dim() == 2:
            if x.shape[1] == num_vars_grammar and x.shape[0] != num_vars_grammar:
                x = x.T.contiguous()
            elif x.shape[0] == num_vars_grammar and x.shape[1] != num_vars_grammar:
                pass
            elif x.shape[0] == num_vars_grammar and x.shape[1] == num_vars_grammar:
                n_samples = y_target.shape[0]
                if x.shape[1] == n_samples:
                    pass
                elif x.shape[0] == n_samples:
                    x = x.T.contiguous()
            elif x.shape[1] == y_target.shape[0] and x.shape[0] != y_target.shape[0]:
                pass
            elif x.shape[0] == y_target.shape[0] and x.shape[0] != x.shape[1]:
                x = x.T.contiguous()
        
        N_samples = x.shape[1]
        
        # Run VM with strict_mode=1
        preds, sp, err = self.vm.eval(population, x, constants, strict_mode=1)
        
        # Reshape to [B, N]
        preds = preds.view(B, N_samples)
        sp = sp.view(B, N_samples)
        err = err.view(B, N_samples)
        
        # A point has an error if: kernel error OR stack mismatch OR NaN/Inf
        has_error = (err != 0) | (sp != 1) | torch.isnan(preds) | torch.isinf(preds)
        
        # Count errors per individual
        n_errors = has_error.sum(dim=1)  # [B]
        
        # Compute RMSE only for fully valid individuals
        y_expand = y_target.flatten().unsqueeze(0).expand_as(preds)
        diff = preds - y_expand
        sq_diff = diff ** 2
        
        # Replace error points with 0 for RMSE computation (or inf)
        sq_diff = torch.where(has_error, self._inf_tensor, sq_diff)
        mse = torch.mean(sq_diff, dim=1)
        rmse = torch.sqrt(mse)
        
        # Mark individuals with any errors as inf RMSE
        rmse = torch.where(n_errors > 0, self._inf_tensor, rmse)
        
        return {
            'rmse': rmse,           # [B] tensor
            'n_errors': n_errors,   # [B] tensor  
            'n_total': N_samples,
            'is_valid': (n_errors == 0),  # [B] bool tensor
        }

