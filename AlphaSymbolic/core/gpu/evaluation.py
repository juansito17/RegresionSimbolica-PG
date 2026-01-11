
import torch
import numpy as np
from typing import Tuple
from core.grammar import ExpressionTree
from .grammar import PAD_ID, GPUGrammar

class GPUEvaluator:
    def __init__(self, grammar: GPUGrammar, device, max_stack=10):
        self.grammar = grammar
        self.device = device
        self.MAX_STACK = max_stack
        
        # Precompute common token IDs for VM speed
        self._cache_ids()

    def _cache_ids(self):
        self.id_C = self.grammar.token_to_id.get('C', -100)
        self.id_pi = self.grammar.token_to_id.get('pi', -100)
        self.id_e = self.grammar.token_to_id.get('e', -100)
        
        self.op_add = self.grammar.token_to_id.get('+', -100)
        self.op_sub = self.grammar.token_to_id.get('-', -100)
        self.op_mul = self.grammar.token_to_id.get('*', -100)
        self.op_div = self.grammar.token_to_id.get('/', -100)
        self.op_pow = self.grammar.token_to_id.get('pow', -100)
        self.op_mod = self.grammar.token_to_id.get('%', -100)
        
        self.op_sin = self.grammar.token_to_id.get('sin', -100)
        self.op_cos = self.grammar.token_to_id.get('cos', -100)
        self.op_tan = self.grammar.token_to_id.get('tan', -100)
        self.op_asin = self.grammar.token_to_id.get('S', -100)
        self.op_acos = self.grammar.token_to_id.get('acos', -100)
        self.op_atan = self.grammar.token_to_id.get('T', -100)
        self.op_exp = self.grammar.token_to_id.get('e', -100) # 'e' operator
        self.op_log = self.grammar.token_to_id.get('log', -100)
        self.op_sqrt = self.grammar.token_to_id.get('sqrt', -100)
        self.op_abs = self.grammar.token_to_id.get('abs', -100)
        self.op_neg = self.grammar.token_to_id.get('neg', -100)
        
        self.op_fact = self.grammar.token_to_id.get('!', -100)
        self.op_floor = self.grammar.token_to_id.get('_', -100)
        self.op_gamma = self.grammar.token_to_id.get('g', -100)
        
        self.var_ids = [self.grammar.token_to_id.get(v, -100) for v in self.grammar.active_variables]
        self.id_x_legacy = self.grammar.token_to_id.get('x', -100)
        
        self.pi_val = torch.tensor(np.pi, device=self.device, dtype=torch.float64)
        self.e_val = torch.tensor(np.e, device=self.device, dtype=torch.float64)

    def _run_vm(self, population: torch.Tensor, x: torch.Tensor, constants: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Internal VM interpreter to evaluate RPN population on the GPU.
        Uses JIT optimized version.
        Returns: (final_predictions, stack_pointer, has_error)
        """
        # Prepare inputs for JIT
        if constants is None:
            # JIT expects tensor, pass empty or create 1 dummy column
            # Be careful with empty tensor shape.
            # But constants handles 'c' logic. 
            # If constants is None, we should pass a dummy valid tensor? 
            # run_vm_jit handles constants expansion.
            # Let's pass a dummy tensor of size [B, 0] or [B, 1] filled with zeros?
            # constants arg in JIT: 
            # const_expanded = constants.unsqueeze(1).expand(B, D, constants.shape[1])...
            # if constants is None, shape[1] is problem?
            # We'll handle constants=None by passing a 1-element dummy and ignoring it via masks (if no C token).
            # But simpler: pass empty tensor [B, 0]
            constants = torch.zeros((population.shape[0], 0), device=self.device, dtype=torch.float64)
            
        
        # Collect IDs for JIT/Cupy
        # We need these to be ints.
        # Operators
        op_add = self.grammar.token_to_id.get('+', -999)
        op_sub = self.grammar.token_to_id.get('-', -999)
        op_mul = self.grammar.token_to_id.get('*', -999)
        op_div = self.grammar.token_to_id.get('/', -999)
        op_pow = self.grammar.token_to_id.get('pow', -999)
        op_mod = self.grammar.token_to_id.get('%', -999)
        
        op_sin = self.grammar.token_to_id.get('sin', -999)
        op_cos = self.grammar.token_to_id.get('cos', -999)
        op_tan = self.grammar.token_to_id.get('tan', -999)
        
        op_log = self.grammar.token_to_id.get('log', -999)
        op_exp = self.grammar.token_to_id.get('exp', -999)
        
        op_sqrt = self.grammar.token_to_id.get('sqrt', -999)
        op_abs = self.grammar.token_to_id.get('abs', -999)
        op_neg = self.grammar.token_to_id.get('neg', -999)
        
        op_fact = self.grammar.token_to_id.get('!', -999)
        op_floor = self.grammar.token_to_id.get('_', -999)
        op_gamma = self.grammar.token_to_id.get('g', -999)
        
        op_asin = self.grammar.token_to_id.get('S', -999)
        op_acos = self.grammar.token_to_id.get('acos', -999)
        op_atan = self.grammar.token_to_id.get('T', -999)
        
        
        # Literals
        id_1 = self.grammar.token_to_id.get('1', -999)
        id_2 = self.grammar.token_to_id.get('2', -999)
        id_3 = self.grammar.token_to_id.get('3', -999)
        id_5 = self.grammar.token_to_id.get('5', -999)
        
        # Create literals tensor
        lit_vals = torch.tensor([1.0, 2.0, 3.0, 5.0], device=self.device, dtype=torch.float64)
        
        
        # from .jit_vm_unrolled import run_vm_jit_unrolled
        from .cupy_vm import run_vm_cupy
        
        
        # Determine variable mapping
        first_var = self.grammar.active_variables[0]
        id_x_start = self.grammar.token_to_id[first_var]
        num_vars = len(self.grammar.active_variables)

        stack_top, sp, err = run_vm_cupy(
            population, x, constants,
            self.grammar.token_to_id['<PAD>'],
            id_x_start, num_vars,
            self.id_C,
            self.id_pi,
            self.id_e,
            id_1, id_2, id_3, id_5,
            op_add, op_sub, op_mul, op_div, op_pow, op_mod,
            op_sin, op_cos, op_tan,
            op_log, op_exp,
            op_sqrt, op_abs, op_neg,
            op_fact, op_floor, op_gamma,
            op_asin, op_acos, op_atan,
            self.pi_val, self.e_val
        )
        
        return stack_top, sp, err
                    


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
        max_chunk_inds = 2000 # 2000 individuals per chunk -> 200k executions
        
        # Output buffer for RMSE only
        all_rmse = []
        
        # Pre-process Target
        y_target_chunk = y_target.flatten().unsqueeze(0) # [1, N]
        
        # Transpose X for Cupy VM: [N, Vars]
        # cupy_vm expects x as [Samples, Features] and expands population against it
        x_for_vm = x.T # [N, Vars]
        
        for i in range(0, B_pop, max_chunk_inds):
            end_i = min(B_pop, i + max_chunk_inds)
            
            # Sub-batch of population
            sub_pop = population[i:end_i]
            sub_c = constants[i:end_i] if constants is not None else None
            
            # Run VM - Let Cupy VM handle expansion [B, N]
            # We pass x_for_vm [N, Vars]. 
            # Cupy VM expands sub_pop to [current_B * N]
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
            diff = preds_mat - y_target_chunk
            sq_diff = diff**2
            mse = torch.mean(sq_diff, dim=1) # [current_B]
            
            rmse = torch.sqrt(torch.where(torch.isnan(mse) | torch.isinf(mse), 
                                          torch.tensor(1e150, device=self.device, dtype=torch.float64), 
                                          mse))
                                          
            all_rmse.append(rmse)
             
            # Cleanup
            del sub_pop, sub_c, f_preds, sp, err, preds_mat, diff, sq_diff, mse, rmse
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
        if x.dim() == 1:
            x = x.unsqueeze(1) # [Samples, 1]
            
        if x.dim() == 2:
            if x.shape[0] == n_samples:
                # [Samples, Vars]
                pass
            elif x.shape[1] == n_samples:
                # [Vars, Samples]
                x = x.T
        
        N_samples = x.shape[0]
        final_preds, sp, has_error = self._run_vm(population, x, constants)
        is_valid = (sp == 1) & (~has_error)
        
        # Reshape to [B, N_samples]
        valid_matrix = is_valid.view(population.shape[0], N_samples)
        preds = final_preds.view(population.shape[0], N_samples)
        target = y_target.flatten().unsqueeze(0).expand_as(preds)
        
        sq_err = (preds - target)**2
        sq_err = torch.clamp(sq_err, max=1e10)
        
        masked_sq_err = torch.where(valid_matrix, sq_err, torch.tensor(0.0, device=self.device, dtype=torch.float64))
        loss = masked_sq_err.mean(dim=1)
        return loss
    
    def evaluate_batch_full(self, population: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor, constants: torch.Tensor = None) -> torch.Tensor:
        B, L = population.shape
        y_flat = y_target.flatten()
        n_samples = y_flat.shape[0]
        
        # Robust Shape Detection (Same as evaluate_batch)
        if x.dim() == 1:
            x = x.unsqueeze(1)

        if x.dim() == 2:
            if x.shape[0] == n_samples:
                # [Samples, Vars]
                pass
            elif x.shape[1] == n_samples:
                # [Vars, Samples]
                x = x.T
        
        D = x.shape[0] # Samples
        
        # VM expects [Samples, Vars]
        x_for_vm = x
        
        final_preds, sp, has_error = self._run_vm(population, x_for_vm, constants) 
        is_valid = (sp == 1) & (~has_error)
        
        final_preds = torch.where(is_valid & ~torch.isnan(final_preds) & ~torch.isinf(final_preds), 
                                  final_preds, 
                                  torch.tensor(1e300, device=self.device, dtype=torch.float64))
        
        # Reshape to [B, D]
        preds_matrix = final_preds.view(B, D)
        target_matrix = y_target.flatten().unsqueeze(0).expand(B, -1)
        abs_err = torch.abs(preds_matrix - target_matrix)
        
        abs_err = torch.where(torch.isnan(abs_err) | torch.isinf(abs_err), 
                              torch.tensor(1e300, device=self.device, dtype=torch.float64), 
                              abs_err)
        return abs_err
