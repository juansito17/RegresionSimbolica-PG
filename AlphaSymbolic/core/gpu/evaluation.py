
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
        self.op_acos = self.grammar.token_to_id.get('C', -100)
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

    @torch.compile(mode="reduce-overhead", dynamic=False)
    def _run_vm(self, population: torch.Tensor, x: torch.Tensor, constants: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Internal VM interpreter to evaluate RPN population on the GPU.
        Returns: (final_predictions, stack_pointer, has_error)
        """
        B, L = population.shape
        D = x.shape[0]
        eff_B = B * D
        MAX_STACK = self.MAX_STACK
        
        pop_expanded = population.unsqueeze(1).expand(-1, D, -1).reshape(eff_B, L)
        const_expanded = None
        if constants is not None:
             const_expanded = constants.unsqueeze(1).expand(-1, D, -1).reshape(eff_B, -1)
             
        if x.ndim == 1:
            x_expanded = x.unsqueeze(0).expand(B, -1).reshape(eff_B, 1)
        else:
            x_expanded = x.unsqueeze(0).expand(B, -1, -1).reshape(eff_B, x.shape[1])
            
        stack = torch.zeros(eff_B, MAX_STACK, device=self.device, dtype=torch.float64)
        sp = torch.zeros(eff_B, device=self.device, dtype=torch.long)
        const_counters = torch.zeros(eff_B, device=self.device, dtype=torch.long)
        has_error = torch.zeros(eff_B, dtype=torch.bool, device=self.device)
        
        for i in range(L):
            token = pop_expanded[:, i]
            active_mask = (token != PAD_ID)
            if not active_mask.any(): continue
            
            push_vals = torch.zeros(eff_B, device=self.device, dtype=torch.float64)
            is_operand = torch.zeros(eff_B, dtype=torch.bool, device=self.device)
            
            # --- 1. Push Operands ---
            
            # Legacy 'x'
            mask = (token == self.id_x_legacy)
            if mask.any():
                push_vals[mask] = x_expanded[mask, 0]
                is_operand = is_operand | mask
                
            # Variables x0..xn
            for v_idx, vid in enumerate(self.var_ids):
                mask = (token == vid)
                if mask.any():
                    v_col = v_idx if v_idx < x_expanded.shape[1] else 0
                    push_vals[mask] = x_expanded[mask, v_col]
                    is_operand = is_operand | mask
            
            # Global Constants (pi, e)
            mask = (token == self.id_pi)
            if mask.any(): push_vals[mask] = self.pi_val; is_operand = is_operand | mask
            # For 'e', check if it's token ID for e constant or exp operator?
            # Grammar usually distinguishes literals vs operators.
            # Original code check `id_e` for constant.
            if self.grammar.token_arity.get('e', 1) == 0:
                 # If 'e' is constant
                 mask = (token == self.id_e)
                 if mask.any(): push_vals[mask] = self.e_val; is_operand = is_operand | mask

            # Learned Constants 'C'
            mask = (token == self.id_C)
            if mask.any():
                if const_expanded is not None:
                     safe_idx = torch.clamp(const_counters, 0, const_expanded.shape[1]-1)
                     c_vals = const_expanded.gather(1, safe_idx.unsqueeze(1)).squeeze(1)
                     push_vals[mask] = c_vals[mask]
                     const_counters[mask] += 1
                else:
                     push_vals[mask] = 1.0 
                is_operand = is_operand | mask
            
            # Numeric Literals
            for val_str in ['1', '2', '3', '5']:
                vid = self.grammar.token_to_id.get(val_str, -999)
                mask = (token == vid)
                if mask.any():
                    push_vals[mask] = float(val_str)
                    is_operand = is_operand | mask
                    
            if is_operand.any():
                safe_sp = torch.clamp(sp, 0, MAX_STACK-1)
                stack = stack.scatter(1, safe_sp.unsqueeze(1), push_vals.unsqueeze(1))
                sp = sp + is_operand.long()
                
            # --- 2. Binary Operators ---
            is_binary = (token == self.op_add) | (token == self.op_sub) | (token == self.op_mul) | \
                        (token == self.op_div) | (token == self.op_pow) | (token == self.op_mod)
            
            enough_stack = (sp >= 2)
            valid_op = is_binary & enough_stack
            has_error = has_error | (is_binary & ~enough_stack)
            
            if valid_op.any():
                idx_b = torch.clamp(sp - 1, 0, MAX_STACK - 1).unsqueeze(1); val_b = stack.gather(1, idx_b).squeeze(1)
                idx_a = torch.clamp(sp - 2, 0, MAX_STACK - 1).unsqueeze(1); val_a = stack.gather(1, idx_a).squeeze(1)
                res = torch.zeros_like(val_a)
                
                m = (token == self.op_add) & valid_op; res[m] = val_a[m] + val_b[m]
                m = (token == self.op_sub) & valid_op; res[m] = val_a[m] - val_b[m]
                m = (token == self.op_mul) & valid_op; res[m] = val_a[m] * val_b[m]
                
                m = (token == self.op_div) & valid_op
                if m.any(): 
                    d = val_b[m]; bad = d.abs() < 1e-9
                    sd = torch.where(bad, torch.tensor(1.0, device=self.device, dtype=torch.float64), d)
                    out = val_a[m] / sd; out[bad] = 1e150; res[m] = out
                    
                m = (token == self.op_mod) & valid_op
                if m.any():
                    d = val_b[m]; bad = d.abs() < 1e-9
                    sd = torch.where(bad, torch.tensor(1.0, device=self.device, dtype=torch.float64), d)
                    out = torch.fmod(val_a[m], sd); out[bad] = 1e150; res[m] = out
                    
                m = (token == self.op_pow) & valid_op; 
                if m.any(): 
                    p = torch.pow(val_a[m], val_b[m])
                    bad_p = torch.isnan(p) | torch.isinf(p)
                    p[bad_p] = 1e300
                    res[m] = p
                
                wp = torch.clamp(sp - 2, 0, MAX_STACK-1)
                curr = stack.gather(1, wp.unsqueeze(1)).squeeze(1)
                fw = torch.where(valid_op, res, curr)
                stack = stack.scatter(1, wp.unsqueeze(1), fw.unsqueeze(1)); sp = sp - valid_op.long()
                
            # --- 3. Unary Operators ---
            is_unary = (token == self.op_sin) | (token == self.op_cos) | (token == self.op_tan) | \
                       (token == self.op_asin) | (token == self.op_acos) | (token == self.op_atan) | \
                       (token == self.op_exp) | (token == self.op_log) | \
                       (token == self.op_sqrt) | (token == self.op_abs) | (token == self.op_neg) | \
                       (token == self.op_fact) | (token == self.op_floor) | (token == self.op_gamma)
            
            enough_stack = (sp >= 1)
            valid_op = is_unary & enough_stack
            has_error = has_error | (is_unary & ~enough_stack)
            
            if valid_op.any():
                idx_a = torch.clamp(sp - 1, 0, MAX_STACK - 1).unsqueeze(1); val_a = stack.gather(1, idx_a).squeeze(1)
                res = torch.zeros_like(val_a)
                
                m = (token == self.op_sin) & valid_op; res[m] = torch.sin(val_a[m])
                m = (token == self.op_cos) & valid_op; res[m] = torch.cos(val_a[m])
                m = (token == self.op_tan) & valid_op; res[m] = torch.tan(val_a[m])
                
                m = (token == self.op_log) & valid_op
                if m.any(): 
                    inv = val_a[m]; s = inv > 1e-9; out = torch.full_like(inv, 1e150); out[s] = torch.log(inv[s]); res[m] = out
                
                m = (token == self.op_exp) & valid_op
                if m.any(): 
                    inv = val_a[m]; s = inv <= 700.0; out = torch.full_like(inv, 1e150); out[s] = torch.exp(inv[s]); res[m] = out
                    
                m = (token == self.op_sqrt) & valid_op; res[m] = torch.sqrt(val_a[m].abs())
                m = (token == self.op_abs) & valid_op; res[m] = torch.abs(val_a[m])
                m = (token == self.op_neg) & valid_op; res[m] = -val_a[m]
                
                m = (token == self.op_asin) & valid_op; res[m] = torch.asin(torch.clamp(val_a[m], -1.0, 1.0))
                m = (token == self.op_acos) & valid_op; res[m] = torch.acos(torch.clamp(val_a[m], -1.0, 1.0))
                m = (token == self.op_atan) & valid_op; res[m] = torch.atan(val_a[m])
                m = (token == self.op_floor) & valid_op; res[m] = torch.floor(val_a[m])
                
                m = (token == self.op_fact) & valid_op
                if m.any():
                    inv = val_a[m]; u = (inv < 0) | (inv > 170.0); out = torch.full_like(inv, 1e150)
                    si = inv.clone(); si[u] = 1.0; vc = torch.special.gamma(si + 1.0); out[~u] = vc[~u]; res[m] = out
                    
                m = (token == self.op_gamma) & valid_op
                if m.any():
                    inv = val_a[m]; u = (inv <= -1.0); out = torch.full_like(inv, 1e150)
                    si = inv.clone(); si[u] = 1.0; vc = torch.special.gammaln(si + 1.0); out[~u] = vc[~u]; res[m] = out

                wp = torch.clamp(sp - 1, 0, MAX_STACK-1); curr = stack.gather(1, wp.unsqueeze(1)).squeeze(1)
                fw = torch.where(valid_op, res, curr); stack = stack.scatter(1, wp.unsqueeze(1), fw.unsqueeze(1))
                
        return stack[:, 0], sp, has_error

    def evaluate_batch(self, population: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor, constants: torch.Tensor = None) -> torch.Tensor:
        """
        Evaluates the RPN population on the GPU.
        Returns: RMSE per individual [PopSize]
        """
        B, L = population.shape
        D = x.shape[0]
        
        final_preds, sp, has_error = self._run_vm(population, x, constants)
        
        is_valid = (sp == 1) & (~has_error)
        final_preds = torch.where(is_valid & ~torch.isnan(final_preds) & ~torch.isinf(final_preds), 
                                  final_preds, 
                                  torch.tensor(1e300, device=self.device, dtype=torch.float64))
                                  
        preds_matrix = final_preds.view(B, D)
        target_matrix = y_target.unsqueeze(0).expand(B, -1)
        mse = torch.mean((preds_matrix - target_matrix)**2, dim=1)
        
        rmse = torch.sqrt(torch.where(torch.isnan(mse) | torch.isinf(mse), 
                                      torch.tensor(1e150, device=self.device, dtype=torch.float64), 
                                      mse))
        return rmse

    def evaluate_differentiable(self, population: torch.Tensor, constants: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """
        Evaluates population with Autograd enabled to return Loss [PopSize].
        Supports backprop to constants.
        """
        final_preds, sp, has_error = self._run_vm(population, x, constants)
        is_valid = (sp == 1) & (~has_error)
        
        # Reshape to [B, D]
        valid_matrix = is_valid.view(population.shape[0], x.shape[0])
        preds = final_preds.view(population.shape[0], x.shape[0])
        target = y_target.unsqueeze(0).expand_as(preds)
        
        sq_err = (preds - target)**2
        sq_err = torch.clamp(sq_err, max=1e10)
        
        masked_sq_err = torch.where(valid_matrix, sq_err, torch.tensor(0.0, device=self.device, dtype=torch.float64))
        loss = masked_sq_err.mean(dim=1)
        return loss
    
    def evaluate_batch_full(self, population: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor, constants: torch.Tensor = None) -> torch.Tensor:
        B, L = population.shape
        D = x.shape[0]
        final_preds, sp, has_error = self._run_vm(population, x, constants) # Added has_error
        is_valid = (sp == 1) & (~has_error)
        
        final_preds = torch.where(is_valid & ~torch.isnan(final_preds) & ~torch.isinf(final_preds), 
                                  final_preds, 
                                  torch.tensor(1e300, device=self.device, dtype=torch.float64))
        
        preds_matrix = final_preds.view(B, D)
        target_matrix = y_target.unsqueeze(0).expand(B, -1)
        abs_err = torch.abs(preds_matrix - target_matrix)
        
        abs_err = torch.where(torch.isnan(abs_err) | torch.isinf(abs_err), 
                              torch.tensor(1e300, device=self.device, dtype=torch.float64), 
                              abs_err)
        return abs_err
