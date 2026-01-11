
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
        
        stack_top, sp, err = run_vm_cupy(
            population, x, constants,
            self.grammar.token_to_id['<PAD>'],
            self.id_x_legacy,
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
