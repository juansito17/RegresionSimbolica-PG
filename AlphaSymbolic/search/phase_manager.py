
import numpy as np
import torch
from AlphaSymbolic.core.grammar import ExpressionTree
from AlphaSymbolic.utils.optimize_constants import optimize_constants

class PhaseManager:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def detect_phase_change(self, x, residuals, min_samples=4):
        """
        Detects the best Regime Split:
        1. Linear Threshold (x < S)
        2. Parity (Even vs Odd)
        
        Returns:
            best_meta: dict with 'type', 'split_val' (if linear), 'gain'
        """
        n = len(x)
        if n < 2 * min_samples:
            return {'type': None, 'gain': 0.0}

        total_var = np.var(residuals) + 1e-9
        
        # --- Strategy A: Linear Threshold ---
        best_gain_linear = 0.0
        best_split_linear = None
        
        # Sort for linear scan
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        r_sorted = residuals[sort_idx]
        
        for i in range(min_samples, n - min_samples):
            if x_sorted[i] == x_sorted[i-1]: continue
            
            threshold = (x_sorted[i-1] + x_sorted[i]) / 2.0
            r_left = r_sorted[:i]
            r_right = r_sorted[i:]
            
            w_var = (len(r_left)/n)*np.var(r_left) + (len(r_right)/n)*np.var(r_right)
            gain = total_var - w_var
            
            if gain > best_gain_linear:
                best_gain_linear = gain
                best_split_linear = threshold
                
        # --- Strategy B: Parity (Mod 2) ---
        # Mask 0: Even, Mask 1: Odd
        mask_odd = (x.astype(int) % 2 != 0)
        mask_even = ~mask_odd
        
        n_odd = np.sum(mask_odd)
        n_even = np.sum(mask_even)
        
        best_gain_parity = 0.0
        
        if n_odd >= min_samples and n_even >= min_samples:
             r_odd = residuals[mask_odd]
             r_even = residuals[mask_even]
             
             w_var_parity = (n_odd/n)*np.var(r_odd) + (n_even/n)*np.var(r_even)
             best_gain_parity = total_var - w_var_parity
             
        # --- Compare ---
        
        # Determine Winner
        if best_gain_parity > best_gain_linear:
             return {'type': 'parity', 'gain': best_gain_parity / total_var}
        else:
             return {'type': 'linear', 'split_val': best_split_linear, 'gain': best_gain_linear / total_var}

    def construct_parity_formula(self, f_even, f_odd):
        """
        Constructs a formula that switches based on parity.
        Uses sin^2(pi*x/2) as a smooth switch.
        x=0 -> sin(0)=0 -> Even
        x=1 -> sin(pi/2)=1 -> Odd
        
        Formula: (1 - Mask)*f_even + Mask*f_odd
        Mask = sin((3.14159 * x0) / 2.0)^2
        """
        mask = "sq(sin((3.1415926535 * x0) / 2.0))"
        combined = f"((1.0 - {mask}) * ({f_even}) + {mask} * ({f_odd}))"
        return combined
        
    def construct_gated_formula(self, f1_str, f2_str, split_val, transition_k=10.0):
        """
        Combines two formulas using a smooth gate (Sigmoid) for Linear Splits.
        """
        k_str = f"{float(transition_k):.4f}" 
        s_str = f"{float(split_val):.4f}"
        
        sigmoid_core = f"exp(-1.0 * {k_str} * (x0 - {s_str}))"
        sigmoid = f"(1.0 / (1.0 + {sigmoid_core}))"
        
        combined = f"((1.0 - {sigmoid}) * ({f1_str}) + {sigmoid} * ({f2_str}))"
        
        return combined
