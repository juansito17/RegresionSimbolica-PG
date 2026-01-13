
import torch
import numpy as np
from .formatting import format_const

class Sniper:
    """
    'The Sniper': Special Pattern Detection Unit.
    Quickly identifies simple patterns (Linear, Geometric) before evolution begins.
    """
    def __init__(self, device):
        self.device = device
        
    def check_linear(self, x_t, y_t):
        """
        Check for y = m*x + c
        Returns formula string if found, else None
        """
        try:
            # Solve [x, 1] * [m, c]^T = y
            # A: [N, 2]
            ones = torch.ones_like(x_t)
            A = torch.stack([x_t, ones], dim=1)
            
            # Least squares
            # solution = (A^T A)^-1 A^T y
            solution = torch.linalg.lstsq(A, y_t).solution
            m = solution[0].item()
            c = solution[1].item()
            
            # Predict
            y_pred = m * x_t + c
            mse = torch.mean((y_pred - y_t)**2)
            
            if mse < 1e-6:
                m_str = format_const(m)
                c_str = format_const(c)
                # Formats: 
                # (m * x) + c if c > 0
                return f"({m_str} * x + {c_str})" if c >= 0 else f"({m_str} * x - {format_const(abs(c))})"
        except Exception as e:
            pass
        return None

    def check_geometric(self, x_t, y_t):
        """
        Check for y = A * exp(B*x) -> log(y) = log(A) + B*x
        Returns formula string if found, else None
        """
        try:
            if (y_t <= 0).any(): return None
            
            log_y = torch.log(y_t)
            
            # Solve [x, 1] * [B, log_A]^T = log_y
            ones = torch.ones_like(x_t)
            A_mat = torch.stack([x_t, ones], dim=1)
            
            solution = torch.linalg.lstsq(A_mat, log_y).solution
            B = solution[0].item()
            log_A = solution[1].item()
            
            # Predict
            y_pred = torch.exp(log_A + B * x_t) # exp(lnA + Bx) = A exp(Bx)
            mse = torch.mean((y_pred - y_t)**2)
            
            if mse < 1e-4:
                b_str = format_const(B)
                ln_a_str = format_const(log_A)
                
                inner = f"({b_str} * x)"
                if log_A >= 0:
                    inner = f"({inner} + {ln_a_str})"
                else:
                     inner = f"({inner} - {format_const(abs(log_A))})"
                     
                return f"exp({inner})"
        except Exception as e:
            pass
        return None

    def check_log_linear(self, x_t, y_t):
        """
        Check for exp(y) = mx + c -> y = log(mx + c).
        Useful when engine uses Log Transform (y_t is log(raw)).
        If this passes, it means Raw Data is Linear (exp(log(raw)) = raw = mx+c).
        We return log(mx+c).
        """
        try:
            exp_y = torch.exp(y_t)
            
            # Solve [x, 1] * [m, c]^T = exp_y
            ones = torch.ones_like(x_t)
            A = torch.stack([x_t, ones], dim=1)
            
            solution = torch.linalg.lstsq(A, exp_y).solution
            m = solution[0].item()
            c = solution[1].item()
            
            # Predict
            pred_exp_y = m * x_t + c
            # Only valid if argument to log is positive
            if (pred_exp_y <= 0).any(): return None
            
            y_pred = torch.log(pred_exp_y)
            mse = torch.mean((y_pred - y_t)**2)
            
            if mse < 1e-6:
                m_str = format_const(m)
                c_str = format_const(c)
                # Formats: log(mx + c)
                term = f"({m_str} * x)"
                if c >= 0:
                    inner = f"({term} + {c_str})"
                else:
                    inner = f"({term} - {format_const(abs(c))})"
                return f"log({inner})"
        except:
            pass
        return None

    def run(self, x_data, y_data):
        """
        Run all checks.
        x_data, y_data: CPU lists or arrays.
        """
        try:
            x_t = torch.tensor(x_data, device=self.device, dtype=torch.float32).flatten()
            y_t = torch.tensor(y_data, device=self.device, dtype=torch.float32).flatten()
            
            # Check 1: Linear (y = mx+c)
            res = self.check_linear(x_t, y_t)
            if res: 
                print(f"[The Sniper] Detected Linear Pattern: {res}")
                return res
            
            # Check 2: Geometric (y = A exp(Bx))
            res = self.check_geometric(x_t, y_t)
            if res:
                print(f"[The Sniper] Detected Geometric Pattern: {res}")
                return res
                
            # Check 3: Log-Linear (y = log(mx+c)) / Linear Raw Data
            res = self.check_log_linear(x_t, y_t)
            if res:
                print(f"[The Sniper] Detected Log-Linear Pattern: {res}")
                return res
            
        except Exception as e:
            # print(f"[The Sniper] Failed: {e}")
            pass
        return None
