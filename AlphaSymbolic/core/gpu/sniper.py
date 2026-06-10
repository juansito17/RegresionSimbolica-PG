"""
GPU-Native Sniper - Special Pattern Detection Unit
100% PyTorch implementation - NO CPU transfers.

Quickly identifies simple patterns (Linear, Polynomial, Geometric, Power Law) before evolution begins.
All operations stay on GPU for maximum performance.
"""
import torch
import math
from .formatting import format_const

class Sniper:
    """
    'The Sniper': Special Pattern Detection Unit.
    100% GPU-native - no CPU transfers.
    """
    def __init__(self, device):
        self.device = device
    
    def _simplify_val(self, val, tol=0.05):
        """Simplify a numeric value to a clean constant if close."""
        if abs(val) < tol:
            return 0.0
        rounded = round(val)
        if abs(val - rounded) < tol:
            return float(rounded)
        half = round(val * 2) / 2
        if abs(val - half) < 0.05:
            return half
        return None  # Can't simplify
    
    def _check_linear(self, x_t: torch.Tensor, y_t: torch.Tensor):
        """
        Check for y = m*x + c using torch.linalg.lstsq (GPU).
        Returns formula string if found, else None.
        """
        try:
            # Build design matrix [x, 1]
            ones = torch.ones_like(x_t)
            A = torch.stack([x_t, ones], dim=1)
            
            # Solve least squares
            solution = torch.linalg.lstsq(A, y_t).solution
            m = solution[0].item()
            c = solution[1].item()
            
            # Predict and compute MSE
            y_pred = m * x_t + c
            mse = torch.mean((y_pred - y_t) ** 2).item()
            y_var = torch.var(y_t).item()
            r2 = 1.0 - mse / y_var if y_var > 1e-12 else 1.0
            
            if mse < 1e-4 or r2 > 0.999:
                m_str = format_const(m)
                c_str = format_const(c)
                if c >= 0:
                    return f"({m_str} * x0 + {c_str})"
                else:
                    return f"({m_str} * x0 - {format_const(abs(c))})"
        except Exception:
            pass
        return None
    
    def _check_polynomial(self, x_t: torch.Tensor, y_t: torch.Tensor, max_degree=6):
        """
        Check for polynomial y = a_n*x^n + ... + a_1*x + a_0.
        Uses Vandermonde matrix + lstsq (GPU).
        Returns formula string if found, else None.
        """
        try:
            y_var = torch.var(y_t).item()
            if y_var < 1e-12:
                return None
            
            best_formula = None
            best_mse = float('inf')
            
            for degree in range(2, max_degree + 1):
                # Build Vandermonde matrix: [x^d, x^(d-1), ..., x, 1]
                # Shape: [N, degree+1]
                cols = []
                for p in range(degree, -1, -1):
                    cols.append(x_t ** p)
                V = torch.stack(cols, dim=1)
                
                # Solve least squares
                coeffs = torch.linalg.lstsq(V, y_t).solution  # [degree+1]
                
                # Predict
                y_pred = torch.zeros_like(y_t)
                for i, p in enumerate(range(degree, -1, -1)):
                    y_pred += coeffs[i] * (x_t ** p)
                
                mse = torch.mean((y_pred - y_t) ** 2).item()
                r2 = 1.0 - mse / y_var
                
                if (mse < 1e-4 or r2 > 0.999) and mse < best_mse:
                    # Try to simplify coefficients
                    formula = self._coeffs_to_formula(coeffs, degree)
                    if formula:
                        best_formula = formula
                        best_mse = mse
            
            return best_formula
        except Exception:
            pass
        return None
    
    def _coeffs_to_formula(self, coeffs: torch.Tensor, degree: int):
        """Convert polynomial coefficients to formula string."""
        pos_terms = []
        neg_terms = []
        
        for i, c in enumerate(coeffs.tolist()):
            power = degree - i
            if abs(c) < 1e-6:
                continue
            
            ac = abs(c)
            ac_str = format_const(ac)
            is_neg = c < 0
            
            if power == 0:
                term = ac_str
            elif power == 1:
                if abs(ac - 1.0) < 1e-6:
                    term = "x0"
                else:
                    term = f"({ac_str} * x0)"
            else:
                if abs(ac - 1.0) < 1e-6:
                    term = f"(x0 ** {power})"
                else:
                    term = f"({ac_str} * (x0 ** {power}))"
            
            if is_neg:
                neg_terms.append(term)
            else:
                pos_terms.append(term)
        
        if not pos_terms and not neg_terms:
            return None
        
        parts = []
        for j, t in enumerate(pos_terms):
            if j == 0:
                parts.append(t)
            else:
                parts.append(f" + {t}")
        for t in neg_terms:
            if not parts:
                parts.append(f"(0 - {t})")
            else:
                parts.append(f" - {t}")
        
        return f"({''.join(parts)})"
    
    def _check_power_law(self, x_t: torch.Tensor, y_t: torch.Tensor):
        """
        Check for y = A * x^B (power law).
        Uses log-log regression: log(y) = log(A) + B*log(x).
        Returns formula string if found, else None.
        """
        try:
            # Only valid for positive x and y
            if (x_t <= 0).any() or (y_t <= 0).any():
                return None
            
            log_x = torch.log(x_t)
            log_y = torch.log(y_t)
            
            ones = torch.ones_like(log_x)
            A_mat = torch.stack([log_x, ones], dim=1)
            
            solution = torch.linalg.lstsq(A_mat, log_y).solution
            B = solution[0].item()
            log_A = solution[1].item()
            A = math.exp(log_A)
            
            # Verify
            y_pred = A * (x_t ** B)
            mse = torch.mean((y_pred - y_t) ** 2).item()
            y_var = torch.var(y_t).item()
            r2 = 1.0 - mse / y_var if y_var > 1e-12 else 1.0
            
            if mse < 1e-4 or r2 > 0.999:
                # Check if B is close to integer
                B_round = round(B)
                if abs(B - B_round) < 0.01:
                    B = B_round
                
                if abs(A - 1.0) < 1e-6:
                    if isinstance(B, int) or (isinstance(B, float) and B == int(B)):
                        return f"(x0 ** {int(B)})"
                    else:
                        return f"(x0 ** {format_const(B)})"
                else:
                    a_str = format_const(A)
                    if isinstance(B, int) or (isinstance(B, float) and B == int(B)):
                        return f"({a_str} * (x0 ** {int(B)}))"
                    else:
                        return f"({a_str} * (x0 ** {format_const(B)}))"
        except Exception:
            pass
        return None
    
    def _check_geometric(self, x_t: torch.Tensor, y_t: torch.Tensor):
        """
        Check for y = A * exp(B*x) -> log(y) = log(A) + B*x
        Returns formula string if found, else None.
        """
        try:
            if (y_t <= 0).any():
                return None
            
            log_y = torch.log(y_t)
            
            ones = torch.ones_like(x_t)
            A_mat = torch.stack([x_t, ones], dim=1)
            
            solution = torch.linalg.lstsq(A_mat, log_y).solution
            B = solution[0].item()
            log_A = solution[1].item()
            
            # Predict
            y_pred = torch.exp(log_A + B * x_t)
            mse = torch.mean((y_pred - y_t) ** 2).item()
            y_var = torch.var(y_t).item()
            r2 = 1.0 - mse / y_var if y_var > 1e-12 else 1.0
            
            if mse < 1e-3 or r2 > 0.999:
                b_str = format_const(B)
                ln_a_str = format_const(log_A)
                
                if log_A >= 0:
                    inner = f"(({b_str} * x0) + {ln_a_str})"
                else:
                    inner = f"(({b_str} * x0) - {format_const(abs(log_A))})"
                
                return f"exp({inner})"
        except Exception:
            pass
        return None
    
    def _check_log_linear(self, x_t: torch.Tensor, y_t: torch.Tensor):
        """
        Check for exp(y) = mx + c -> y = log(mx + c).
        Useful when engine uses Log Transform.
        """
        try:
            exp_y = torch.exp(y_t)
            
            ones = torch.ones_like(x_t)
            A = torch.stack([x_t, ones], dim=1)
            
            solution = torch.linalg.lstsq(A, exp_y).solution
            m = solution[0].item()
            c = solution[1].item()
            
            # Predict
            pred_exp_y = m * x_t + c
            # Only valid if argument to log is positive
            if (pred_exp_y <= 0).any():
                return None
            
            y_pred = torch.log(pred_exp_y)
            mse = torch.mean((y_pred - y_t) ** 2).item()
            y_var = torch.var(y_t).item()
            r2 = 1.0 - mse / y_var if y_var > 1e-12 else 1.0
            
            if mse < 1e-4 or r2 > 0.999:
                m_str = format_const(m)
                c_str = format_const(c)
                term = f"({m_str} * x0)"
                if c >= 0:
                    inner = f"({term} + {c_str})"
                else:
                    inner = f"({term} - {format_const(abs(c))})"
                return f"log({inner})"
        except Exception:
            pass
        return None
    
    def _check_trigonometric_gpu(self, x_t: torch.Tensor, y_t: torch.Tensor):
        """
        Check for y = a*sin(b*x + c) + d using torch autograd optimization.
        Returns formula string if found, else None.
        """
        try:
            y_var = torch.var(y_t).item()
            if y_var < 1e-12:
                return None
            
            best_formula = None
            best_r2 = 0.98
            
            # Template: y = a*sin(b*x + c) + d
            # Initialize with reasonable guesses
            for fn_name in ['sin', 'cos']:
                for b_init in [1.0, 2.0, 0.5, -1.0]:
                    try:
                        # Parameters: a, b, c, d
                        params = torch.tensor([1.0, b_init, 0.0, 0.0], 
                                              device=self.device, dtype=y_t.dtype, 
                                              requires_grad=True)
                        
                        optimizer = torch.optim.Adam([params], lr=0.1)
                        
                        for _ in range(100):  # Fast optimization
                            optimizer.zero_grad()
                            a, b, c, d = params
                            
                            if fn_name == 'sin':
                                y_pred = a * torch.sin(b * x_t + c) + d
                            else:
                                y_pred = a * torch.cos(b * x_t + c) + d
                            
                            loss = torch.mean((y_pred - y_t) ** 2)
                            loss.backward()
                            optimizer.step()
                        
                        # Final evaluation
                        with torch.no_grad():
                            a, b, c, d = params
                            if fn_name == 'sin':
                                y_pred = a * torch.sin(b * x_t + c) + d
                            else:
                                y_pred = a * torch.cos(b * x_t + c) + d
                            
                            mse = torch.mean((y_pred - y_t) ** 2).item()
                            r2 = 1.0 - mse / y_var
                            
                            if r2 > best_r2:
                                formula = self._build_trig_formula(fn_name, a.item(), b.item(), c.item(), d.item())
                                if formula:
                                    best_formula = formula
                                    best_r2 = r2
                    except Exception:
                        continue
            
            return best_formula
        except Exception:
            pass
        return None
    
    def _build_trig_formula(self, func: str, a: float, b: float, c: float, d: float):
        """Build a clean trigonometric formula string."""
        a_s = self._simplify_val(a)
        b_s = self._simplify_val(b)
        c_s = self._simplify_val(c)
        d_s = self._simplify_val(d)
        
        if any(v is None for v in [a_s, b_s, c_s, d_s]):
            return None
        if abs(b_s) < 0.01:
            return None
        
        # Build inner: b*x + c
        if c_s == 0:
            if b_s == 1:
                inner = "x0"
            elif b_s == -1:
                inner = "(-x0)"
            else:
                inner = f"({format_const(b_s)} * x0)"
        else:
            if b_s == 1:
                inner = f"(x0 + {format_const(c_s)})"
            else:
                inner = f"({format_const(b_s)} * x0 + {format_const(c_s)})"
        
        # Build: a*func(inner)
        if a_s == 1:
            core = f"{func}({inner})"
        elif a_s == -1:
            core = f"(-{func}({inner}))"
        else:
            core = f"({format_const(a_s)} * {func}({inner}))"
        
        # Add offset d
        if d_s == 0:
            return core
        else:
            if d_s > 0:
                return f"({core} + {format_const(d_s)})"
            else:
                return f"({core} - {format_const(abs(d_s))})"
    
    def _check_composite_gpu(self, x_t: torch.Tensor, y_t: torch.Tensor):
        """
        Check for composite patterns: y = x^n * f(x)
        where f(x) is sin(x), cos(x), exp(-x), etc.
        Uses torch autograd optimization.
        """
        try:
            y_var = torch.var(y_t).item()
            if y_var < 1e-12:
                return None
            
            best_formula = None
            best_r2 = 0.98
            
            # Templates to try
            templates = [
                # (name, init_fn, pred_fn)
                ('x^n*sin', lambda p: [1.0, 1.0], 
                 lambda p, x: p[0] * (x ** p[1]) * torch.sin(x)),
                ('x^n*cos', lambda p: [1.0, 1.0], 
                 lambda p, x: p[0] * (x ** p[1]) * torch.cos(x)),
                ('exp(-x)*sin', lambda p: [1.0], 
                 lambda p, x: p[0] * torch.exp(-x) * torch.sin(x)),
                ('exp(-x)*cos', lambda p: [1.0], 
                 lambda p, x: p[0] * torch.exp(-x) * torch.cos(x)),
                ('exp(-x)*sin*cos', lambda p: [1.0], 
                 lambda p, x: p[0] * torch.exp(-x) * torch.sin(x) * torch.cos(x)),
            ]
            
            for name, init_fn, pred_fn in templates:
                try:
                    # Initialize parameters
                    init_vals = init_fn(None)
                    params = torch.tensor(init_vals, device=self.device, dtype=y_t.dtype, requires_grad=True)
                    
                    optimizer = torch.optim.Adam([params], lr=0.1)
                    
                    for _ in range(100):
                        optimizer.zero_grad()
                        y_pred = pred_fn(params, x_t)
                        loss = torch.mean((y_pred - y_t) ** 2)
                        loss.backward()
                        optimizer.step()
                    
                    # Final evaluation
                    with torch.no_grad():
                        y_pred = pred_fn(params, x_t)
                        mse = torch.mean((y_pred - y_t) ** 2).item()
                        r2 = 1.0 - mse / y_var
                        
                        if r2 > best_r2:
                            formula = self._build_composite_formula(name, params.tolist())
                            if formula:
                                best_formula = formula
                                best_r2 = r2
                except Exception:
                    continue
            
            return best_formula
        except Exception:
            pass
        return None
    
    def _build_composite_formula(self, name: str, params: list):
        """Build composite formula string from optimized parameters."""
        if 'x^n*sin' in name:
            a, n = params
            a_s = self._simplify_val(a)
            n_s = self._simplify_val(n, tol=0.1)
            if a_s is None or n_s is None or n_s < 0.5:
                return None
            n_int = int(round(n_s))
            if abs(n_s - n_int) > 0.15:
                return None
            if a_s == 1.0:
                return f"((x0 ** {n_int}) * sin(x0))"
            else:
                return f"({format_const(a_s)} * (x0 ** {n_int}) * sin(x0))"
        
        elif 'x^n*cos' in name:
            a, n = params
            a_s = self._simplify_val(a)
            n_s = self._simplify_val(n, tol=0.1)
            if a_s is None or n_s is None or n_s < 0.5:
                return None
            n_int = int(round(n_s))
            if abs(n_s - n_int) > 0.15:
                return None
            if a_s == 1.0:
                return f"((x0 ** {n_int}) * cos(x0))"
            else:
                return f"({format_const(a_s)} * (x0 ** {n_int}) * cos(x0))"
        
        elif 'exp(-x)*sin*cos' in name:
            a = params[0]
            a_s = self._simplify_val(a)
            if a_s is None:
                return None
            if a_s == 1.0:
                return f"(exp(-x0) * sin(x0) * cos(x0))"
            else:
                return f"({format_const(a_s)} * exp(-x0) * sin(x0) * cos(x0))"
        
        elif 'exp(-x)*sin' in name:
            a = params[0]
            a_s = self._simplify_val(a)
            if a_s is None:
                return None
            if a_s == 1.0:
                return f"(exp(-x0) * sin(x0))"
            else:
                return f"({format_const(a_s)} * exp(-x0) * sin(x0))"
        
        elif 'exp(-x)*cos' in name:
            a = params[0]
            a_s = self._simplify_val(a)
            if a_s is None:
                return None
            if a_s == 1.0:
                return f"(exp(-x0) * cos(x0))"
            else:
                return f"({format_const(a_s)} * exp(-x0) * cos(x0))"
        
        return None

    def _check_lgamma_pattern(self, x_t: torch.Tensor, y_t: torch.Tensor):
        """
        CONVERGENCE FIX: Check for lgamma-based patterns.
        
        Fits templates:
          1. y = a*lgamma(b*x+c) + d*x + e   (Stirling-like)
          2. y = lgamma(x) - a*x + b           (N-Queens dominant structure)
          3. y = a*lgamma(x+b) + c*sqrt(x) + d (lgamma + correction)
        
        Uses torch autograd optimization.
        Returns formula string if found, else None.
        """
        try:
            y_var = torch.var(y_t).item()
            if y_var < 1e-12:
                return None
            
            best_formula = None
            best_r2 = 0.98  # High threshold: only report if good fit
            
            templates = [
                # Template 1: y = a*lgamma(b*x+c) + d*x + e
                {
                    'name': 'lgamma_linear',
                    'init': [1.0, 1.0, 0.5, -1.0, 0.0],
                    'fn': lambda p, x: p[0] * torch.lgamma(torch.clamp(p[1] * x + p[2], min=0.5)) + p[3] * x + p[4],
                    'build': lambda p: self._build_lgamma_formula(p, 'lgamma_linear'),
                },
                # Template 2: y = lgamma(x+a) - b*x + c (N-Queens)
                {
                    'name': 'lgamma_nqueens',
                    'init': [0.0, 1.0, 0.0],
                    'fn': lambda p, x: torch.lgamma(torch.clamp(x + p[0], min=0.5)) - p[1] * x + p[2],
                    'build': lambda p: self._build_lgamma_formula(p, 'lgamma_nqueens'),
                },
                # Template 3: y = a*lgamma(x+b) + c*sqrt(x) + d
                {
                    'name': 'lgamma_sqrt',
                    'init': [1.0, 0.5, 0.0, 0.0],
                    'fn': lambda p, x: p[0] * torch.lgamma(torch.clamp(x + p[1], min=0.5)) + p[2] * torch.sqrt(torch.clamp(x, min=1e-6)) + p[3],
                    'build': lambda p: self._build_lgamma_formula(p, 'lgamma_sqrt'),
                },
            ]
            
            for tmpl in templates:
                try:
                    params = torch.tensor(tmpl['init'], 
                                         device=self.device, dtype=torch.float64,
                                         requires_grad=True)
                    optimizer = torch.optim.Adam([params], lr=0.05)
                    
                    for _ in range(200):  # More steps for lgamma convergence
                        optimizer.zero_grad()
                        y_pred = tmpl['fn'](params, x_t)
                        loss = torch.mean((y_pred - y_t) ** 2)
                        if torch.isnan(loss) or torch.isinf(loss):
                            break
                        loss.backward()
                        optimizer.step()
                    
                    with torch.no_grad():
                        y_pred = tmpl['fn'](params, x_t)
                        mse = torch.mean((y_pred - y_t) ** 2).item()
                        r2 = 1.0 - mse / y_var
                        
                        if r2 > best_r2:
                            formula = tmpl['build'](params.tolist())
                            if formula:
                                best_formula = formula
                                best_r2 = r2
                except Exception:
                    continue
            
            return best_formula
        except Exception:
            pass
        return None
    
    def _build_lgamma_formula(self, params, template_name):
        """Build lgamma formula string from optimized parameters."""
        try:
            if template_name == 'lgamma_linear':
                a, b, c, d, e = params
                # y = a*lgamma(b*x+c) + d*x + e
                parts = []
                b_s, c_s = format_const(b), format_const(c)
                inner = f"({b_s} * x0 + {c_s})" if abs(c) > 0.01 else f"({b_s} * x0)"
                
                if abs(a - 1.0) < 0.01:
                    parts.append(f"lgamma({inner})")
                else:
                    parts.append(f"({format_const(a)} * lgamma({inner}))")
                
                if abs(d) > 0.01:
                    if d > 0:
                        parts.append(f"({format_const(d)} * x0)")
                    else:
                        parts.append(f"({format_const(d)} * x0)")
                
                if abs(e) > 0.01:
                    parts.append(format_const(e))
                
                return "(" + " + ".join(parts) + ")" if parts else None
                
            elif template_name == 'lgamma_nqueens':
                a, b, c = params
                # y = lgamma(x+a) - b*x + c
                if abs(a) > 0.01:
                    inner = f"(x0 + {format_const(a)})"
                else:
                    inner = "x0"
                
                parts = [f"lgamma({inner})"]
                if abs(b) > 0.01:
                    parts.append(f"({format_const(-b)} * x0)")
                if abs(c) > 0.01:
                    parts.append(format_const(c))
                
                return "(" + " + ".join(parts) + ")" if parts else None
                
            elif template_name == 'lgamma_sqrt':
                a, b, c, d = params
                # y = a*lgamma(x+b) + c*sqrt(x) + d
                if abs(b) > 0.01:
                    inner = f"(x0 + {format_const(b)})"
                else:
                    inner = "x0"
                
                parts = []
                if abs(a - 1.0) < 0.01:
                    parts.append(f"lgamma({inner})")
                else:
                    parts.append(f"({format_const(a)} * lgamma({inner}))")
                
                if abs(c) > 0.01:
                    parts.append(f"({format_const(c)} * sqrt(x0))")
                if abs(d) > 0.01:
                    parts.append(format_const(d))
                
                return "(" + " + ".join(parts) + ")" if parts else None
        except Exception:
            pass
        return None
    
    def run(self, x_data, y_data):
        """
        Run all pattern checks.
        x_data, y_data: torch.Tensor or array-like.
        Returns: formula string or None.
        """
        try:
            # Convert to torch tensors if needed
            if isinstance(x_data, torch.Tensor):
                x_t = x_data.clone().to(self.device).to(torch.float64)
            else:
                x_t = torch.tensor(x_data, device=self.device, dtype=torch.float64)
            
            if isinstance(y_data, torch.Tensor):
                y_t = y_data.flatten().to(self.device).to(torch.float64)
            else:
                y_t = torch.tensor(y_data, device=self.device, dtype=torch.float64).flatten()
            
            # Safety guards: check length and NaNs/Infs to prevent downstream crashes
            if x_t.numel() < 2 or y_t.numel() < 2:
                return None
            if torch.isnan(x_t).any() or torch.isnan(y_t).any() or torch.isinf(x_t).any() or torch.isinf(y_t).any():
                return None
            
            # FIX: Extract primary variable correctly for multi-variable inputs
            if x_t.dim() > 1:
                if x_t.shape[0] == y_t.shape[0]:     # [Samples, Vars]
                    x_t = x_t[:, 0] if x_t.shape[1] > 1 else x_t.flatten()
                elif x_t.shape[1] == y_t.shape[0]:   # [Vars, Samples]
                    x_t = x_t[0, :] if x_t.shape[0] > 1 else x_t.flatten()
                else:
                    return None
            else:
                x_t = x_t.flatten()
            
            # Check 1: Linear (y = mx+c)
            res = self._check_linear(x_t, y_t)
            if res:
                print(f"[Sniper GPU] Detected Linear: {res}")
                return res
            
            # Check 2: Log-Linear (y = log(mx+c))
            res = self._check_log_linear(x_t, y_t)
            if res:
                print(f"[Sniper GPU] Detected Log-Linear: {res}")
                return res
            
            # Check 3: Geometric (y = A exp(Bx))
            res = self._check_geometric(x_t, y_t)
            if res:
                print(f"[Sniper GPU] Detected Geometric: {res}")
                return res
            
            # Check 4: Power Law (y = A * x^B)
            res = self._check_power_law(x_t, y_t)
            if res:
                print(f"[Sniper GPU] Detected Power Law: {res}")
                return res
            
            # Check 5: Trigonometric (y = a*sin(b*x+c)+d)
            res = self._check_trigonometric_gpu(x_t, y_t)
            if res:
                print(f"[Sniper GPU] Detected Trig: {res}")
                return res
            
            # Check 6: Composite (y = x^n * f(x))
            res = self._check_composite_gpu(x_t, y_t)
            if res:
                print(f"[Sniper GPU] Detected Composite: {res}")
                return res
            
            # Check 7: lgamma patterns (y = a*lgamma(bx+c) + dx + e)
            # CONVERGENCE FIX: Essential for N-Queens and combinatorial sequences
            res = self._check_lgamma_pattern(x_t, y_t)
            if res:
                print(f"[Sniper GPU] Detected lgamma pattern: {res}")
                return res
            
            # Check 8: Polynomial (y = ax^n + ...)
            res = self._check_polynomial(x_t, y_t)
            if res:
                print(f"[Sniper GPU] Detected Polynomial: {res}")
                return res
            
        except Exception as e:
            pass
        return None