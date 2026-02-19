
import torch
import numpy as np
from .formatting import format_const

class Sniper:
    """
    'The Sniper': Special Pattern Detection Unit.
    Quickly identifies simple patterns (Linear, Polynomial, Geometric, Power Law) before evolution begins.
    """
    def __init__(self, device):
        self.device = device
    
    def _simplify_coeffs(self, coeffs, x_np, y_np, y_var):
        """Try to round polynomial coefficients to simpler values.
        Returns simplified coeffs if R² stays good, else original."""
        simple = np.copy(coeffs)
        for i in range(len(simple)):
            c = simple[i]
            # Try rounding small coefficients to 0
            if abs(c) < 0.1:
                simple[i] = 0.0
            # Try rounding to nearest integer if close
            elif abs(c - round(c)) < 0.1:
                simple[i] = round(c)
            # Try rounding to nearest 0.5
            elif abs(c - round(c * 2) / 2) < 0.05:
                simple[i] = round(c * 2) / 2
        
        # Check if simplified version is still good
        y_pred_simple = np.polyval(simple, x_np)
        mse_simple = np.mean((y_np - y_pred_simple) ** 2)
        r2_simple = 1.0 - mse_simple / y_var if y_var > 1e-12 else 0.0
        
        if r2_simple > 0.99:
            return simple
        return coeffs
    
    def _coeffs_to_formula(self, coeffs, degree):
        """Convert polynomial coefficients to formula string."""
        pos_terms = []
        neg_terms = []
        for i, c in enumerate(coeffs):
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
    
    def check_trigonometric(self, x_t, y_t):
        """
        Check for y = a*sin(b*x + c) + d or y = a*cos(b*x + c) + d.
        Returns formula string if found, else None.
        """
        try:
            from scipy.optimize import curve_fit
            x_np = x_t.cpu().numpy().astype(np.float64).flatten()
            y_np = y_t.cpu().numpy().astype(np.float64).flatten()
            y_var = np.var(y_np)
            if y_var < 1e-12:
                return None

            def sin_model(x, a, b, c, d):
                return a * np.sin(b * x + c) + d

            def cos_model(x, a, b, c, d):
                return a * np.cos(b * x + c) + d

            best_formula = None
            best_r2 = 0.0

            for func_name, model in [('sin', sin_model), ('cos', cos_model)]:
                for p0 in [(1, 1, 0, 0), (-1, 1, 0, 0), (1, 2, 0, 0), (1, 0.5, 0, 0)]:
                    try:
                        popt, _ = curve_fit(model, x_np, y_np, p0=p0, maxfev=5000)
                        a, b, c, d = popt
                        y_pred = model(x_np, *popt)
                        ss_res = np.sum((y_np - y_pred) ** 2)
                        ss_tot = np.sum((y_np - np.mean(y_np)) ** 2)
                        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

                        if r2 > 0.99 and r2 > best_r2:
                            formula = self._build_trig_formula(func_name, a, b, c, d)
                            if formula:
                                best_formula = formula
                                best_r2 = r2
                    except Exception:
                        continue

            return best_formula
        except Exception:
            return None

    def _build_trig_formula(self, func, a, b, c, d):
        """Build a clean trigonometric formula string with simplified coefficients."""
        def simplify(val, tol=0.08):
            if abs(val) < tol:
                return 0.0
            rounded = round(val)
            if abs(val - rounded) < tol:
                return float(rounded)
            half = round(val * 2) / 2
            if abs(val - half) < 0.05:
                return half
            return None  # Can't simplify

        a_s = simplify(a)
        b_s = simplify(b)
        c_s = simplify(c)
        d_s = simplify(d)

        if any(v is None for v in [a_s, b_s, c_s, d_s]):
            return None
        if abs(b_s) < 0.01:
            return None  # Degenerate

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
            formula = core
        else:
            formula = f"({core} + {format_const(d_s)})"

        return formula

    def check_polynomial(self, x_t, y_t):
        """
        Check for polynomial y = a_n*x^n + ... + a_1*x + a_0 for degrees 2, 3, 4.
        Returns formula string if found, else None.
        """
        try:
            x_np = x_t.cpu().numpy().astype(np.float64)
            y_np = y_t.cpu().numpy().astype(np.float64)
            
            best_formula = None
            best_mse = float('inf')
            y_var = np.var(y_np)
            
            for degree in [2, 3, 4, 5, 6]:
                try:
                    coeffs = np.polyfit(x_np, y_np, degree)
                    y_pred = np.polyval(coeffs, x_np)
                    mse = np.mean((y_np - y_pred) ** 2)
                    
                    # Accept if MSE is very low OR R² is very high (handles noisy data)
                    r2 = 1.0 - mse / y_var if y_var > 1e-12 else 0.0
                    if (mse < 1e-4 or r2 > 0.999) and mse < best_mse:
                        # Try to simplify coefficients (rounds noisy coeffs to clean values)
                        coeffs = self._simplify_coeffs(coeffs, x_np, y_np, y_var)
                        
                        formula = self._coeffs_to_formula(coeffs, degree)
                        if formula:
                            best_formula = formula
                            best_mse = mse
                except np.linalg.LinAlgError:
                    continue
            
            # Prefer lowest degree that fits well
            # Re-check: if degree-2 was accepted but degree-3/4 was chosen,
            # the best_mse logic already handles this via "mse < best_mse"
            return best_formula
        except Exception:
            pass
        return None
    
    def check_power_law(self, x_t, y_t):
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
            A = float(np.exp(log_A))
            
            # Verify
            y_pred = A * (x_t ** B)
            mse = torch.mean((y_pred - y_t) ** 2)
            
            if mse < 1e-6:
                # Check if B is close to integer (cleaner formula)
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
                return f"({m_str} * x0 + {c_str})" if c >= 0 else f"({m_str} * x0 - {format_const(abs(c))})"
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
                
                inner = f"({b_str} * x0)"
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
                term = f"({m_str} * x0)"
                if c >= 0:
                    inner = f"({term} + {c_str})"
                else:
                    inner = f"({term} - {format_const(abs(c))})"
                return f"log({inner})"
        except:
            pass
        return None

    def check_log_additive(self, x_t, y_t):
        """
        Check for y = a*log(x+b) + c using scipy curve_fit.
        Also tries y = log(x+a) + log(x²+b) for Nguyen-7 type problems.
        Returns formula string if found, else None.
        """
        try:
            from scipy.optimize import curve_fit
            x_np = x_t.cpu().numpy().astype(np.float64)
            y_np = y_t.cpu().numpy().astype(np.float64)
            
            # Only if x has values that allow log
            if np.any(x_np <= -1):
                return None
            
            # Template 1: y = a*log(x+b) + c
            def log_func(x, a, b, c):
                return a * np.log(np.maximum(x + b, 1e-15)) + c
            
            try:
                popt, _ = curve_fit(log_func, x_np, y_np, p0=[1.0, 1.0, 0.0], maxfev=5000)
                a, b, c = popt
                y_pred = log_func(x_np, a, b, c)
                mse = np.mean((y_np - y_pred) ** 2)
                if mse < 1e-6:
                    b_str = format_const(b)
                    if abs(a - 1.0) < 1e-4:
                        inner = f"log((x0 + {b_str}))"
                    else:
                        inner = f"({format_const(a)} * log((x0 + {b_str})))"
                    if abs(c) > 1e-6:
                        if c >= 0:
                            inner = f"({inner} + {format_const(c)})"
                        else:
                            inner = f"({inner} - {format_const(abs(c))})"
                    return inner
            except Exception:
                pass
            
            # Template 2: y = log(x+a) + log(x²+b)  (Nguyen-7 type)
            def log_sum_func(x, a, b):
                return np.log(np.maximum(x + a, 1e-15)) + np.log(np.maximum(x**2 + b, 1e-15))
            
            try:
                popt, _ = curve_fit(log_sum_func, x_np, y_np, p0=[1.0, 1.0], maxfev=5000)
                a, b = popt
                y_pred = log_sum_func(x_np, a, b)
                mse = np.mean((y_np - y_pred) ** 2)
                if mse < 1e-6:
                    a_str = format_const(a)
                    b_str = format_const(b)
                    return f"(log((x0 + {a_str})) + log(((x0 ** 2) + {b_str})))"
            except Exception:
                pass
            
            # Template 3: y = a*log(x) + b  (simpler)
            if np.all(x_np > 0):
                def log_simple(x, a, b):
                    return a * np.log(x) + b
                try:
                    popt, _ = curve_fit(log_simple, x_np, y_np, p0=[1.0, 0.0], maxfev=3000)
                    a, b = popt
                    y_pred = log_simple(x_np, a, b)
                    mse = np.mean((y_np - y_pred) ** 2)
                    if mse < 1e-6:
                        if abs(a - 1.0) < 1e-4:
                            inner = "log(x0)"
                        else:
                            inner = f"({format_const(a)} * log(x0))"
                        if abs(b) > 1e-6:
                            if b >= 0:
                                inner = f"({inner} + {format_const(b)})"
                            else:
                                inner = f"({inner} - {format_const(abs(b))})"
                        return inner
                except Exception:
                    pass
        except ImportError:
            pass
        except Exception:
            pass
        return None

    def check_composite(self, x_t, y_t):
        """
        Check for composite patterns: y = x^n * f(x)
        where f(x) is sin(x), cos(x), exp(-x), sin(x)*cos(x), exp(-x)*sin(x), etc.
        Detects: x*sin(x), x²*cos(x), x³*exp(-x)*cos(x)*sin(x) (Keijzer-4), etc.
        """
        try:
            from scipy.optimize import curve_fit
            x_np = x_t.cpu().numpy().astype(np.float64).flatten()
            y_np = y_t.cpu().numpy().astype(np.float64).flatten()
            
            # Skip if x contains zeros (division by x^n would fail)
            x_abs_min = np.min(np.abs(x_np))
            
            # Define composite templates: (name, model_func, p0, formula_builder)
            templates = []
            
            # Template: y = a * x^n * sin(x) for n=1,2,3
            for n in [1, 2, 3]:
                def make_model_sin(n=n):
                    def model(x, a):
                        return a * (x**n) * np.sin(x)
                    return model
                templates.append((f'x^{n}*sin', make_model_sin(), [1.0], n, 'sin'))
            
            # Template: y = a * x^n * cos(x) for n=1,2,3
            for n in [1, 2, 3]:
                def make_model_cos(n=n):
                    def model(x, a):
                        return a * (x**n) * np.cos(x)
                    return model
                templates.append((f'x^{n}*cos', make_model_cos(), [1.0], n, 'cos'))
            
            # Template: y = a * x^n * exp(b*x) for n=0,1,2,3
            # BUG-9 FIX: Agregar proteccion contra overflow en exp()
            # exp(>700) = inf, causando warnings y deteccion fallida
            for n in [0, 1, 2, 3]:
                def make_model_exp(n=n):
                    def model(x, a, b):
                        exp_arg = np.clip(b * x, -700, 700)  # Previene overflow
                        return a * (x**n) * np.exp(exp_arg)
                    return model
                templates.append((f'x^{n}*exp', make_model_exp(), [1.0, -1.0], n, 'exp'))
            
            # Template: y = a * x^n * exp(b*x) * sin(x) for n=0,1,2,3
            for n in [0, 1, 2, 3]:
                def make_model_exp_sin(n=n):
                    def model(x, a, b):
                        exp_arg = np.clip(b * x, -700, 700)  # BUG-9 FIX
                        return a * (x**n) * np.exp(exp_arg) * np.sin(x)
                    return model
                templates.append((f'x^{n}*exp*sin', make_model_exp_sin(), [1.0, -1.0], n, 'exp_sin'))
            
            # Template: y = a * x^n * exp(b*x) * cos(x) for n=0,1,2,3
            for n in [0, 1, 2, 3]:
                def make_model_exp_cos(n=n):
                    def model(x, a, b):
                        exp_arg = np.clip(b * x, -700, 700)  # BUG-9 FIX
                        return a * (x**n) * np.exp(exp_arg) * np.cos(x)
                    return model
                templates.append((f'x^{n}*exp*cos', make_model_exp_cos(), [1.0, -1.0], n, 'exp_cos'))
            
            # Template: y = a * x^n * exp(b*x) * sin(x) * cos(x) for n=0,1,2,3
            for n in [0, 1, 2, 3]:
                def make_model_exp_sincos(n=n):
                    def model(x, a, b):
                        exp_arg = np.clip(b * x, -700, 700)  # BUG-9 FIX
                        return a * (x**n) * np.exp(exp_arg) * np.sin(x) * np.cos(x)
                    return model
                templates.append((f'x^{n}*exp*sin*cos', make_model_exp_sincos(), [1.0, -1.0], n, 'exp_sin_cos'))
            
            # Template: y = a * sin(b*x^2 + c) for sin(x²) pattern
            def sin_xsq_model(x, a, b, c):
                return a * np.sin(b * x**2 + c)
            templates.append(('sin(x²)', sin_xsq_model, [1.0, 1.0, 0.0], 0, 'sin_xsq'))
            
            best_formula = None
            best_r2 = 0.98  # Minimum R² threshold
            y_var = np.var(y_np)
            if y_var < 1e-12:
                return None
            
            for name, model, p0, n_pow, ftype in templates:
                try:
                    popt, _ = curve_fit(model, x_np, y_np, p0=p0, maxfev=3000)
                    y_pred = model(x_np, *popt)
                    if not np.all(np.isfinite(y_pred)):
                        continue
                    ss_res = np.sum((y_np - y_pred) ** 2)
                    ss_tot = np.sum((y_np - np.mean(y_np)) ** 2)
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
                    
                    if r2 > best_r2:
                        formula = self._build_composite_formula(ftype, n_pow, popt)
                        if formula:
                            best_formula = formula
                            best_r2 = r2
                except Exception:
                    continue
            
            return best_formula
        except Exception:
            return None
    
    def _build_composite_formula(self, ftype, n_pow, popt):
        """Build a composite formula string from curve_fit results."""
        def simplify_val(val, tol=0.05):
            if abs(val) < tol:
                return 0.0
            rounded = round(val)
            if abs(val - rounded) < tol:
                return float(rounded)
            return None
        
        # Build x^n part
        if n_pow == 0:
            x_part = ""
        elif n_pow == 1:
            x_part = "x0"
        else:
            x_part = f"(x0 ** {n_pow})"
        
        if ftype == 'sin':
            a = popt[0]
            a_s = simplify_val(a)
            if a_s is None: return None
            trig_part = "sin(x0)"
            parts = [p for p in [x_part, trig_part] if p]
            core = " * ".join(parts) if len(parts) > 1 else parts[0]
            if a_s == 1.0:
                return core
            elif a_s == -1.0:
                return f"(-{core})"
            else:
                return f"({format_const(a_s)} * {core})"
        
        elif ftype == 'cos':
            a = popt[0]
            a_s = simplify_val(a)
            if a_s is None: return None
            trig_part = "cos(x0)"
            parts = [p for p in [x_part, trig_part] if p]
            core = " * ".join(parts) if len(parts) > 1 else parts[0]
            if a_s == 1.0:
                return core
            elif a_s == -1.0:
                return f"(-{core})"
            else:
                return f"({format_const(a_s)} * {core})"
        
        elif ftype == 'exp':
            a, b = popt
            a_s = simplify_val(a)
            b_s = simplify_val(b)
            if a_s is None or b_s is None: return None
            if b_s == 0: return None
            if b_s == -1:
                exp_part = "exp(-x0)"
            elif b_s == 1:
                exp_part = "exp(x0)"
            else:
                exp_part = f"exp({format_const(b_s)} * x0)"
            parts = [p for p in [x_part, exp_part] if p]
            core = " * ".join(parts) if len(parts) > 1 else parts[0]
            if a_s == 1.0:
                return core
            elif a_s == -1.0:
                return f"(-{core})"
            else:
                return f"({format_const(a_s)} * {core})"
        
        elif ftype == 'exp_sin':
            a, b = popt
            a_s = simplify_val(a)
            b_s = simplify_val(b)
            if a_s is None or b_s is None: return None
            if b_s == -1:
                exp_part = "exp(-x0)"
            elif b_s == 1:
                exp_part = "exp(x0)"
            else:
                exp_part = f"exp({format_const(b_s)} * x0)"
            parts = [p for p in [x_part, exp_part, "sin(x0)"] if p]
            core = " * ".join(parts) if len(parts) > 1 else parts[0]
            if a_s == 1.0:
                return core
            elif a_s == -1.0:
                return f"(-{core})"
            else:
                return f"({format_const(a_s)} * {core})"
        
        elif ftype == 'exp_cos':
            a, b = popt
            a_s = simplify_val(a)
            b_s = simplify_val(b)
            if a_s is None or b_s is None: return None
            if b_s == -1:
                exp_part = "exp(-x0)"
            elif b_s == 1:
                exp_part = "exp(x0)"
            else:
                exp_part = f"exp({format_const(b_s)} * x0)"
            parts = [p for p in [x_part, exp_part, "cos(x0)"] if p]
            core = " * ".join(parts) if len(parts) > 1 else parts[0]
            if a_s == 1.0:
                return core
            elif a_s == -1.0:
                return f"(-{core})"
            else:
                return f"({format_const(a_s)} * {core})"
        
        elif ftype == 'exp_sin_cos':
            a, b = popt
            a_s = simplify_val(a)
            b_s = simplify_val(b)
            if a_s is None or b_s is None: return None
            if b_s == -1:
                exp_part = "exp(-x0)"
            elif b_s == 1:
                exp_part = "exp(x0)"
            else:
                exp_part = f"exp({format_const(b_s)} * x0)"
            parts = [p for p in [x_part, exp_part, "sin(x0)", "cos(x0)"] if p]
            core = " * ".join(parts) if len(parts) > 1 else parts[0]
            if a_s == 1.0:
                return core
            elif a_s == -1.0:
                return f"(-{core})"
            else:
                return f"({format_const(a_s)} * {core})"
        
        elif ftype == 'sin_xsq':
            a, b, c = popt
            a_s = simplify_val(a)
            b_s = simplify_val(b)
            c_s = simplify_val(c, tol=0.08)
            if a_s is None or b_s is None or c_s is None: return None
            if b_s == 0: return None
            if b_s == 1 and c_s == 0:
                inner = "sin((x0 ** 2))"
            elif c_s == 0:
                inner = f"sin(({format_const(b_s)} * (x0 ** 2)))"
            else:
                inner = f"sin(({format_const(b_s)} * (x0 ** 2)) + {format_const(c_s)})"
            if a_s == 1.0:
                return inner
            elif a_s == -1.0:
                return f"(-{inner})"
            else:
                return f"({format_const(a_s)} * {inner})"
        
        return None

    def run(self, x_data, y_data):
        """
        Run all checks.
        x_data, y_data: CPU lists or arrays.
        """
        try:
            # Optimize: If inputs are already tensors, use them directly
            if isinstance(x_data, torch.Tensor):
                x_t = x_data.flatten()
            else:
                x_t = torch.tensor(x_data, device=self.device, dtype=torch.float32).flatten()
                
            if isinstance(y_data, torch.Tensor):
                y_t = y_data.flatten()
            else:
                y_t = torch.tensor(y_data, device=self.device, dtype=torch.float32).flatten()
            
            # Check 1: Linear (y = mx+c)
            res = self.check_linear(x_t, y_t)
            if res: 
                print(f"[The Sniper] Detected Linear Pattern: {res}")
                return res
            
            # Check 2: Log Additive (y = a*log(x+b)+c or y = log(x+a)+log(x²+b))
            # Check BEFORE polynomial since log data is well-approximated by polynomials
            res = self.check_log_additive(x_t, y_t)
            if res:
                print(f"[The Sniper] Detected Log Pattern: {res}")
                return res
            
            # Check 3: Trigonometric (y = a*sin(b*x+c)+d or a*cos(b*x+c)+d)
            res = self.check_trigonometric(x_t, y_t)
            if res:
                print(f"[The Sniper] Detected Trig Pattern: {res}")
                return res
            
            # Check 4: Composite (y = x^n * f(x) where f is sin/cos/exp or product)
            res = self.check_composite(x_t, y_t)
            if res:
                print(f"[The Sniper] Detected Composite Pattern: {res}")
                return res
            
            # Check 5: Polynomial (y = ax^n + bx^(n-1) + ... + c) for n=2,3,4,5,6
            res = self.check_polynomial(x_t, y_t)
            if res:
                print(f"[The Sniper] Detected Polynomial Pattern: {res}")
                return res
            
            # Check 6: Geometric (y = A exp(Bx))
            res = self.check_geometric(x_t, y_t)
            if res:
                print(f"[The Sniper] Detected Geometric Pattern: {res}")
                return res
                
            # Check 7: Power Law (y = A * x^B)
            res = self.check_power_law(x_t, y_t)
            if res:
                print(f"[The Sniper] Detected Power Law Pattern: {res}")
                return res
                
            # Check 8: Log-Linear (y = log(mx+c)) / Linear Raw Data
            res = self.check_log_linear(x_t, y_t)
            if res:
                print(f"[The Sniper] Detected Log-Linear Pattern: {res}")
                return res
            
        except Exception as e:
            # print(f"[The Sniper] Failed: {e}")
            pass
        return None
