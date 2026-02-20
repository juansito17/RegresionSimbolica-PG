"""
╔══════════════════════════════════════════════════════════════════╗
║       Verify RMSE — Comparación 1:1 GPU vs CPU                  ║
║                                                                 ║
║  Evalúa una fórmula por 3 vías:                                 ║
║    Vía 1: GPU evaluate_batch (CUDA kernels protegidos)          ║
║    Vía 2: CPU con semántica protegida idéntica a CUDA           ║
║    Vía 3: CPU estricta (grammar.py — sin protección)            ║
║                                                                 ║
║  Uso:                                                           ║
║    python verify_rmse.py "sin(x0) + x0^2" -r -3,3              ║
║    python verify_rmse.py "log(x0) + sqrt(x0)" -r 0.1,5         ║
║    python verify_rmse.py "x0^3 - 3.3628" --xy 1,2,3 4,5,6      ║
║    python verify_rmse.py "x0^2 + x1" --vars 2 -r -2,2          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sys, os
import traceback

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import argparse
import time
import re
from scipy.special import gamma as scipy_gamma

# ═══════════════════════════════════════════════════════════════════
#  VÍA 2: CPU con semántica protegida idéntica a CUDA
#  Replica EXACTAMENTE lo que hacen los safe_* de rpn_kernels.cu
# ═══════════════════════════════════════════════════════════════════

def _safe_div(a, b):
    """GPU: if |b| < 1e-9 return a; else a/b"""
    mask = np.abs(b) < 1e-9
    result = np.where(mask, a, np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=~mask))
    return result

def _safe_mod(a, b):
    """GPU: if |b| < 1e-9 return 0; else fmod with Python-style sign correction"""
    mask = np.abs(b) < 1e-9
    r = np.fmod(a, b + mask.astype(float) * 1.0)  # avoid fmod-by-zero
    # Sign correction: match CUDA behavior
    sign_fix = ((b > 0) & (r < 0)) | ((b < 0) & (r > 0))
    r = np.where(sign_fix, r + b, r)
    return np.where(mask, 0.0, r)

def _safe_log(a):
    """GPU: log(|a| + 1e-9) — never NaN"""
    return np.log(np.abs(a) + 1e-9)

def _safe_exp(a):
    """GPU: clamp to [-80, 80] then exp"""
    x = np.clip(a, -80.0, 80.0)
    return np.exp(x)

def _safe_sqrt(a):
    """GPU: sqrt(|a|) — never NaN"""
    return np.sqrt(np.abs(a))

def _safe_pow(a, b):
    """GPU protected pow — complex negative-base handling, overflow caps"""
    result = np.zeros_like(a, dtype=np.float64)
    
    for i in range(len(a)):
        ai, bi = float(a[i]), float(b[i])
        
        # NaN check
        if np.isnan(ai) or np.isnan(bi):
            result[i] = 0.0
            continue
        
        # (0, 0) -> 1.0
        if abs(ai) < 1e-10 and abs(bi) < 1e-10:
            result[i] = 1.0
            continue
        
        # Protected negative-base handling
        if ai < 0.0:
            ib = round(bi)
            if abs(bi - ib) > 1e-3:
                ai = abs(ai)  # non-integer exponent → use |a|
            else:
                bi = ib  # integer-ish exponent → round it
        
        # Overflow safety
        if abs(ai) > 1.0 and bi > 80.0:
            bi = 80.0
        if abs(ai) > 100.0 and bi > 10.0:
            bi = 10.0
        
        try:
            res = ai ** bi
            if np.isnan(res) or np.isinf(res):
                result[i] = 0.0
            else:
                result[i] = res
        except:
            result[i] = 0.0
    
    return result

def _safe_asin(a):
    """GPU: clamp to [-1, 1] then asin"""
    return np.arcsin(np.clip(a, -1.0, 1.0))

def _safe_acos(a):
    """GPU: clamp to [-1, 1] then acos"""
    return np.arccos(np.clip(a, -1.0, 1.0))

def _safe_tgamma(a):
    """GPU: 0 at poles, 0 for NaN/Inf results"""
    is_pole = (a <= 0.0) & (np.floor(a) == a)
    res = scipy_gamma(a)
    res = np.where(np.isnan(res) | np.isinf(res), 0.0, res)
    return np.where(is_pole, 0.0, res)

def _safe_lgamma(a):
    """GPU: 0 at poles, 0 for NaN/Inf results"""
    from scipy.special import gammaln
    is_pole = (a <= 0.0) & (np.floor(a) == a)
    # gammaln can produce NaN at poles
    with np.errstate(divide='ignore', invalid='ignore'):
        res = gammaln(np.where(is_pole, 1.0, a))  # dummy value at poles
    res = np.where(np.isnan(res) | np.isinf(res), 0.0, res)
    return np.where(is_pole, 0.0, res)


def _parse_formula_to_rpn(formula_str):
    """
    Parses an infix formula to RPN token list.
    Reuses ExpressionTree from grammar.py.
    Returns list of tokens in RPN order.
    """
    from AlphaSymbolic.core.grammar import ExpressionTree
    
    f_norm = formula_str.replace('**', '^')
    tree = ExpressionTree.from_infix(f_norm)
    
    if not tree.is_valid:
        raise ValueError(f"Could not parse formula: {formula_str}")
    
    rpn_tokens = []
    def traverse(node):
        if not node: return
        for child in node.children:
            traverse(child)
        rpn_tokens.append(node.value)
    traverse(tree.root)
    return rpn_tokens, tree


def protected_eval_cpu(formula_str, x_data):
    """
    Vía 2: Evalúa una fórmula usando semántica protegida idéntica a CUDA.
    
    x_data: dict {'x0': array, 'x1': array, ...}
    Returns: numpy array of predictions
    """
    rpn_tokens, tree = _parse_formula_to_rpn(formula_str)
    
    n_samples = len(next(iter(x_data.values())))
    
    # Stack-based RPN evaluation with CUDA-matching protected ops
    stack = []
    c_idx = 0
    const_values = []  # We extract numeric constants during parse
    
    for token in rpn_tokens:
        # Variables
        if token in x_data:
            stack.append(x_data[token].astype(np.float64))
            continue
        if token == 'x' and 'x0' in x_data:
            stack.append(x_data['x0'].astype(np.float64))
            continue
        
        # Named constants
        if token == 'pi':
            stack.append(np.full(n_samples, np.pi, dtype=np.float64))
            continue
        if token == 'e' and not any(token == 'e' for token in ['exp']):
            # 'e' as Euler's number (terminal, not exp operator)
            stack.append(np.full(n_samples, np.e, dtype=np.float64))
            continue
        
        # Numeric literals
        try:
            val = float(token)
            stack.append(np.full(n_samples, val, dtype=np.float64))
            continue
        except ValueError:
            pass
        
        # 'C' placeholder (default 1.0)
        if token == 'C':
            stack.append(np.full(n_samples, 1.0, dtype=np.float64))
            continue
        
        # Binary operators
        if token in ['+', '-', '*', '/', 'pow', '^', '%', 'mod']:
            if len(stack) < 2:
                return np.full(n_samples, np.nan)
            b = stack.pop()
            a = stack.pop()
            
            if token == '+': stack.append(a + b)
            elif token == '-': stack.append(a - b)
            elif token == '*': stack.append(a * b)
            elif token == '/': stack.append(_safe_div(a, b))
            elif token in ['pow', '^']: stack.append(_safe_pow(a, b))
            elif token in ['%', 'mod']: stack.append(_safe_mod(a, b))
            continue
        
        # Unary operators
        if len(stack) < 1:
            return np.full(n_samples, np.nan)
        a = stack.pop()
        
        if token == 'sin': stack.append(np.sin(a))
        elif token == 'cos': stack.append(np.cos(a))
        elif token == 'tan': stack.append(np.tan(a))
        elif token == 'abs': stack.append(np.abs(a))
        elif token == 'neg': stack.append(-a)
        elif token == 'sqrt': stack.append(_safe_sqrt(a))
        elif token == 'log': stack.append(_safe_log(a))
        elif token in ['exp', 'e']: stack.append(_safe_exp(a))
        elif token == 'floor' or token == '_': stack.append(np.floor(a))
        elif token == 'ceil': stack.append(np.ceil(a))
        elif token == 'sign': stack.append(np.sign(a))
        elif token in ['asin', 'S']: stack.append(_safe_asin(a))
        elif token in ['acos']: stack.append(_safe_acos(a))
        elif token in ['atan', 'T']: stack.append(np.arctan(a))
        elif token in ['fact', '!']: stack.append(_safe_tgamma(a + 1.0))
        elif token == 'gamma': stack.append(_safe_tgamma(a))
        elif token in ['lgamma', 'g']: stack.append(_safe_lgamma(a))
        else:
            print(f"  [!] Unknown token: '{token}' — treating as 0")
            stack.append(np.zeros(n_samples, dtype=np.float64))
    
    if len(stack) != 1:
        print(f"  [!] Stack has {len(stack)} elements after eval (expected 1)")
        return np.full(n_samples, np.nan) if not stack else stack[-1]
    
    return stack[0]


def strict_eval_cpu(formula_str, x_data):
    """
    Vía 3: Evalúa con ExpressionTree.evaluate de grammar.py (semántica estricta).
    """
    from AlphaSymbolic.core.grammar import ExpressionTree
    
    f_norm = formula_str.replace('**', '^')
    tree = ExpressionTree.from_infix(f_norm)
    
    if not tree.is_valid:
        raise ValueError(f"Could not parse formula: {formula_str}")
    
    return tree.evaluate(x_data)


def _detect_formula_ops(formula_str):
    """Detect which operators a formula uses (for auto-enabling in GPU grammar)."""
    rpn_tokens, _ = _parse_formula_to_rpn(formula_str)
    
    # Map from RPN token names to GpuGlobals attribute names
    OP_TO_GLOBAL = {
        'sin': 'USE_OP_SIN', 'cos': 'USE_OP_COS', 'tan': 'USE_OP_TAN',
        'log': 'USE_OP_LOG', 'exp': 'USE_OP_EXP', 'sqrt': 'USE_OP_SQRT',
        'abs': 'USE_OP_ABS', 'pow': 'USE_OP_POW', '^': 'USE_OP_POW',
        'fact': 'USE_OP_FACT', '!': 'USE_OP_FACT',
        'gamma': 'USE_OP_GAMMA', 'lgamma': 'USE_OP_GAMMA',
        'asin': 'USE_OP_ASIN', 'S': 'USE_OP_ASIN',
        'acos': 'USE_OP_ACOS',
        'atan': 'USE_OP_ATAN', 'T': 'USE_OP_ATAN',
        'floor': 'USE_OP_FLOOR', '_': 'USE_OP_FLOOR',
        'ceil': 'USE_OP_CEIL', 'sign': 'USE_OP_SIGN',
        '%': 'USE_OP_MOD', 'mod': 'USE_OP_MOD',
        # Basic ops (+,-,*,/) are always enabled
    }
    
    needed_globals = set()
    for token in rpn_tokens:
        if token in OP_TO_GLOBAL:
            needed_globals.add(OP_TO_GLOBAL[token])
    
    return needed_globals


def gpu_eval(formula_str, x_np, y_np, num_variables=1, use_strict=False):
    """
    Vía 1: Evalúa con GPU evaluate_batch.
    Auto-enables any operators needed by the formula.
    Returns: (rmse, predictions)
    """
    import torch
    import traceback
    from AlphaSymbolic.core.gpu.engine import TensorGeneticEngine
    from AlphaSymbolic.core.gpu.config import GpuGlobals
    
    # Detect which ops the formula needs and temporarily enable them
    needed_ops = _detect_formula_ops(formula_str)
    saved_ops = {}
    enabled_ops = []
    for op_global in needed_ops:
        old_val = getattr(GpuGlobals, op_global, True)
        saved_ops[op_global] = old_val
        if not old_val:
            setattr(GpuGlobals, op_global, True)
            enabled_ops.append(op_global)
    
    if enabled_ops:
        print(f"        [auto-enabled: {', '.join(enabled_ops)}]")
    
    try:
        engine = TensorGeneticEngine(
            pop_size=8,  # Minimal
            num_variables=num_variables,
            max_len=60,
            n_islands=1
        )
        
        # Convert formula to RPN tensor
        pop, consts = engine.load_population_from_strings([formula_str])
        
        if pop is None:
            return float('inf'), np.full(len(y_np), np.nan)
        
        # Verify RPN conversion: check no PAD in the middle of tokens
        rpn_str = engine.rpn_to_infix(pop[0].unsqueeze(0))
        print(f"        [GPU RPN->Infix: {rpn_str}]")
        
        # Prepare data as tensors
        x_t = torch.tensor(x_np, dtype=engine.dtype, device=engine.device)
        y_t = torch.tensor(y_np, dtype=engine.dtype, device=engine.device)
        
        # x_t should be [Vars, Samples]
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        
        # Get RMSE via evaluate_batch
        if use_strict:
            print("        [Mode: STRICT (validate_strict)]")
            res = engine.evaluator.validate_strict(pop, x_t, y_t, consts)
            rmse = res['rmse'][0].item()
            # For predictions, we need to call vm.eval manually with strict_mode=1
            preds_t, sp, err = engine.evaluator.vm.eval(pop, x_t, consts, strict_mode=1)
            preds = preds_t
        else:
            rmse_tensor = engine.evaluate_batch(pop, x_t, y_t, consts)
            rmse = rmse_tensor[0].item()
            
            # Get predictions via predict_individual
            preds = engine.predict_individual(pop[0], consts[0], x_t)
            
        preds_np = preds.cpu().numpy().flatten()
        
        return rmse, preds_np
    finally:
        # Restore original config
        for op_global, old_val in saved_ops.items():
            setattr(GpuGlobals, op_global, old_val)


def compute_rmse(predictions, targets):
    """Standard RMSE: sqrt(mean((pred - target)^2))"""
    diff = predictions - targets
    mse = np.mean(diff ** 2)
    return np.sqrt(mse)


# ═══════════════════════════════════════════════════════════════════
#  REPORT
# ═══════════════════════════════════════════════════════════════════

def compare_and_report(formula, x_data, y_target, num_variables=1, use_strict=False):
    """Runs all 3 paths and prints a comparative report."""
    
    n = len(y_target)
    
    # Prepare x as numpy array for GPU path: [Vars, Samples]
    if num_variables == 1:
        x_np = x_data['x0'].reshape(1, -1)
    else:
        x_np = np.array([x_data[f'x{i}'] for i in range(num_variables)])
    
    print(f"\n{'='*70}")
    print(f"  RMSE VERIFICATION -- Triple Path Comparison")
    print(f"{'='*70}")
    print(f"  Formula:   {formula}")
    print(f"  Samples:   {n}")
    print(f"  Variables: {num_variables}")
    print(f"{'-'*70}")
    
    # ── Vía 1: GPU ──
    print(f"\n  [1/3] GPU evaluate_batch...")
    t0 = time.time()
    try:
        gpu_rmse, gpu_preds = gpu_eval(formula, x_np, y_target, num_variables, use_strict)
        gpu_time = time.time() - t0
        gpu_ok = True
    except Exception as e:
        print(f"        ERROR: {e}")
        traceback.print_exc()
        gpu_rmse = float('inf')
        gpu_preds = np.full(n, np.nan)
        gpu_time = time.time() - t0
        gpu_ok = False
    
    # ── Vía 2: CPU protegida ──
    print(f"  [2/3] CPU protected (CUDA-matching)...")
    t0 = time.time()
    try:
        cpu_prot_preds = protected_eval_cpu(formula, x_data)
        cpu_prot_rmse = compute_rmse(cpu_prot_preds, y_target)
        cpu_prot_time = time.time() - t0
        cpu_prot_ok = True
    except Exception as e:
        print(f"        ERROR: {e}")
        cpu_prot_preds = np.full(n, np.nan)
        cpu_prot_rmse = float('inf')
        cpu_prot_time = time.time() - t0
        cpu_prot_ok = False
    
    # ── Vía 3: CPU estricta ──
    print(f"  [3/3] CPU strict (grammar.py)...")
    t0 = time.time()
    try:
        cpu_strict_preds = strict_eval_cpu(formula, x_data)
        cpu_strict_rmse = compute_rmse(cpu_strict_preds, y_target)
        cpu_strict_time = time.time() - t0
        cpu_strict_ok = True
    except Exception as e:
        print(f"        ERROR: {e}")
        cpu_strict_preds = np.full(n, np.nan)
        cpu_strict_rmse = float('inf')
        cpu_strict_time = time.time() - t0
        cpu_strict_ok = False
    
    # ===================== RMSE COMPARISON TABLE =====================
    print(f"\n{'='*70}")
    print(f"  RMSE COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Path':<30s} {'RMSE':>15s} {'Time':>10s} {'Status':>8s}")
    print(f"  {'-'*30} {'-'*15} {'-'*10} {'-'*8}")
    
    print(f"  {'[1] GPU evaluate_batch':<30s} {gpu_rmse:>15.10f} {gpu_time*1000:>8.1f}ms {'OK' if gpu_ok else 'FAIL':>8s}")
    print(f"  {'[2] CPU protected (CUDA)':<30s} {cpu_prot_rmse:>15.10f} {cpu_prot_time*1000:>8.1f}ms {'OK' if cpu_prot_ok else 'FAIL':>8s}")
    print(f"  {'[3] CPU strict (grammar.py)':<30s} {cpu_strict_rmse:>15.10f} {cpu_strict_time*1000:>8.1f}ms {'OK' if cpu_strict_ok else 'FAIL':>8s}")
    
    # ── Deltas ──
    if gpu_ok and cpu_prot_ok:
        delta_12 = abs(gpu_rmse - cpu_prot_rmse)
        rel_delta_12 = delta_12 / max(gpu_rmse, 1e-15) * 100
        match_12 = "MATCH" if delta_12 < 1e-6 else "DIFF"
        print(f"\n  Delta GPU vs CPU-protected:  {delta_12:.2e}  ({rel_delta_12:.4f}%)  [{match_12}]")
    
    if gpu_ok and cpu_strict_ok:
        delta_13 = abs(gpu_rmse - cpu_strict_rmse)
        rel_delta_13 = delta_13 / max(gpu_rmse, 1e-15) * 100
        match_13 = "MATCH" if delta_13 < 1e-6 else "DIFF"
        print(f"  Delta GPU vs CPU-strict:     {delta_13:.2e}  ({rel_delta_13:.4f}%)  [{match_13}]")
    
    if cpu_prot_ok and cpu_strict_ok:
        delta_23 = abs(cpu_prot_rmse - cpu_strict_rmse)
        match_23 = "MATCH" if delta_23 < 1e-6 else "DIFF"
        print(f"  Delta CPU-prot vs CPU-strict: {delta_23:.2e}  [{match_23}]")
    
    # ===================== PER-POINT TABLE =====================
    print(f"\n{'='*70}")
    print(f"  PER-POINT PREDICTIONS")
    print(f"{'='*70}")
    
    # Determine which x to show
    x_show = x_data['x0'] if num_variables == 1 else np.arange(n)
    x_label = 'x0' if num_variables == 1 else 'idx'
    
    print(f"  {x_label:>8s} {'Target':>12s} {'GPU':>12s} {'CPU-Prot':>12s} {'CPU-Strict':>12s} {'D(GPU,Prot)':>12s}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    # Limit rows for readability
    max_rows = min(n, 30)
    indices = np.linspace(0, n - 1, max_rows, dtype=int) if n > 30 else range(n)
    
    for i in indices:
        xi = x_show[i]
        ti = y_target[i]
        gi = gpu_preds[i] if i < len(gpu_preds) else float('nan')
        pi = cpu_prot_preds[i] if i < len(cpu_prot_preds) else float('nan')
        si = cpu_strict_preds[i] if i < len(cpu_strict_preds) else float('nan')
        delta_gp = abs(gi - pi) if not (np.isnan(gi) or np.isnan(pi)) else float('nan')
        
        print(f"  {xi:>8.4f} {ti:>12.6f} {gi:>12.6f} {pi:>12.6f} {si:>12.6f} {delta_gp:>12.2e}")
    
    if n > 30:
        print(f"  ... ({n - 30} more rows omitted)")
    
    # ===================== SEMANTIC NOTES =====================
    # Check if any predictions differ between CPU-protected and CPU-strict
    if cpu_prot_ok and cpu_strict_ok:
        diff_mask = ~np.isclose(cpu_prot_preds, cpu_strict_preds, rtol=1e-9, atol=1e-12, equal_nan=True)
        n_diff = np.sum(diff_mask)
        if n_diff > 0:
            print(f"\n  [!] {n_diff}/{n} points differ between protected and strict semantics.")
            print(f"    This is expected for formulas that use operators in edge domains")
            print(f"    (log with negatives, sqrt with negatives, pow with negative bases, etc.)")
        else:
            print(f"\n  [OK] All {n} points match between protected and strict semantics.")
            print(f"    This formula doesn't hit any protected-operator edge cases.")
    
    print(f"\n{'='*70}\n")
    
    return {
        'gpu_rmse': gpu_rmse,
        'cpu_prot_rmse': cpu_prot_rmse,
        'cpu_strict_rmse': cpu_strict_rmse,
        'gpu_preds': gpu_preds,
        'cpu_prot_preds': cpu_prot_preds,
        'cpu_strict_preds': cpu_strict_preds,
    }


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Verify RMSE: 1:1 comparison GPU vs CPU (protected + strict)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_rmse.py "sin(x0) + x0^2"
  python verify_rmse.py "log(x0) + sqrt(x0)" -r 0.1,5
  python verify_rmse.py "x0^3 - 3.3628" --xy "1,2,3,4,5" "0,-5,-24,-61,-122"
  python verify_rmse.py "x0^2 + x1" --vars 2 -r -2,2 -n 15
        """
    )
    parser.add_argument("formula", help="Infix formula to evaluate (use x0, x1, ... for variables)")
    parser.add_argument("-r", "--range", type=str, default="-5,5",
                        help="X range as 'min,max' (default: -5,5)")
    parser.add_argument("-n", "--n-points", type=int, default=20,
                        help="Number of data points (default: 20)")
    parser.add_argument("--vars", type=int, default=1,
                        help="Number of variables (default: 1)")
    parser.add_argument("--xy", nargs=2, type=str, default=None,
                        help="Explicit X and Y data as comma-separated values: --xy '1,2,3' '4,5,6'")
    parser.add_argument("--target-formula", type=str, default=None,
                        help="Target formula to generate Y data (default: same as formula)")
    parser.add_argument("--strict", action='store_true', help="Use GPU strict validation mode")
    
    args = parser.parse_args()
    
    if args.xy:
        # Explicit X,Y data
        x_vals = np.array([float(v) for v in args.xy[0].split(',')], dtype=np.float64)
        y_vals = np.array([float(v) for v in args.xy[1].split(',')], dtype=np.float64)
        x_data = {'x0': x_vals}
        num_variables = 1
    else:
        # Auto-generate from range
        r = [float(v) for v in args.range.split(',')]
        num_variables = args.vars
        n = args.n_points
        
        x_data = {}
        for vi in range(num_variables):
            x_data[f'x{vi}'] = np.linspace(r[0], r[1], n).astype(np.float64)
        
        # Generate target Y from target formula (or use the formula itself as target)
        target_formula = args.target_formula or args.formula
        
        # Use protected eval to generate Y (since we want Y consistent with GPU)
        y_vals = protected_eval_cpu(target_formula, x_data)
        
        if np.any(np.isnan(y_vals)):
            print(f"  [!] Warning: Target formula produces NaN at some points.")
            print(f"      Using strict eval for target generation instead...")
            y_vals = strict_eval_cpu(target_formula, x_data)
    
    compare_and_report(args.formula, x_data, y_vals, num_variables=args.vars if not args.xy else 1, use_strict=args.strict)


if __name__ == "__main__":
    main()
