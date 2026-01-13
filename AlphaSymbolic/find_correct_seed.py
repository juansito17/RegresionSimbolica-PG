
import numpy as np
import torch
from core.grammar import ExpressionTree
from scipy.special import gamma, gammaln

# Data
# SOTA 0.059 might be on N=25?
X_RAW_FULL = np.arange(4, 28)
Y_RAW_FULL = np.array([2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528], dtype=np.float64)

def check_scenario(max_n):
    limit = max_n - 3 # indices 4..max_n. n=4 is idx 0. n=25 is idx 21 (22 items)
    # wait. X=4..27. 24 items.
    # if max_n=25. 4..25. 22 items.
    
    indices = np.where(X_RAW_FULL <= max_n)[0]
    
    x = X_RAW_FULL[indices]
    y_log = np.log(Y_RAW_FULL[indices])
    
    return x, y_log

print("Searching variations on N=27 and N=25...")

def get_error(formula_str):
    try:
        tree = ExpressionTree.from_infix(formula_str)
        # Construct input dict
        x0 = X_RAW
        x1 = X_RAW % 6
        x2 = X_RAW % 2
        vals = {'x0': x0, 'x1': x1, 'x2': x2}
        
        preds = tree.evaluate(vals)
        
        # RMSE
        mse = np.mean((preds - Y_LOG)**2)
        rmse = np.sqrt(mse)
        return rmse
    except Exception as e:
        return 999.0

# Base Template (from Gen 25)
# old: gamma(x0)
# old: lgamma( ... + x0 )

gamma_replacements = [
    "lgamma(x0)", 
    "lgamma(x0+1)", 
    "lgamma(x0+2)",
    "lgamma(x0-1)",
    "gamma(x0)", # tgamma
    "gamma(x0+1)", # factorial
]

lgamma_outer_replacements = [
    "lgamma(ARG)",
    "lgamma(ARG+1)",
    "lgamma(ARG+2)",
]

# Inner part of Lgamma argument:
# cos((cos(gamma(x0)) / sqrt(x0))) + x0
# We need to replace gamma(x0) here too

candidates = []

base = "((OUTER - cos(((x0 + x0) / ((0.3046774 * (INNER - x0)) ^ x0)))) - x0)"


scenarios = [25, 27]

for max_n in scenarios:
    print(f"\n--- Checking SCENARIO N <= {max_n} ---")
    x_set, y_log_set = check_scenario(max_n)
    
    # helper
    def calc_rmse(f):
        try:
            tree = ExpressionTree.from_infix(f)
            vals = {'x0': x_set, 'x1': x_set%6, 'x2': x_set%2}
            preds = tree.evaluate(vals)
            return np.sqrt(np.mean((preds - y_log_set)**2))
        except: return 999.0

    best_s_rmse = 999.0
    best_s_f = ""

    for g_rep in gamma_replacements:
        arg_struct = f"(cos((cos({g_rep}) / sqrt(x0))) + x0)"
        for l_wrap in lgamma_outer_replacements:
            outer_term = l_wrap.replace("ARG", arg_struct)
            f = f"(( {outer_term} - cos(((x0 + x0) / ((0.3046774 * ({g_rep} - x0)) ** x0)))) - x0)"
            
            rmse = calc_rmse(f)
            if rmse < best_s_rmse:
                best_s_rmse = rmse
                best_s_f = f
                
    print(f"Best for N={max_n}: {best_s_rmse:.5f} | Formula: {best_s_f}")

