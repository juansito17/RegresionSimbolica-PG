
import numpy as np
from core.grammar import ExpressionTree
from scipy.special import gamma, gammaln

# Data (N=4..27)
X = np.arange(4, 28, dtype=np.float64)
Y_RAW = np.array([2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528], dtype=np.float64)
Y_LOG = np.log(Y_RAW)

def test(f_str):
    try:
        tree = ExpressionTree.from_infix(f_str)
        vals = {'x0': X, 'x1': X % 6, 'x2': X % 2}
        preds = tree.evaluate(vals)
        rmse = np.sqrt(np.mean((preds - Y_LOG)**2))
        return rmse
    except:
        return 999.0

# Base Components
# ((lgamma((cos((cos(InnerG)) / sqrt(x0))) + InnerL)) - cos(((x0 + x0) / ((0.3046774 * (InnerG2 - x0)) ^ x0)))) - x0)

# Variations for InnerG (gamma(x0))
# Old gamma(x0) -> lgamma(x0+1)
# Maybe lgamma(x0)? lgamma(x0+2)? gamma(x0)?
g_vars = [
    "lgamma(x0)", "lgamma(x0+1)", "lgamma(x0+2)", "lgamma(x0+3)",
    "gamma(x0)", "gamma(x0+1)",
]

# Variations for InnerL (Outer lgamma argument shift)
# Formula: lgamma( ... + x0 )
# Old kernel added +1 to arg.
# So effectively: lgamma( ... + x0 + 1 )
# Maybe +2? +0?
l_vars = [
    "x0", "x0+1", "x0+2", "x0-1"
]

print("Searching variations...")

best_rmse = 999.0
best_formula = ""

for g in g_vars:
    # We assume 'gamma(x0)' in both places was same token?
    # Yes.
    g1 = g
    g2 = g
    
    for l_shift in l_vars:
        # Construct formula
        # ((lgamma((cos((cos(g1) / sqrt(x0))) + l_shift)) - cos(((x0 + x0) / ((0.3046774 * (g2 - x0)) ^ x0)))) - x0)
        
        f = f"((lgamma((cos((cos({g1}) / sqrt(x0))) + {l_shift})) - cos(((x0 + x0) / ((0.3046774 * ({g2} - x0)) ^ x0)))) - x0)"
        
        err = test(f)
        if err < 2.0:
            print(f"RMSE: {err:.5f} | Shift: {l_shift} | Gamma: {g1}")
        
        if err < best_rmse:
            best_rmse = err
            best_formula = f

print("-" * 40)
print(f"BEST RMSE: {best_rmse}")
print(f"BEST FORMULA: {best_formula}")
