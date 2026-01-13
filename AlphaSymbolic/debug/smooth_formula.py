
import numpy as np
import math
from scipy.optimize import minimize
from scipy.special import gamma

Y_RAW = np.array([
    1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 
    2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 
    314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528
], dtype=np.float64)

X = np.arange(1, 28, dtype=np.float64)

def safe_gamma(x):
    return gamma(abs(x) + 1e-10)

# Precompute constant sin_g for even/odd to save time? 
# n%2==0 -> x2=0 -> sin(gamma(0)).
# n%2==1 -> x2=1 -> sin(gamma(1)).
SIG_EVEN = math.sin(safe_gamma(0))
SIG_ODD = math.sin(safe_gamma(1))

# Target Log
Y_LOG = np.zeros_like(Y_RAW)
for i, y in enumerate(Y_RAW):
    if y > 0: Y_LOG[i] = math.log(y)

def evaluate_vec(k):
    k1, k2 = k
    
    preds_log = []
    
    for i, x0 in enumerate(X):
        n = int(x0)
        target = Y_RAW[i]
        if target == 0: 
            preds_log.append(0) # Ignored
            continue
            
        x1 = float(n % 6)
        x2 = float(n % 2)
        
        try:
            # Formula Structure
            # ((lgamma((x0 + 2)) - cos((x1 / ((k1 * x0) ** (((sin(k2) + (x0 / sin_g)) + x0) + x2))))) - x0)
            
            term1 = math.lgamma(x0 + 2)
            
            sin_g = SIG_ODD if x2 == 1 else SIG_EVEN
            if abs(sin_g) < 1e-10: sin_g = 1e-10
            
            # exp = sin(k2) + x0/sin_g + x0 + x2
            exp_val = math.sin(k2) + (x0 / sin_g) + x0 + x2
            
            base = k1 * x0
            if base <= 0: denom = 1e-10
            else: denom = base ** exp_val
            
            if denom == 0: arg = 0
            else: arg = x1 / denom
            
            term2 = math.cos(arg)
            
            log_p = term1 - term2 - x0
            preds_log.append(log_p)
            
        except:
             preds_log.append(999.0)

    # Calculate Loss
    # Default RMSE
    # But add penalty for large deviations
    
    valid_indices = [i for i, y in enumerate(Y_RAW) if y > 0]
    y_log_valid = Y_LOG[valid_indices]
    preds_valid = np.array([preds_log[i] for i in valid_indices])
    
    diff = preds_valid - y_log_valid
    mse = np.mean(diff**2)
    
    # Max Deviation Penalty (To reduce spikes)
    max_dev = np.max(np.abs(diff))
    
    # Combined Loss: RMSE + 0.5 * MaxDev
    loss = np.sqrt(mse) + 0.5 * max_dev
    return loss

# Initial guess from user formula
# k1: 0.11580232
# k2: -0.16773652
k0 = [0.11580232, -0.16773652]

print(f"Initial Loss function value: {evaluate_vec(k0)}")
print("Optimizing...")

res = minimize(evaluate_vec, k0, method='Nelder-Mead', tol=1e-5)

print(f"Optimized K1: {res.x[0]}")
print(f"Optimized K2: {res.x[1]}")
print(f"Final Loss: {res.fun}")
