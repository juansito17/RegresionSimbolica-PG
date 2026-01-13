import numpy as np
import math
from scipy.optimize import minimize

# Dataset (N=1..27)
Y_RAW = np.array([
    1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 
    2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 
    314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528
], dtype=np.float64)

X_RAW = np.arange(1, 28)

def seed_model(params, n):
    # Form: log(Pred) = log_C0 + lgamma(n+1) - c*n + gamma*log(n)
    # Params: [log_C0, c, gamma]
    log_c0, c, gamma = params
    
    val = log_c0 + math.lgamma(n + 1) - (c * n) + (gamma * math.log(n))
    return val # Returns Log(Prediction)

def objective(params):
    # Fit against N=8..27
    start_idx = 7
    
    weighted_errors = []
    total_weight = 0
    
    for i in range(start_idx, len(X_RAW)):
        n = X_RAW[i]
        log_y_true = math.log(Y_RAW[i])
        log_y_pred = seed_model(params, n)
        
        # Weighting: Increase weight with N to force asymptotic fit
        # Using N^2 as weight
        weight = n ** 2
        
        sq_error = (log_y_true - log_y_pred) ** 2
        weighted_errors.append(weight * sq_error)
        total_weight += weight
        
    return np.sum(weighted_errors) / total_weight

# Initial guess from previous run
# C0=96.6 -> log(C0)=4.57
# c=0.84
# gamma=-1.9
x0 = [4.57, 0.84, -1.9]

print("Optimizing Simplified Seed Formula...")
res = minimize(objective, x0, method='Nelder-Mead', tol=1e-6)

best_params = res.x
log_c0, c, gamma = best_params

print(f"Optimization Success: {res.success}")
print(f"Best Params: log_C0={log_c0:.4f}, c={c:.4f}, gamma={gamma:.4f}")

# Verification
print(f"{'N':<3} | {'Actual':<22} | {'Predicted':<22} | {'Error %':<10}")
print("-" * 65)

errors = []
for i in range(7, 27): # N=8..27
    n = X_RAW[i]
    actual = Y_RAW[i]
    log_pred = seed_model(best_params, n)
    pred = math.exp(log_pred)
    
    error_pct = (pred - actual) / actual * 100
    errors.append(abs(error_pct))
    print(f"{n:<3} | {actual:<22.0f} | {pred:<22.0f} | {error_pct:+.2f}%")

print("-" * 65)
print(f"MAPE (N=8..27): {np.mean(errors):.2f}%")
