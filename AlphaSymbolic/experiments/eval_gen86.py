import numpy as np
import math

# Dataset (N=1..27)
Y_RAW = np.array([
    1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 
    2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 
    314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528
], dtype=np.float64)

X_RAW = np.arange(1, 28)

def eval_gen95_legacy(n):
    # Formula (Gen 95):
    # (((log(gamma(x0)) - x0) + log((lgamma(log(gamma(sqrt(x0)))) + x0))) - cos((sqrt(x1) / ((lgamma(x0) - x0) + sin((sin(x0) + sqrt(x0)))))))
    
    # "Old Gamma" Interpretation:
    # gamma(x) -> factorial(x) -> lgamma(x+1)
    # lgamma(x) -> lgamma(x)
    
    x0 = float(n)
    x1 = float(n % 6)
    
    try:
        # Term 1: log(gamma(x0)) - x0
        # gamma(x0) -> lgamma(x0+1)
        term1 = math.lgamma(x0 + 1) - x0
        
        # Term 2: log((lgamma(log(gamma(sqrt(x0)))) + x0))
        # Inner: gamma(sqrt(x0)) -> factorial(sqrt(x0)) -> lgamma(sqrt(x0)+1)
        sqrt_x0 = math.sqrt(x0)
        inner_gamma = math.lgamma(sqrt_x0 + 1) # log(gamma_func)
        
        # log(gamma(sqrt(x0))) ??
        # The formula says: log(gamma(sqrt(x0)))
        # Wait, gamma(x) returns the VALUE. log(VALUE) = lgamma(x+1).
        # So "log(gamma(sqrt(x0)))" is effectively lgamma(sqrt(x0)+1).
        
        # Then lgamma() of THAT.
        # So lgamma( lgamma(sqrt(x0)+1) )
        if inner_gamma <= 0: inner_gamma = 1e-9
        nested_term = math.lgamma(inner_gamma)
        
        term2 = math.log(abs(nested_term + x0))
        
        # Term 3: Cosine Modulation (Same as before)
        denom_base = math.lgamma(x0) - x0
        denom_osc = math.sin(math.sin(x0) + math.sqrt(x0))
        denom = denom_base + denom_osc
        if abs(denom) < 1e-9: denom = 1e-9
        
        arg = math.sqrt(x1) / denom
        term3 = math.cos(arg)
        
        log_pred = (term1 + term2) - term3
        return math.exp(log_pred)
        
    except Exception as e:
        return 0

print(f"{'N':<3} | {'Actual':<22} | {'Gen 95 (Old)':<22} | {'Error %':<10}")
print("-" * 65)

errors = []
large_n_errors = []

# Engine excludes N=1,2,3. Evaluation loop typically N=1..27 for us.
sq_errors_engine = []
valid_n_engine = []

for i in range(27):
    n = i + 1
    actual = Y_RAW[i]
    pred = eval_gen95_legacy(n)
    
    # Display All
    if actual > 0:
        error_pct = (pred - actual) / actual * 100
        errors.append(abs(error_pct))
        if n >= 20: large_n_errors.append(abs(error_pct))
        print(f"{n:<3} | {actual:<22.0f} | {pred:<22.0f} | {error_pct:+.2f}%")
    else:
        print(f"{n:<3} | {actual:<22.0f} | {pred:<22.4f} | N/A")
        
    # Engine Metric (N=4..27)
    if n >= 4:
        valid_n_engine.append(n)
        if pred <= 1e-9: pred = 1e-9
        sq_errors_engine.append( (math.log(pred) - math.log(actual))**2 )



print("-" * 65)
print(f"MAPE (Global): {np.mean(errors):.2f}%")
if large_n_errors:
    print(f"MAPE (N>=20): {np.mean(large_n_errors):.2f}%")

# Calculate RMSE (Fitness Metric) - ENGINE REPRODUCTION
# Engine Logic (core/gpu/engine.py):
# if USE_LOG_TRANSFORMATION:
#      mask = y_t > 1e-9
#      y_t = torch.log(y_t[mask])
#      x_t = x_t[mask]

# FILTERING: Start from n=4 to avoid 0s (problematic for Log transform)
# Indices 0, 1, 2 correspond to n=1, 2, 3.
# n=2,3 are 0.
# START_INDEX = 3 # Start from n=4 (Index 3)
# Train N = 4..27 (Indices 3..26 if array is 0-indexed len 27)

sq_errors_engine = []
valid_n_engine = []

# Loop from N=4 to 25 (Training range usually stops at 25? Or 27?)
# solve_n_queens.py uses X_VALUES[START_INDEX:] where X_END=27.
# So it trains on N=4..27.
# But often we validate on N=4..25.
# Let's check typical behavior. The raw array has 27 elements.
# If Engine runs on all passed Y_TRAIN, it runs on N=4..27.

for i in range(3, 27): # N=4..27 (Indices 3..26)
    n = i + 1
    actual = Y_RAW[i]
    
    # Engine logic: Skip zeros (redundant if N>=4)
    if actual <= 1e-9: 
        continue
        
    valid_n_engine.append(n)
    
    pred = eval_gen95_legacy(n)
    if pred <= 1e-9: pred = 1e-9
    
    log_actual = math.log(actual) 
    log_pred = math.log(pred)
    
    sq_errors_engine.append((log_pred - log_actual) ** 2)

rmse_engine = math.sqrt(np.mean(sq_errors_engine))
print("-" * 65)
print(f"Gen 95 Fitness (N=4..27): {rmse_engine:.8f}")
print(f"Evaluated Points: {valid_n_engine}")


