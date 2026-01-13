import math
import numpy as np

# Ground Truth (OEIS A000170)
Y_RAW = [
    1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 
    2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 
    314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528
]

def sota_formula(x0, x1, x2):
    """
    SOTA Formula (Gen 100)
    fitness: 0.0517 (RMSLE)
    """
    try:
        # Term 1: lgamma(x0 + 2) is approx ln((n+1)!)
        t1 = math.lgamma(x0 + 2)
        
        # Term 2: Modulation
        # Denom power structure
        exponent = (math.sin(-0.18788848) + (x0 / math.sin(1))) + x0 + x2
        denom_base = 0.11575544 * x0
        denom = denom_base ** exponent
        t2 = math.cos(x1 / denom)
        
        # Term 3: Subtraction (Linear correction)
        # cos(sin(small)) ~ 1, so t3 ~ x0
        t3_pow = math.cos(math.sin(0.02719994))
        t3 = x0 ** t3_pow
        
        return (t1 - t2) - t3
    except Exception:
        return float('nan')

print(f"{'N':<4} | {'Real':<25} | {'Predicted':<25} | {'Error %':<15} | {'Log Error'}")
print("-" * 90)

total_log_error = 0
count = 0

for i, real_val in enumerate(Y_RAW):
    n = i + 1
    
    # Inputs
    x0 = float(n)
    x1 = float(n % 6)
    x2 = float(n % 2)
    
    # Predict
    pred_log = sota_formula(x0, x1, x2)
    pred_raw = math.exp(pred_log)
    
    # Display for all N
    if real_val == 0:
        error_pct_str = "N/A"
        log_err_str = "N/A"
    else:
        error_pct = abs(pred_raw - real_val) / real_val * 100
        error_pct_str = f"{error_pct:.6f}%"
        
        log_real = math.log(real_val)
        log_diff = (pred_log - log_real)**2
        log_err_str = f"{log_diff:.8f}"
        
        # RMSLE Calculation (Standardized on N>=4 to avoid zeros)
        if n >= 4:
            total_log_error += log_diff
            count += 1
            
    print(f"{n:<4} | {real_val:<25} | {pred_raw:<25.4f} | {error_pct_str:<15} | {log_err_str}")

print("-" * 90)
if count > 0:
    rmse = math.sqrt(total_log_error / count)
    print(f"RMSLE (N=4..27): {rmse:.8f}")
