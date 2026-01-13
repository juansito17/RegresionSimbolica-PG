import math
import numpy as np
import os

# Ground Truth (OEIS A000170)
Y_RAW = [
    1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 
    2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 
    314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528
]

def sota_formula(x0, x1, x2):
    try:
        # Formula Gen 100
        # ((lgamma((x0 + 2)) - cos((x1 / ((0.11575544 * x0) ** (((sin(-0.18788848) + (x0 / math.sin(1))) + x0) + x2))))) - (x0 ** math.cos(math.sin(0.02719994))))
        
        t1 = math.lgamma(x0 + 2)
        
        exponent = (math.sin(-0.18788848) + (x0 / math.sin(1))) + x0 + x2
        denom_base = 0.11575544 * x0
        
        # Protect against negative base with float power (though base seems positive for n>0)
        # 0.115... * n is > 0
        denom = denom_base ** exponent
        
        t2 = math.cos(x1 / denom)
        
        t3_pow = math.cos(math.sin(0.02719994))
        t3 = x0 ** t3_pow
        
        return (t1 - t2) - t3
    except Exception as e:
        return float('nan')

with open("sota_eval_full.txt", "w") as f:
    f.write(f"{'N':<4} | {'Real':<25} | {'Predicted':<25} | {'Error %':<15} | {'LogError'}\n")
    f.write("-" * 95 + "\n")

    total_log_error = 0
    count = 0

    # Start from N=4 as used in training often, but list all
    # solve_n_queens says "Training on n=4..27"
    
    sq_errors = []

    for i, real_val in enumerate(Y_RAW):
        n = i + 1
        
        x0 = float(n)
        x1 = float(n % 6)
        x2 = float(n % 2)
        
        pred_log = sota_formula(x0, x1, x2)
        pred_raw = math.exp(pred_log)
        
        if real_val == 0:
            f.write(f"{n:<4} | {real_val:<25} | {pred_raw:<25.4f} | {'N/A':<15} | N/A\n")
            continue
            
        error_pct = abs(pred_raw - real_val) / real_val * 100
        log_real = math.log(real_val)
        log_diff = (pred_log - log_real)**2
        
        # Only count N >= 4 for RMSLE to match training
        if n >= 4:
            total_log_error += log_diff
            count += 1
            sq_errors.append(log_diff)

        f.write(f"{n:<4} | {real_val:<25} | {pred_raw:<25.4f} | {error_pct:<15.6f}% | {log_diff:.8f}\n")

    if count > 0:
        rmse = math.sqrt(total_log_error / count)
        f.write("-" * 95 + "\n")
        f.write(f"RMSLE (N=4..27): {rmse:.8f}\n")
        f.write(f"SQ Errors: {sq_errors}\n")

print("Evaluation complete. Results in sota_eval_full.txt")
