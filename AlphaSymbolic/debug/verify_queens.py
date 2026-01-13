
import sys
import math
import numpy as np

# A000170: Number of ways to place n non-attacking queens on an n X n board.
# n=1..27
# 
# VERIFICATION NOTE:
# This script verifies the formula found by the GPU Genetic Engine.
# The low error (RMSLE ~0.09) is only achievable if we replicate the GPU Kernel's non-standard operator definitions:
# 1. 'lgamma(x)' in formula maps to GPU 'safe_gamma' which computes lgamma(x + 1) -> log(x!).
#    So 'lgamma(1+x0)' effectively becomes lgamma(x0+2) -> log((x0+1)!).
# 2. 'gamma(x)' (token '!') in formula ALSO maps to 'safe_gamma' -> lgamma(x + 1) -> log(x!).
#    So 'gamma(x0)' leads to lgamma(x0+1) -> log(x0!).
#
# These shifts mean the formula is exploiting the kernel's implementation details.

Y_RAW = [
    1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 
    2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 
    314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528
]

print(f"{'N':<4} | {'Log Target':<15} | {'Log Pred':<15} | {'Diff':<10} | {'Pred':<15} | {'Target':<15}")
print("-" * 95)


print(f"Total items in Y_RAW: {len(Y_RAW)}")
total_log_error = 0
count = 0

for i, target in enumerate(Y_RAW):
    n = i + 1
    # print(f"DEBUG: n={n}, target={target}")
    if n < 4: continue # Skip 0s

    
    # Calculate Log Target
    log_target = math.log(target)
    
    # Variables
    x0 = float(n)
    x1 = float(n % 6)
    x2 = float(n % 2)
    
    # Formula Calculation
    try:
        # Gen 9 Formula: ((lgamma((1 + x0)) - cos(((x0 + x0) / ((0.30467582 * (gamma(x0) - x0)) ^ x0)))) - x0)
        
        # ROOT CAUSE ANALYSIS:
        # The GPU Kernel 'safe_gamma' function implements lgamma(x) as lgamma(x + 1).
        # This was likely intended for Factorial (x! = Gamma(x+1)), but it applies to the 'lgamma' operator too.
        # Therefore, 'lgamma(1 + x0)' in the formula is actually evaluated as:
        #   lgamma((1 + x0) + 1) = lgamma(x0 + 2) = log((x0 + 1)!)
        
        # Corrected Python Equivalent for term1:
        term1 = math.lgamma(x0 + 2)
        
        # Denominator
        try:
             # Hypothesis 2: 'gamma' token in formula ALSO maps to 'safe_gamma' in kernel.
             # 'safe_gamma' returns lgamma(x+1).
             # So 'gamma(x0)' is actually lgamma(x0+1) which is log(x0!).
             
             # Previous incorrect: math.gamma(x0) -> huge number.
             # New Correct: math.lgamma(x0 + 1)
             
             gamma_val = math.lgamma(x0 + 1)
             
             inner_base = 0.30467582 * (gamma_val - x0)
             denom = inner_base ** x0
        except OverflowError:
             denom = float('inf')
        except ValueError:
             denom = float('inf') # math.pow domain error

        numerator = x0 + x0
        
        if denom == 0: cos_arg = 0
        elif denom == float('inf'): cos_arg = 0
        else: cos_arg = numerator / denom
        
        term2 = math.cos(cos_arg)
        
        # Final Formula
        log_pred = term1 - term2 - x0
        
        # Pred
        pred = math.exp(log_pred)
        
        # Error
        diff = abs(log_pred - log_target)
        sq_error = diff ** 2
        
        total_log_error += sq_error
        count += 1

        print(f"{n:<4} | {log_target:<15.4f} | {log_pred:<15.4f} | {diff:<10.4f} | {pred:<15.2e} | {target:<15.2e}")
        
    except Exception as e:
        print(f"{n:<4} | Error: {e}")

# Note: RMSLE calculated here will be only for these 3 points, so ignore final RMSLE print
if count > 0:
    print(f"\nRMSLE (Full Set N=4..27): {math.sqrt(total_log_error / count):.8f}")
