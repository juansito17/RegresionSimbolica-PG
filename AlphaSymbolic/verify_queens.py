
import math
import numpy as np

# A000170: Number of ways to place n non-attacking queens on an n X n board.
# n=1..27
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
        # Gen 1 Formula: ((lgamma((1 + x0)) - cos((x1 / ((0.13184391 * x0) ^ x0)))) - x0)
        
        # Term 1
        term1 = math.lgamma(1 + x0)
        
        # Cos Inner
        # Denominator: (0.13184391 * x0) ^ x0
        base = 0.13184391 * x0
        denom = base ** x0 # x0 is positive integer, base > 0 (for n>=1). Safe.
        
        cos_arg = x1 / denom if abs(denom) > 1e-15 else 0
        
        term2 = math.cos(cos_arg)
        
        # log_pred = term1 - term2 - x0
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

if count > 0:
    rmsle = math.sqrt(total_log_error / count)
    print("-" * 95)
    print(f"Calculated RMSLE: {rmsle:.8f}")
