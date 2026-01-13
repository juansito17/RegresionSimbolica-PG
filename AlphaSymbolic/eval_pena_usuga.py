import numpy as np
import math

# Dataset (N=1..27)
Y_RAW = np.array([
    1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 
    2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 
    314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528
], dtype=np.float64)

X_RAW = np.arange(1, 28)

def pena_usuga_formula(n):
    # Log Space calculation: log(Q) = lgamma(n+1) - An + B
    
    if n % 2 == 0: # PARES
        A = 0.945525
        B = 0.966099
    else: # IMPARES
        A = 0.943389
        B = 0.911941
        
    log_val = math.lgamma(n + 1) - (A * n) + B
    return math.exp(log_val)

print(f"{'N':<3} | {'Actual':<22} | {'Peña-Usuga':<22} | {'Error %':<10}")
print("-" * 65)

errors = []
large_n_errors = []

for i, n in enumerate(X_RAW):
    actual = Y_RAW[i]
    pred = pena_usuga_formula(n)
    
    if actual > 0:
        error_pct = (pred - actual) / actual * 100
        errors.append(abs(error_pct))
        
        if n >= 20:
            large_n_errors.append(abs(error_pct))
            
        print(f"{n:<3} | {actual:<22.0f} | {pred:<22.0f} | {error_pct:+.2f}%")
    else:
        print(f"{n:<3} | {actual:<22.0f} | {pred:<22.4f} | N/A")

print("-" * 65)
print(f"MAPE (Global): {np.mean(errors):.2f}%")
if large_n_errors:
    print(f"MAPE (N>=20): {np.mean(large_n_errors):.2f}%")
