import numpy as np
import math

# Dataset (N=1..27)
Y_RAW = np.array([
    1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 
    2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 
    314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528
], dtype=np.float64)

X_RAW = np.arange(1, 28)

def sota_gen69(n):
    # Variables
    x0 = float(n)
    x1 = float(n % 6)
    
    # Formula: 
    # (((log(gamma(x0)) - x0) + log((sin(abs(sin(cos(x0)))) + x0))) - cos((sqrt(x1) / ((lgamma(x0) - x0) + sin((sin(x0) + sqrt(x0)))))))
    
    try:
        # Term 1: log(gamma(x0)) - x0
        # gamma(n) = (n-1)! -> log(gamma(n)) = lgamma(n)
        term1 = math.lgamma(x0) - x0
        
        # Term 2: log((sin(abs(sin(cos(x0)))) + x0))
        # Note: sin(abs(sin(cos(x0)))) is always in [0, sin(1)] approx [0, 0.84]
        inner_osc = math.sin(abs(math.sin(math.cos(x0))))
        term2 = math.log(inner_osc + x0)
        
        # Term 3: Cosine Modulation
        # Denom: (lgamma(x0) - x0) + sin((sin(x0) + sqrt(x0)))
        denom = (math.lgamma(x0) - x0) + math.sin(math.sin(x0) + math.sqrt(x0))
        if abs(denom) < 1e-9: denom = 1e-9
        
        arg = math.sqrt(x1) / denom
        term3 = math.cos(arg)
        
        log_pred = (term1 + term2) - term3
        return math.exp(log_pred)
        
    except Exception as e:
        return 0

print(f"{'N':<3} | {'Actual':<22} | {'Gen 69':<22} | {'Error %':<10}")
print("-" * 65)

errors = []
for i, n in enumerate(X_RAW):
    actual = Y_RAW[i]
    pred = sota_gen69(n)
    
    if actual > 0:
        error_pct = (pred - actual) / actual * 100
        errors.append(abs(error_pct))
        print(f"{n:<3} | {actual:<22.0f} | {pred:<22.0f} | {error_pct:+.2f}%")
    else:
        print(f"{n:<3} | {actual:<22.0f} | {pred:<22.4f} | N/A")

print("-" * 65)
print(f"MAPE (N=4..27): {np.mean(errors[3:]):.2f}%")
