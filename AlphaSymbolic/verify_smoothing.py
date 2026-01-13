
import math
import numpy as np
from scipy.special import gamma

Y_RAW = [
    1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 
    2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 
    314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528
]

def safe_gamma(x):
    return gamma(abs(x) + 1e-10)

def evaluate_smooth(n):
    x0 = float(n)
    x1 = float(n % 6)
    x2 = float(n % 2)
    
    # New Constants (Smoothed)
    k1 = 0.10944318152432879
    k2 = -0.17501817532234493
    
    try:
        term1 = math.lgamma(x0 + 2)
        g_x2 = safe_gamma(x2)
        sin_g = math.sin(g_x2)
        if abs(sin_g) < 1e-10: sin_g = 1e-10 
        exponent = math.sin(k2) + (x0 / sin_g) + x0 + x2
        base = k1 * x0
        if base <= 0: denom = 1e-10 
        else: denom = base ** exponent
        if denom == 0: cos_arg = 0
        else: cos_arg = x1 / denom
        term2 = math.cos(cos_arg)
        log_pred = term1 - term2 - x0
        pred = math.exp(log_pred)
        return pred
    except Exception as e:
        return float('nan')

with open("smoothed_report.txt", "w") as f:
    f.write(f"{'N':<4} | {'Target':<22} | {'Prediction':<22} | {'Error %':<10} | {'Log Diff'}\n")
    f.write("-" * 80 + "\n")

    for i, target in enumerate(Y_RAW):
        n = i + 1
        if target == 0: 
            f.write(f"{n:<4} | {target:<22} | {'---':<22} | {'---':<10} | ---\n")
            continue
            
        pred = evaluate_smooth(n)
        
        if math.isnan(pred):
            f.write(f"{n:<4} | {target:<22} | {'NaN':<22} | {'NaN':<10}\n")
            continue
            
        error_percent = abs(pred - target) / target * 100
        log_diff = abs(math.log(pred) - math.log(target))
        
        f.write(f"{n:<4} | {target:<22} | {pred:<22.2f} | {error_percent:<9.4f}% | {log_diff:.4f}\n")
