import numpy as np
import math
from math import factorial, gcd, log, cos, exp, pi
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Dataset (N=1..27)
Y_RAW = np.array([
    1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 
    2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 
    314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528
], dtype=np.float64)

X_RAW = np.arange(1, 28)

# Function to calculate phi_KQ (Count of valid linear slopes in Z_n)
# condition: gcd(k, n)=1 AND gcd(1-k, n)=1 AND gcd(1+k, n)=1
def phi_kq(n):
    if n <= 3: return 0 # Trivial handling for small n where logic might break
    count = 0
    # k ranges from 2 to n-2 (slopes 0, 1, -1 are never valid for queens on torus generally, 
    # though 1 and -1 are diagonals. The paper says k in {2...n-2})
    for k in range(2, n-1):
        if gcd(k, n) == 1 and gcd(n - k, n) == 1 and gcd(n + k, n) == 1:
            # Note: gcd(n-k, n) is same as gcd(-k, n) -> gcd(k,n).
            # But the condition for diagonal conflicts is:
            # y = kx + b
            # diag1: y - x = (k-1)x + b -> need gcd(k-1, n)=1
            # diag2: y + x = (k+1)x + b -> need gcd(k+1, n)=1
            # So checks are gcd(k, n)=1, gcd(k-1, n)=1, gcd(k+1, n)=1
            if gcd(k-1, n) == 1 and gcd(k+1, n) == 1:
                count += 1
    return count

# Precompute structural features to speed up optimization
print("Precomputing Topological Features...")
PHI_KQ_CACHE = {n: phi_kq(n) for n in X_RAW}
IS_PRIME_CACHE = {n: all(n % i != 0 for i in range(2, int(n**0.5) + 1)) and n > 1 for n in X_RAW}

def is_prime(n):
    return IS_PRIME_CACHE[n]

class UnifiedFieldTheory:
    def __init__(self, params):
        # Unpack parameters
        # C0: Global scaling constant
        # c_simkin: Entropic compression (approx 0.94-0.97 validated)
        # A: Amplitude of Prime Resonance (Exponential)
        # B: Amplitude of Modular Density Resonance (phi_KQ)
        # gamma: Fractal dimension exponent
        # omega: Fractal oscillation frequency
        # phase: Fractal oscillation phase
        self.C0, self.c_simkin, self.A, self.B, self.gamma, self.omega, self.phase = params
        
    def predict(self, n):
        if n == 1: return 1.0 
        if n == 2: return 0.0
        if n == 3: return 0.0
        
        # 1. Combinatorial Base ~ N!
        # working in log space to avoid overflow for intermediate calc, but final result is float
        # Gamma(n+1) = n!
        log_fact = math.lgamma(n + 1)
        
        # 2. Entropic Compression ~ e^(-c * n)
        log_entropy = -self.c_simkin * n
        
        # 3. Modular Resonance (Exponential Form)
        # K_mod = exp( A * I_prime / n + B * phi_kq / n )
        mod_density = PHI_KQ_CACHE[n] / n
        prime_term = 1.0 if is_prime(n) else 0.0
        
        log_modular = (self.A * prime_term / n) + (self.B * mod_density)
        
        # 4. Fractal Oscillation
        # Term: N^gamma * exp( osc_amp * cos(...) ) ?
        # Or just multiplicative: * (1 + cos...)
        # Let's use exponential modulation for safety and consistency with field theory
        # log_fractal = gamma * log(N) + log(1 + ...) 
        # For simplicity/stability: correction = N^gamma * exp(cos(...))
        
        log_power = self.gamma * math.log(n)
        
        # Fractal Oscillation Component
        # We'll treat the cosine as a small perturbation in the exponent (Action)
        oscillation = math.cos(math.log(n) * self.omega + self.phase)
        # We need an amplitude for the oscillation. 
        # Implicitly, let's say the oscillation amplitude is small/fixed or let B absorb it?
        # Let's add an explicit amplitude if needed, but for now let's say it's part of the fractal term.
        # Actually, let's reuse A or B or just stick to the text "N^(delta-1) cos".
        # Let's add an amplitude param 'alpha_osc'. 
        # But we correspond to the implementation plan: just C0, c, A, gamma, omega, phase.
        # We'll assume the oscillation is multiplied: * (1 + 0.1 cos) roughly.
        # Let's compute in non-log for the cosine to be safe
        
        # Sum of Logs
        total_log = math.log(self.C0) + log_fact + log_entropy + log_modular + log_power
        
        try:
            pred = math.exp(total_log)
            # Apply oscillation multiplicatively (avoiding <=0)
            # 1 + 0.05 * cos(...)
            pred *= (1.0 + 0.05 * oscillation)
            return pred
        except OverflowError:
            return float('inf')

def objective(params):
    # Loss function: RMSLE (Root Mean Squared Log Errors)
    # We focus on N=8..27 (where data is stable and non-zero)
    start_idx = 7 # N=8
    model = UnifiedFieldTheory(params)
    
    errors = []
    for i in range(start_idx, len(X_RAW)):
        n = X_RAW[i]
        y_true = Y_RAW[i]
        y_pred = model.predict(n)
        
        if y_pred <= 1e-9: return 1e9 # Penalty
        
        log_diff = math.log(y_true) - math.log(y_pred)
        errors.append(log_diff ** 2)
        
    return np.mean(errors) 

# ---------------------------------------------------------
# OPTIMIZATION
# ---------------------------------------------------------
try:
    from scipy.optimize import minimize
    
    # Initial Guesses
    # [C0, c_simkin, A, B, gamma, omega, phase]
    x0 = [2.0, 0.94, 1.0, 1.0, 0.0, 10.0, 0.0] 
    
    print("Optimizing Unified Field Parameters (Exponential Form)...")
    res = minimize(objective, x0, method='Nelder-Mead', tol=1e-5, options={'maxiter': 5000})

    
    best_params = res.x
    print(f"Optimization Success: {res.success}")
    print(f"Best Params: {best_params}")
    
    # Validation Run
    model = UnifiedFieldTheory(best_params)
    
    print("\n----------------------------------------------------------------")
    print(f"{'N':<3} | {'Actual Q(N)':<20} | {'Predicted Q(N)':<20} | {'LogErr':<8} | {'% Error':<8}")
    print("----------------------------------------------------------------")
    
    total_log_sq_err = 0
    count = 0
    
    for i in range(len(X_RAW)):
        n = X_RAW[i]
        y_true = Y_RAW[i]
        y_pred = model.predict(n)
        
        if y_true > 0 and y_pred > 0:
            log_err = math.log(y_true) - math.log(y_pred)
            pct_err = 100 * (y_pred - y_true) / y_true
            
            # Formatting
            y_true_s = f"{y_true:.2e}" if y_true > 1e9 else f"{int(y_true)}"
            y_pred_s = f"{y_pred:.2e}" if y_pred > 1e9 else f"{int(y_pred)}"
            
            if n >= 4: # Only count stats for N>=4
                total_log_sq_err += log_err ** 2
                count += 1
                
            print(f"{n:<3} | {y_true_s:<20} | {y_pred_s:<20} | {log_err:+.4f}   | {pct_err:+.1f}%")
        else:
             print(f"{n:<3} | {int(y_true):<20} | {y_pred:.2f}")

    rmsle = np.sqrt(total_log_sq_err / count) if count > 0 else 0
    print("----------------------------------------------------------------")
    print(f"Final RMSLE (N=4..27): {rmsle:.6f}")
    
except ImportError:
    print("Scipy not found. Please install scipy to run optimization.")
except Exception as e:
    print(f"An error occurred: {e}")
