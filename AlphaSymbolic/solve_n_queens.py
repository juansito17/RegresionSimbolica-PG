
import sys
import os
import torch
import numpy as np
import time
import math
from core.gpu import TensorGeneticEngine
from core.gpu.config import GpuGlobals
from core.grammar import ExpressionTree

# ============================================================
#                  SOTA N-Queens Solver Config
# ============================================================

# 1. OVERRIDE GLOBALS FOR MAX PERFORMANCE
# ----------------------------------------
GpuGlobals.POP_SIZE = 4_000_000      # MAX STABLE for RTX 3050 (Chunked)
GpuGlobals.NUM_ISLANDS = 40          # 100k per island
GpuGlobals.GENERATIONS = 100_000     # Effectively infinite

# Operators specific for Combinatorial/Number Theory Sequences
GpuGlobals.USE_OP_FACT = True        # Factorial is key for N-Queens (n!)
GpuGlobals.USE_OP_MOD = True         # Modulo helpful for periodic patterns
GpuGlobals.USE_LOG_TRANSFORMATION = True # Explicitly enabled (critical for exponential growth)
GpuGlobals.USE_OP_LOG = True         # Log transformation enabled
GpuGlobals.USE_OP_EXP = True
GpuGlobals.USE_OP_POW = True
GpuGlobals.USE_OP_SIN = True         # Sometimes useful for periodic correction
GpuGlobals.USE_OP_COS = True
GpuGlobals.USE_OP_FLOOR = True       # Integer sequences often need floor
GpuGlobals.USE_OP_GAMMA = True       # Gamma function for generalized factorial

# Constants
GpuGlobals.FORCE_INTEGER_CONSTANTS = False 
GpuGlobals.CONSTANT_MIN_VALUE = -10.0
GpuGlobals.CONSTANT_MAX_VALUE = 10.0
# Inject SOTA Constants explicitly
# 0.143 (Simkin), 2.54 (Conjecture), 1.942 (Simkin Exponent)
GpuGlobals.CUSTOM_CONSTANTS = [0.143, 2.54, 1.942, 1.0, 2.0, 0.5, math.pi, math.e]

# ----------------------------------------
# DATA PREPARATION (OEIS A000170)
# ----------------------------------------
# n=1..27 (Using extended data from OEIS)
# 1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528
Y_RAW = np.array([
    1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 
    2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 
    314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528
], dtype=np.float64)

X_START = 1
X_END = 27
indices = np.arange(X_START, X_END + 1, dtype=np.float64)

# Create Inputs (x0=n, x1=n%6, x2=n%2)
x1_vals = indices % 6
x2_vals = indices % 2 # Parity
X_VALUES = np.column_stack((indices, x1_vals, x2_vals))

# FILTERING: Start from n=4 to avoid 0s (problematic for Log transform)
# Indices 0, 1, 2 correspond to n=1, 2, 3.
# n=2,3 are 0.
START_INDEX = 3 # Start from n=4 (Index 3)
X_TRAIN = X_VALUES[START_INDEX:]
Y_TRAIN = Y_RAW[START_INDEX:]

print(f"Loaded {len(Y_RAW)} samples.")
print(f"Training on n={X_TRAIN[0,0]}..{X_TRAIN[-1,0]} ({len(Y_TRAIN)} samples) to avoid zeros.")

# ----------------------------------------
# SEEDS (Simkin, Factorial, Parity)
# ----------------------------------------
seeds = []

# 1. Simkin: (0.143 * n)^n -> log(...)
seeds.append("log(((0.143 * x0) ** x0))")

# 2. Factorial Conjecture: n! / 2.54^n -> log(...)
seeds.append("log(lgamma(x0 + 1) / (2.54 ** x0))")

# 3. Parity Split (using x2): If Even (x2=0) use A, if Odd (x2=1) use B
# Combining Simkin with Parity split structure -> log(...)
seeds.append("log((x2 * ((0.143 * x0) ** x0)) + ((1 - x2) * ((0.143 * x0) ** x0)))")

# 4. Bell's Approximation (n! / c^n) -> log(...)
seeds.append("log(lgamma(x0 + 1) / (2.54 ** x0))")

# 5. Smoothed Gen 8/7 - Optimized for Reduced Spikes (N=16 600%->170%. N=26 29%->4%)
# Trade-off: N=9 Error increased. Better Asymptotic Stability.
#seeds.append("((lgamma((x0 + 2)) - cos((x1 / ((0.11575544 * x0) ^ (((sin(-0.18788845) + (x0 / sin(1))) + x0) + x2))))) - (x0 ^ cos(0.0271873)))")

# 6. Peña-Usuga Asymptotic Formula (The "Golden Seed")
# Accuracy: <0.01% Error for N >= 24. 
# Implements Parity Switch: 
#   Even (x2=0): A=0.945525, B=0.966099
#   Odd  (x2=1): A=0.943389, B=0.911941
# Formula: lgamma(N+1) - A*N + B
seeds.append("lgamma(x0 + 1) - (((0.945525 * (1 - x2)) + (0.943389 * x2)) * x0) + ((0.966099 * (1 - x2)) + (0.911941 * x2))")

GpuGlobals.USE_INITIAL_FORMULA = True

# ============================================================
#                  CALLBACK & REPORTING
# ============================================================

start_time_global = time.time()

def console_callback(gen, best_rmse, best_rpn, best_consts, is_new_best, island_idx=-1):
    if is_new_best:
        formula_str = engine.rpn_to_infix(best_rpn, best_consts)
        size = engine.get_tree_size(best_rpn)
        
        print(f"\n>>> NEW SOTA FOUND (Gen {gen}) <<<")
        print(f"Fitness (RMSE/RMSLE): {best_rmse:.8f}")
        print(f"Formula: {formula_str}")
        
        # Validation
        try:
            tree = ExpressionTree.from_infix(formula_str)
            # Verification output suppressed as requested
            pass
        except Exception as e:
            pass
        sys.stdout.flush()
        
    elif gen % 100 == 0:
        elapsed = time.time() - start_time_global
        speed = (gen * GpuGlobals.POP_SIZE) / elapsed if elapsed > 0 else 0
        print(f"Gen {gen} | Time: {elapsed:.1f}s | Speed: {speed:.0f} evals/s | Best: {best_rmse:.6f}")

# ============================================================
#                  MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("Initializing SOTA GPU Engine for N-Queens...")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Population: {GpuGlobals.POP_SIZE:,}")
    print(f"Islands: {GpuGlobals.NUM_ISLANDS}")
    print(f"Parity Variable (x2) Enabled: YES")
    print(f"SOTA Seeds Injected: {len(seeds)}")
    
    engine = TensorGeneticEngine(num_variables=3)
    
    print("Starting Infinite Search...")
    try:
        engine.run(
            X_TRAIN, 
            Y_TRAIN, 
            seeds=seeds,
            timeout_sec=None, 
            callback=console_callback
        )
    except KeyboardInterrupt:
        print("\nSearch Stopped.")
