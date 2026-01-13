
import torch
import numpy as np
import math
from core.gpu import TensorGeneticEngine
from core.gpu.config import GpuGlobals
from core.grammar import ExpressionTree

# Override Globals for this specific formula test
GpuGlobals.USE_OP_FACT = True
GpuGlobals.USE_OP_GAMMA = True
GpuGlobals.USE_OP_COS = True
GpuGlobals.USE_OP_SIN = True
GpuGlobals.USE_OP_POW = True


# 1. Setup Data
X_START = 4
X_END = 27
indices = np.arange(X_START, X_END + 1, dtype=np.float64)
x1_vals = indices % 6
x2_vals = indices % 2
X_TRAIN = np.column_stack((indices, x1_vals, x2_vals))
Y_RAW = [
    1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 
    2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 
    314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528
]
Y_TRAIN = np.array(Y_RAW[3:], dtype=np.float64)

# Formula
# Formula (Seed 6 - Gen 25)
formula_str = "((lgamma((cos((cos(lgamma(x0 + 1)) / sqrt(x0))) + x0)) - cos(((x0 + x0) / ((0.3046774 * (lgamma(x0 + 1) - x0)) ** x0)))) - x0)"

print(f"Testing Formula: {formula_str}")

# 2. Evaluate with ExpressionTree (CPU)
print("\n--- CPU Evaluation (ExpressionTree) ---")
try:
    tree = ExpressionTree.from_infix(formula_str)
    # manual eval for sample 0 (N=4) and -1 (N=27)
    
    # N=4
    # For single sample evaluation, need to pass numpy arrays of shape (1,) or scalars inside dict if supported?
    # ExpressionTree.evaluate expects dict of arrays or scalars.
    vars_map_4 = {'x0': np.array([4.0]), 'x1': np.array([4.0%6]), 'x2': np.array([4.0%2])}
    val_4 = tree.evaluate(vars_map_4)[0]
    print(f"N=4: Pred (Log) = {val_4:.4f}")
    
    # N=27
    vars_map_27 = {'x0': np.array([27.0]), 'x1': np.array([27.0%6]), 'x2': np.array([27.0%2])}
    val_27 = tree.evaluate(vars_map_27)[0]
    print(f"N=27: Pred (Log) = {val_27:.4f}")
except Exception as e:
    print(f"CPU Error: {e}")

# 3. Evaluate with GPU Engine
print("\n--- GPU Evaluation (TensorGeneticEngine) ---")
try:
    engine = TensorGeneticEngine(num_variables=3)
    
    # Needs to be pop of 1
    # Convert formula to RPN
    # We can use load_population_from_strings
    pop, consts = engine.load_population_from_strings([formula_str])
    
    if pop is None:
        print("Failed to parse formula for GPU")
    else:
        # Prepare Data
        x_cuda = torch.tensor(X_TRAIN, dtype=engine.dtype, device=engine.device).T # [Vars, Samples]
        y_cuda = torch.tensor(Y_TRAIN, dtype=engine.dtype, device=engine.device)
        
        # Log Transform Y if config says so
        if GpuGlobals.USE_LOG_TRANSFORMATION:
            y_cuda = torch.log(y_cuda)
            
        print(f"GPU Log Target N=27: {y_cuda[-1].item():.4f}")

        # Run VM
        # evaluate_batch returns RMSE/Fitness, we want raw predictions
        # Use internal VM
        
        preds, sp, err = engine.evaluator._run_vm(pop, x_cuda, consts)
        
        # preds is [Pop*Samples] -> [1*Samples]
        preds = preds.flatten()
        
        print(f"N=4 (Index 0): GPU Pred = {preds[0].item():.4f}")
        print(f"N=27 (Index -1): GPU Pred = {preds[-1].item():.4f}")
        
        # Check specific term calculation if possible? No, VM is black box here.
        
except Exception as e:
    print(f"GPU Error: {e}")
