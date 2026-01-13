
import torch
import numpy as np
import math
from core.grammar import ExpressionTree
from core.gpu.config import GpuGlobals
from core.gpu.engine import TensorGeneticEngine

def test_reproduction():
    # 1. Setup Data as in solve_n_queens.py
    Y_RAW_LIST = [
        1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 
        2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884, 
        314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352, 22317699616364044, 234907967154122528
    ]
    Y_RAW = np.array(Y_RAW_LIST, dtype=np.float64)
    
    indices = np.arange(1, 27 + 1, dtype=np.float64)
    x1_vals = indices % 6
    x2_vals = indices % 2
    X_VALUES = np.column_stack((indices, x1_vals, x2_vals))
    
    START_INDEX = 3 # n=4
    X_TRAIN = X_VALUES[START_INDEX:]
    Y_TRAIN = Y_RAW[START_INDEX:]
    

    # 2. Formula from Gen 1 (New SOTA 0.128)
    formula_str = "((lgamma((1 + x0)) - cos((x1 / ((0.13184391 * x0) ^ x0)))) - x0)"
    
    print(f"Formula: {formula_str}")
    
    # 3. Method A: ExpressionTree (Python)
    print("\n--- Method A: ExpressionTree (Python Evaluation) ---")
    tree = ExpressionTree.from_infix(formula_str)
    
    mae_log = 0
    mse_log = 0
    count = 0
    
    for i in range(len(X_TRAIN)):
        x_row = X_TRAIN[i]
        y_val = Y_TRAIN[i]
        
        # Predict
        # Note: ExpressionTree evaluates with standard math

        try:
            pred = tree.evaluate(x_row)
            if hasattr(pred, 'item'): pred = pred.item()
        except Exception as e:
            pred = float('nan')

            
        # Target Log
        log_y = math.log(y_val)
        
        diff = pred - log_y
        mse_log += diff ** 2
        count += 1
        
        if i % 5 == 0 or i == len(X_TRAIN)-1:
            print(f"n={x_row[0]:.0f} | LogY={log_y:.4f} | Pred={pred:.4f} | Diff={diff:.4f}")
            
    rmse_tree = math.sqrt(mse_log / count)
    print(f"Tree RMSE: {rmse_tree:.8f}")
    
    # 4. Method B: GPU Engine (CUDA VM)
    print("\n--- Method B: GPU Engine (CUDA VM Evaluation) ---")
    
    # Configure Globals
    GpuGlobals.USE_LOG_TRANSFORMATION = True
    GpuGlobals.LOSS_FUNCTION = 'RMSE'
    
    # Setup Engine
    engine = TensorGeneticEngine(num_variables=3)
    
    # Prepare Tensors
    # engine.run logic: Log transform Y
    
    x_t = torch.tensor(X_TRAIN, dtype=engine.dtype, device=engine.device)
    y_t = torch.tensor(Y_TRAIN, dtype=engine.dtype, device=engine.device)
    
    # Log transform
    mask = y_t > 1e-9
    y_t_log = torch.log(y_t[mask])
    x_t = x_t[mask]
    

    # Parse Formula to RPN
    pop_rpn, pop_consts = engine.load_population_from_strings([formula_str])
    
    # Direct VM Call to inspect NaNs
    print("\n--- Direct VM Inspection ---")
    x_for_vm = x_t.T # [Vars, Samples]
    preds, sp, err = engine.evaluator._run_vm(pop_rpn, x_for_vm, pop_consts)
    
    print(f"Preds shape: {preds.shape}")
    print(f"Preds first 5: {preds[0, :5]}")
    print(f"Has NaN: {torch.isnan(preds).any().item()}")
    print(f"Has Inf: {torch.isinf(preds).any().item()}")
    
    # Evaluate Batch (Original metric)
    fitness = engine.evaluate_batch(pop_rpn, x_t, y_t_log, pop_consts)
    print(f"Engine RMSE: {fitness.item():.8f}")

    
    return
    
if __name__ == "__main__":
    test_reproduction()
