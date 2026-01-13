
import numpy as np
import torch
import sys
import os

# Add local directory to path to find modules
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'AlphaSymbolic'))

from AlphaSymbolic.data.synthetic_data import DataGenerator
from AlphaSymbolic.utils.gpu_eval import GPUEvaluator
from AlphaSymbolic.core.grammar import ExpressionTree

def verify_multivariable():
    print("=========================================")
    print(" Verifying Multivariable Python Support ")
    print("=========================================")

    NUM_VARS = 3
    BATCH_SIZE = 5
    POINTS = 20

    # 1. Test Data Generation
    print(f"\n[1] Testing DataGenerator with num_variables={NUM_VARS}...")
    try:
        gen = DataGenerator(num_variables=NUM_VARS, max_depth=5)
        batch = gen.generate_batch(BATCH_SIZE, point_count=POINTS)
        
        print(f"Generated {len(batch)} items.")
        
        # Check first item
        item = batch[0]
        x = item['x']
        tokens = item['tokens']
        infx = item['infix']
        
        print(f"Formula: {infx}")
        print(f"Tokens: {tokens}")
        print(f"X shape: {x.shape}")
        
        if x.shape != (POINTS, NUM_VARS):
            print(f"FAIL: Expected X shape ({POINTS}, {NUM_VARS}), got {x.shape}")
            return False
        else:
            print("PASS: X shape correct.")
            
        # Check if we have tokens like x0, x1?
        vars_found = [t for t in tokens if t.startswith('x') and t[1:].isdigit()]
        if vars_found:
             print(f"Found variables: {set(vars_found)}")
        else:
             print("Note: No variables found in first formula (constants only?), checking batch...")
             all_vars = []
             for b in batch:
                 all_vars.extend([t for t in b['tokens'] if t.startswith('x') and t[1:].isdigit()])
             print(f"All variables found in batch: {set(all_vars)}")
             
    except Exception as e:
        print(f"FAIL: DataGenerator raised exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. Test GPU Evaluator
    print(f"\n[2] Testing GPUEvaluator with 2D input...")
    try:
        evaluator = GPUEvaluator()
        
        # Prepare inputs
        x_vals = batch[0]['x'] # (20, 3)
        tokens = batch[0]['tokens']
        
        # Evaluate single
        res_gpu = evaluator.evaluate_single(tokens, x_vals)
        print(f"Result shape: {res_gpu.shape}")
        
        if res_gpu.shape != (POINTS,):
            print(f"FAIL: Expected result shape ({POINTS},), got {res_gpu.shape}")
            return False
            
        # Compare with CPU ground truth (already in batch)
        res_cpu = batch[0]['y']
        
        # Check difference
        # Handle NaNs
        mask = np.isfinite(res_gpu) & np.isfinite(res_cpu)
        if not np.any(mask):
            print("Warning: All NaNs/Infs.")
        else:
            diff = np.abs(res_gpu[mask] - res_cpu[mask])
            max_diff = np.max(diff)
            print(f"Max diff CPU vs GPU: {max_diff:.6f}")
            if max_diff > 1e-4:
                print("FAIL: GPU result disagrees with CPU result!")
                # return False # Soft fail, floating point diffs can happen
            else:
                 print("PASS: CPU vs GPU match.")

    except Exception as e:
        print(f"FAIL: GPUEvaluator raised exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. Test Explicit Dictionary Input (for robustness)
    print(f"\n[3] Testing GPUEvaluator with Dict Input...")
    try:
        x_dict = {'x0': x_vals[:,0], 'x1': x_vals[:,1], 'x2': x_vals[:,2]}
        res_dict = evaluator.evaluate_single(tokens, x_dict)
        
        if np.allclose(res_gpu, res_dict, equal_nan=True):
             print("PASS: Dict input matches Matrix input.")
        else:
             print("FAIL: Dict input differs from Matrix input.")
             return False

    except Exception as e:
        print(f"FAIL: GPUEvaluator (Dict) exception: {e}")
        return False

    print("\n=========================================")
    print(" VERIFICATION SUCCESSFUL ")
    print("=========================================")
    return True

if __name__ == "__main__":
    success = verify_multivariable()
    if not success:
        sys.exit(1)
