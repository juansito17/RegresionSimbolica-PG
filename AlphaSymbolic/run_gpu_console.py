
import sys
import os
import torch
import numpy as np
import time
from core.gpu import TensorGeneticEngine

# Configuration matching C++ Globals
# --- CONFIGURATION ---
# Configuration matching C++ Globals
from core.gpu.config import GpuGlobals

# --- CONFIGURATION ---
TARGETS = GpuGlobals.PROBLEM_Y_FULL

# Generate X_VALUES procedurally to match the pattern from config:
# x0 = start..end
# x1 = x0 % VAR_MOD_X1
# x2 = x0 % VAR_MOD_X2
indices = np.arange(GpuGlobals.PROBLEM_X_START, GpuGlobals.PROBLEM_X_END + 1, dtype=np.float64)
x1_vals = indices % GpuGlobals.VAR_MOD_X1
x2_vals = indices % GpuGlobals.VAR_MOD_X2
X_VALUES = np.column_stack((indices, x1_vals, x2_vals))

def console_mimic_callback(gen, best_rmse, best_rpn_tensor, best_consts_tensor, is_new_best, island_idx=-1):
    """
    Mimics EXACTLY the C++ console output.
    """
    
    # 1. Decode Formula
    # We need access to engine instance to decode? 
    # The callback doesn't have 'self'. We can assume external engine variable or pass it.
    # But RPN decoding is simple if we have the method.
    # We will use the global 'engine' instance defined below.
    
    formula_str = engine.rpn_to_infix(best_rpn_tensor, best_consts_tensor)
    formula_size = engine.get_tree_size(best_rpn_tensor) 

    if is_new_best:
        print(f"\n========================================")
        print(f"New Global Best Found (Gen {gen}, Island {island_idx})")
        print(f"Fitness: {best_rmse:.8f}")
        print(f"Size: {formula_size}")
        print(f"Formula: {formula_str}")
        print("Predictions vs Targets:")
        
        # Show Predictions (Top 5 rows only to avoid spam? C++ showed all X_values)
        # C++ showed all. Let's show all if small, or top 10.
        # Recalculate predictions
        try:
            # We need to run evaluate on the best formula for single points
            # Or just use the batch evaluator on CPU for display?
            # Actually engine has 'rpn_to_infix', we can use ExpressionTree to eval?
            # Or engine.evaluate_batch?
            # engine.evaluate_batch expects a population.
            # Let's use ExpressionTree for clean single-point eval if possible.
            from core.grammar import ExpressionTree
            tree = ExpressionTree.from_infix(formula_str)
            
            # Determine display targets
            display_targets = TARGETS
            if GpuGlobals.USE_LOG_TRANSFORMATION:
                 # Parity with engine filtering if needed, but here we just trans for display
                 # However, to be safe and match engine, we filter too
                 mask = TARGETS > 1e-9
                 display_targets = np.log(np.where(mask, TARGETS, 1.0)) # Safe log for display
            
            for i in range(len(X_VALUES)):
                val = tree.evaluate(X_VALUES[i])
                
                # Ensure val is scalar
                if isinstance(val, np.ndarray):
                    val = val.item() if val.size == 1 else val[0]

                target = display_targets[i] if i < len(display_targets) else float('nan')
                diff = abs(val - target)
                
                # Format: x=(...): Pred=..., Target=..., Diff=... 
                # (Same as C++)
                x_str = ",".join([f"{x:.1f}" for x in X_VALUES[i]])
                
                print(f"  x=({x_str}): Pred={val:12.4f}, Target={target:12.4f}, Diff={diff:12.4f}")
        except Exception as e:
            print(f"  (Error calculating detailed predictions for display: {e})")
        
        print("========================================")
        sys.stdout.flush()
        
    else:
        # Progress Report
        if not hasattr(console_mimic_callback, "last_time"):
             console_mimic_callback.last_time = start_time_global
             console_mimic_callback.last_gen = 0
        
        current_time = time.time()
        delta_t = current_time - console_mimic_callback.last_time
        delta_g = gen - console_mimic_callback.last_gen
        
        instant_speed = (delta_g * engine.pop_size) / delta_t if delta_t > 0 else 0.0
        
        console_mimic_callback.last_time = current_time
        console_mimic_callback.last_gen = gen

        elapsed = time.time() - start_time_global
        print(f"\n--- Gen {gen} (Elapsed: {elapsed:.2f}s) | Instant Speed: {instant_speed:,.0f} Evals/sec ---")
        print(f"Overall Best Fitness: {best_rmse:.4e}")
        print(f"Best Formula Size: {formula_size}")
        sys.stdout.flush()

if __name__ == "__main__":
    print("Starting Genetic Algorithm (GPU Mode)...")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: GPU NOT DETECTED. Running in CPU emulation mode (Slow).")
        

    from core.gpu.config import GpuGlobals
    
    # User can override Globals here
    # GpuGlobals.POP_SIZE override removed to use config default (1M)
    GpuGlobals.NUM_ISLANDS = 20
    GpuGlobals.PROGRESS_REPORT_INTERVAL = 100
    GpuGlobals.USE_PARETO_SELECTION = False  # Disable NSGA-II for speed test
    
    # Engine will use Globals defaults for pop_size and n_islands
    engine = TensorGeneticEngine(num_variables=3) # 3 variables as per new X_VALUES
    
    start_time_global = time.time()
    
    try:
        # Run Infinite Loop (until Ctrl+C or solved)
        # Timeout set to very high (1 hour)
        print("Evaluating initial population...")
        
        # SLICE INPUTS TO MATCH TARGETS (17)
        # And ensure correct shape for num_variables=1
        if engine.num_variables == 1:
            x_input = X_VALUES[:len(TARGETS), 0]
        else:
            x_input = X_VALUES[:len(TARGETS)]
        
        seeds = []
        if GpuGlobals.USE_INITIAL_FORMULA and GpuGlobals.INITIAL_FORMULA_STRING:
            seeds.append(GpuGlobals.INITIAL_FORMULA_STRING)
            print(f"Info: Injecting initial formula: {GpuGlobals.INITIAL_FORMULA_STRING}")

        final_formula = engine.run(
            x_input, 
            TARGETS, 
            seeds=seeds, 
            timeout_sec=3600, 
            callback=console_mimic_callback
        )

        
        print("\nSearch Finished.")
        if final_formula:
            print(f"Final Result: {final_formula}")
            
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")
