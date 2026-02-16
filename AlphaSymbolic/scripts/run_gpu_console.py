import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
indices = np.array(GpuGlobals.PROBLEM_X_FILTERED, dtype=np.float64)
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
    
    display_rpn = best_rpn_tensor
    display_consts = best_consts_tensor
    if GpuGlobals.USE_CONSOLE_BEST_SIMPLIFICATION:
        try:
            simp_pop, simp_consts, _ = engine.gpu_simplifier.simplify_batch(
                best_rpn_tensor.unsqueeze(0),
                best_consts_tensor.unsqueeze(0),
                max_passes=10
            )
            if simp_pop is not None and simp_pop.shape[0] > 0:
                display_rpn = simp_pop[0]
                if simp_consts is not None and simp_consts.shape[0] > 0:
                    display_consts = simp_consts[0]
        except Exception:
            pass

    formula_str = engine.rpn_to_infix(display_rpn, display_consts)
    formula_size = engine.get_tree_size(display_rpn)

    if is_new_best:
        print(f"\n========================================")
        print(f"New Global Best Found (Gen {gen}, Island {island_idx})")
        print(f"Fitness: {best_rmse:.8f}")
        print(f"Size: {formula_size}")
        print(f"Formula: {formula_str}")
        
        # --- Strict Validation Check ---
        try:
            # Prepare Data (subset used for training)
            # Need to match what was passed to engine.run()
            n_targets = len(TARGETS)
            if engine.num_variables == 1:
                x_val_np = X_VALUES[:n_targets, 0] # [N]
                x_val_np = x_val_np.reshape(-1, 1) # [N, 1]
            else:
                x_val_np = X_VALUES[:n_targets] # [N, Vars]
            
            # Convert to Tensor [Vars, N]
            x_tensor = torch.tensor(x_val_np, dtype=engine.dtype, device=engine.device).T
            
            # Convert Y to Tensor [N]
            y_tensor = torch.tensor(TARGETS, dtype=engine.dtype, device=engine.device)
            if y_tensor.dim() == 1:
                 y_tensor = y_tensor
            
            # Run Strict Validation
            # population: [1, L], constants: [1, K]
            val_res = engine.evaluator.validate_strict(
                display_rpn.unsqueeze(0),
                x_tensor,
                y_tensor,
                display_consts.unsqueeze(0)
            )
            
            if not val_res['is_valid'][0].item():
                n_err = val_res['n_errors'][0].item()
                print(f"[STRICT MODE] WARNING: Formula has {n_err} domain errors (e.g. log(neg)). Valid on GPU-Protected only.")
                
        except Exception as e:
            print(f"[STRICT MODE] Check Failed: {e}")

        print("Predictions vs Targets:")
        
        # Show Predictions (Top 5 rows only to avoid spam? C++ showed all X_values)
        # C++ showed all. Let's show all if small, or top 10.
        # Recalculate predictions
        # Use GPU for vectorized predictions
        # X_VALUES: [N, Vars] -> engine.predict_individual expects x as [D] or [N, D] handled by engine or evaluator?
        # engine.predict_individual(rpn, consts, x) wraps as batch of 1 and returns preds for all x.
        # We convert X_VALUES to tensor first.
        try:
            x_tensor = torch.tensor(X_VALUES, dtype=engine.dtype, device=engine.device)
            preds_tensor = engine.predict_individual(display_rpn, display_consts, x_tensor)
            preds = preds_tensor.detach().cpu().numpy().flatten()
            
            # Determine display targets
            display_targets = TARGETS
            if GpuGlobals.USE_LOG_TRANSFORMATION:
                 mask = TARGETS > 1e-9
                 display_targets = np.log(np.where(mask, TARGETS, 1.0))
            
            for i in range(len(X_VALUES)):
                val = preds[i]
                target = display_targets[i] if i < len(display_targets) else float('nan')
                diff = abs(val - target)
                
                x_str = ",".join([f"{x:.1f}" for x in X_VALUES[i]])
                print(f"  x=({x_str}): Pred={val:12.4f}, Target={target:12.4f}, Diff={diff:12.4f}")
        except Exception as e:
            print(f"  (Error calculating vectorized predictions on GPU: {e})")
        
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
    # GpuGlobals.POP_SIZE and GpuGlobals.NUM_ISLANDS are now strictly taken from config.py 
    # to ensure consistency with the optimized stress test results.
    # GpuGlobals.NUM_ISLANDS = 40 <--- REMOVED (Uses config.py value)
    
    GpuGlobals.PROGRESS_REPORT_INTERVAL = 100
    # GpuGlobals.USE_PARETO_SELECTION = False  # Removed override to respect config.py
    
    # Engine will use Globals defaults for pop_size and n_islands
    # INCREASED max_constants to 10 to support complex seeds with many literals
    engine = TensorGeneticEngine(num_variables=3, max_constants=GpuGlobals.MAX_CONSTANTS)
    
    start_time_global = time.time()
    
    try:
        # Run Infinite Loop (until Ctrl+C or solved)
        # Timeout disabled (None)
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
            timeout_sec=None,  # Infinite time
            callback=console_mimic_callback
        )

        
        print("\nSearch Finished.")
        if final_formula:
            print(f"Final Result: {final_formula}")
            
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")
