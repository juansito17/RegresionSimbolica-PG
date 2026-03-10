import time
import numpy as np
import torch
import pandas as pd
import os
import random
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from AlphaSymbolic.search.hybrid_search import hybrid_solve
# from AlphaSymbolic.ui.app_core import get_model (REMOVED: No NN)
from AlphaSymbolic.core.grammar import ExpressionTree
from AlphaSymbolic.utils.optimize_constants import optimize_constants, substitute_constants, convert_and_extract_constants
from AlphaSymbolic.search.phase_manager import PhaseManager

# --- CONFIGURATION ---
from AlphaSymbolic.core.gpu.config import GpuGlobals
X_FULL = np.array(GpuGlobals.PROBLEM_X_FILTERED, dtype=np.float64)
Y_FULL = GpuGlobals.PROBLEM_Y_FULL

# Targets for Extrapolation
X_TARGETS = np.array([26, 27], dtype=np.float64)
Y_TARGETS = np.array([22317699616364044, 234907967154122528], dtype=np.float64)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'results', 'oeis')
os.makedirs(DATA_DIR, exist_ok=True)

CSV_FILE = os.path.join(DATA_DIR, "top_formulas.csv")
PATTERN_FILE = os.path.join(DATA_DIR, "pattern_memory.json")
TOP_K = 10
MIN_SAMPLE_SIZE = 6 # > 5

import json

# --- PATTERN MEMORY ("La Biblioteca") ---
def extract_structural_skeleton(formula_str):
    """
    Parses formula and replaces all numeric constants with 'C'.
    Returns the structural skeleton (infix).
    """
    try:
        from AlphaSymbolic.core.grammar import ExpressionTree, Node
        tree = ExpressionTree.from_infix(formula_str)
        if not tree.is_valid: return None
        
        def transform(node):
            if not node: return
            # If leaf is number, make it C
            # How to detect number? 
            # In ExpressionTree, numbers are just values.
            # Check if value is numeric string
            try:
                float(node.value)
                node.value = 'C'
            except:
                pass # Operator or Variable
            
            for child in node.children:
                transform(child)
                
        transform(tree.root)
        return tree.root.to_infix()
    except:
        return None

def load_pattern_memory():
    if os.path.exists(PATTERN_FILE):
        try:
            with open(PATTERN_FILE, 'r') as f:
                return json.load(f)
        except: return {}
    return {}

def save_pattern_memory(memory):
    try:
        with open(PATTERN_FILE, 'w') as f:
            json.dump(memory, f, indent=2)
    except: pass

def update_pattern_memory(memory, formula_str):
    skeleton = extract_structural_skeleton(formula_str)
    if skeleton:
        count = memory.get(skeleton, 0)
        memory[skeleton] = count + 1
        return True
    return False

def mutate_formula_structurally(formula_str, max_dist=2):
    """
    Takes a formula, finds a random subtree, and replaces it with a random one.
    This creates structural diversity from elites.
    """
    try:
        tree = ExpressionTree.from_infix(formula_str)
        if not tree or not tree.is_valid: return formula_str
        
        # 1. Collect all nodes
        nodes = []
        def collect(n, p, idx):
            nodes.append((n, p, idx))
            for i, child in enumerate(n.children):
                collect(child, n, i)
        
        collect(tree.root, None, -1)
        
        # 2. Pick a random node to swap (avoid identity root swap if possible)
        target_node, parent, child_idx = random.choice(nodes)
        
        # 3. Generate a small random replacement
        # Use a depth that won't bloat the formula too much
        replacement = ExpressionTree.generate_random(max_depth=2, num_variables=3, p_terminal=0.4).root
        
        if parent is None:
            tree.root = replacement
        else:
            parent.children[child_idx] = replacement
            
        return tree.root.to_infix()
    except:
        return formula_str


def load_or_create_top_list():
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            records = df.to_dict('records')
            # Ensure it's sorted by score immediately after loading
            records.sort(key=lambda x: x.get('rmsle_global', float('inf')))
            return records
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return []
    return []

def save_top_list(top_list):
    df = pd.DataFrame(top_list)
    # Sort by RMSLE (Global Fit) - Match main loop priority
    # Use a safe sort that handles missing values if any
    if 'rmsle_global' in df.columns:
        df = df.sort_values(by='rmsle_global', ascending=True)
    else:
        df = df.sort_values(by='extrapolation_error', ascending=True)
    df.to_csv(CSV_FILE, index=False)
    print(f"Saved Top {len(df)} to {CSV_FILE}")
    
    # Auto-backup if in Colab
    backup_to_drive()

def backup_to_drive():
    """
    If running in Google Colab, copies the top list and pattern memory to Google Drive.
    """
    try:
        import shutil
        if os.path.exists('/content/drive/MyDrive'):
            drive_path = '/content/drive/MyDrive/AlphaSymbolic_Models'
            os.makedirs(drive_path, exist_ok=True)
            
            # Files to backup
            files = [CSV_FILE, PATTERN_FILE]
            for f in files:
                if os.path.exists(f):
                    shutil.copy(f, os.path.join(drive_path, f))
            # print("  [Backup] Synced to Google Drive.")
    except Exception as e:
        # Silently fail if drive not mounted or other issues
        pass

def evaluate_and_save(full_formula_str, residual_formula_str, top_formulas, pattern_memory, origin="Search", time_taken=0):
    """
    Evaluates a candidate formula string on the full dataset (History + Targets),
    calculates metrics, and saves to top_formulas if good.
    """
    try:
        tree = ExpressionTree.from_infix(full_formula_str)
        if not tree.is_valid: return False
        
        # Combine all points for validation
        x_all_raw = np.concatenate((X_FULL, X_TARGETS))
        y_all = np.concatenate((Y_FULL, Y_TARGETS))
        
        # Construct feature matrix for evaluation (3 variables)
        x_all = np.zeros((len(x_all_raw), 3), dtype=np.float64)
        x_all[:, 0] = x_all_raw
        x_all[:, 1] = x_all_raw % GpuGlobals.VAR_MOD_X1
        x_all[:, 2] = x_all_raw % GpuGlobals.VAR_MOD_X2
        
        y_pred_all = tree.evaluate(x_all)
        y_pred_safe = np.maximum(y_pred_all, 0)
        
        # RMSLE
        log_error = np.sqrt(np.mean((np.log1p(y_pred_safe) - np.log1p(y_all))**2))
        
        # Extrapolation Error
        pred_26 = y_pred_all[-2]
        pred_27 = y_pred_all[-1]
        extrap_error_sum = abs(pred_26 - Y_TARGETS[0]) + abs(pred_27 - Y_TARGETS[1])
        
        print(f"\n[SUCCESS] [{origin}] Formula: {full_formula_str}\n          RMSLE (1-27): {log_error:.6f} | Extrap Error: {extrap_error_sum:.2e}")
        
        # Update Top List
        entry = {
            'formula': full_formula_str,
            'residual': residual_formula_str,
            'rmsle_global': log_error,
            'extrapolation_error': extrap_error_sum,
            'pred_26': pred_26,
            'pred_27': pred_27,
            'sample_size': origin, # Using origin string which contains 'Search(k=...)' or similar
            'time': time_taken,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        top_formulas.append(entry)
        
        unique = {d['formula']: d for d in top_formulas}
        top_formulas[:] = list(unique.values()) # In-place update
        
        top_formulas.sort(key=lambda x: x.get('rmsle_global', float('inf')))
        
        if len(top_formulas) > TOP_K:
            top_formulas[:] = top_formulas[:TOP_K] 
            
        save_top_list(top_formulas)
        
        # Feedback on the current standing
        best_now = top_formulas[0].get('rmsle_global', 1.0)
        print(f"  [Top List] Current Global Best RMSLE: {best_now:.6f} ({len(top_formulas)} formulas tracked)")
        
        if update_pattern_memory(pattern_memory, full_formula_str):
             save_pattern_memory(pattern_memory)
             backup_to_drive()
             
        return True
    except Exception as e:
        print(f"Eval Error: {e}")
        return False

def run_phase_search(iteration, top_formulas, phase_manager, model, device, pattern_memory):
    """
    Attempt to find phases (Split & Conquer) based on the current best formula's residuals.
    """
    if not top_formulas: return
    
    # Run every 5 iterations or if forced
    if iteration % 5 != 0: return

    print("\n--- [Phase Detection] Checking for regime changes ---")
    
    # 1. Get Best & Residuals on VALID TRAIN SET (Odds >= 8)
    # 1. Get Best & Residuals on FULL TRAIN SET (No filtering)
    # The AI Physicist should discover parity splits (if any) or N<8 noise automatically.
    mask_train = np.ones_like(X_FULL, dtype=bool) 
    x_train = X_FULL[mask_train]
    y_train = Y_FULL[mask_train]
    
    best_entry = top_formulas[0]
    
    # We need to compute RESIDUALS in SIMKIN SPACE (Flattened)
    # Target Flat: log(y) - lgamma + 1.943*x
    epsilon = 1e-9
    from scipy.special import gammaln
    fact = gammaln(x_train + 1)
    x_features_train = np.zeros((len(x_train), 3))
    x_features_train[:, 0] = x_train
    x_features_train[:, 1] = x_train % GpuGlobals.VAR_MOD_X1
    x_features_train[:, 2] = x_train % GpuGlobals.VAR_MOD_X2
    
    y_flat_true = np.log(np.abs(y_train) + epsilon) - fact + (1.943 * x_features_train[:, 0])
    
    # Predict Flat
    residual_str = best_entry.get('residual')
    if not residual_str: return
    
    try:
        tree_res = ExpressionTree.from_infix(residual_str)
        if not tree_res.is_valid: return
        y_flat_pred = tree_res.evaluate(x_features_train)
        
        residuals = y_flat_true - y_flat_pred
        
        # Detect Split (Linear OR Parity)
        phase_info = phase_manager.detect_phase_change(x_train, residuals, min_samples=4)
        split_type = phase_info.get('type')
        gain = phase_info.get('gain', 0.0)
        
        if split_type and gain > 0.05: # 5% variance reduction
             print(f"  [Phase] Regime Change Found! Type: {split_type.upper()} (Gain: {gain:.2f})")
             
             if split_type == 'parity':
                 # PARITY SPLIT
                 mask_L = (x_train.astype(int) % 2 == 0) # Even (Left logic)
                 mask_R = ~mask_L                        # Odd  (Right logic)
                 label_L, label_R = "Even", "Odd"
                 
             else:
                 # LINEAR SPLIT
                 split_val = phase_info['split_val']
                 print(f"  [Phase] Cutoff at x={split_val:.2f}")
                 mask_L = x_train <= split_val
                 mask_R = x_train > split_val
                 label_L, label_R = "Left", "Right"
             
             # Double check sample sizes
             if np.sum(mask_L) < 3 or np.sum(mask_R) < 3:
                 print("  [Phase] Abort: Fragments too small.")
                 return

             # Solve Phases
             print(f"  [Phase] Solving {label_L} Fragment ({np.sum(mask_L)} pts)...")
             res_L = hybrid_solve(
                 x_features_train[mask_L], y_flat_true[mask_L], 
                 model, device, beam_width=5, gp_timeout=30, num_variables=3, max_workers=1
             )
             if not res_L: return
             f1 = res_L['formula']
             
             print(f"  [Phase] Solving {label_R} Fragment ({np.sum(mask_R)} pts)...")
             res_R = hybrid_solve(
                 x_features_train[mask_R], y_flat_true[mask_R], 
                 model, device, beam_width=5, gp_timeout=30, num_variables=3, max_workers=1
             )
             if not res_R: return
             f2 = res_R['formula']
             
             # Combine
             if split_type == 'parity':
                 # Even is f1, Odd is f2
                 combined_res_str = phase_manager.construct_parity_formula(f1, f2)
             else:
                 combined_res_str = phase_manager.construct_gated_formula(f1, f2, split_val)
             
             # Global Refinement (Crucial for the Sigmoid K and Offset)
             # Use ALL training data for refinement
             tree_comb = ExpressionTree.from_infix(combined_res_str)
             initial_vals = convert_and_extract_constants(tree_comb.root)
             
             print("  [Phase] Refining Composite Formula...")
             consts, rmse = optimize_constants(tree_comb, x_features_train, y_flat_true, initial_guess=initial_vals)
             
             # Substitute back
             refined_res_str = substitute_constants(tree_comb.get_infix(), consts, tree_comb.root.get_constant_positions())
             
             # Reconstruct Full
             full_formula_str = f"exp({refined_res_str} - ({SLOPE:.4f} * x0) + lgamma(x0 + 1))"
             
             evaluate_and_save(full_formula_str, refined_res_str, top_formulas, pattern_memory, origin="PhaseSplit")
             
        else:
             print("  [Phase] No significant split found.")

    except Exception as e:
        print(f"  [Phase] Error: {e}")


def main():
    print("--- Infinite Formula Search Script ---")
    
    # Check dependencies
    # Pandas is required


    # Load Model (REMOVED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = None
    print(f"Bypassing Neural Network. Using Pure Evolutionary Search on {DEVICE}.")
        
    top_formulas = load_or_create_top_list()
    pattern_memory = load_pattern_memory()
    print(f"Loaded Pattern Memory with {len(pattern_memory)} patterns.")
    
    phase_manager = PhaseManager(MODEL, DEVICE)

    # CRITICAL FIX: Disable Global Log Transformation for Infinite Search
    # Infinite Search performs its own manual transformation (Simkin: log(y) - lgamma + ...)
    # which produces small, sometimes negative values.
    # Applying another log() in the engine would crash (log of negative) or double-compress.
    print("Configuring Engine: Forcing USE_LOG_TRANSFORMATION = False for manual handling.")
    GpuGlobals.USE_LOG_TRANSFORMATION = False
    print(f"Starting infinite search loop... (Press Ctrl+C to stop)")
    iteration = 0
    
    # --- AUTO-SIMKIN (Dynamic Slope Calculation) ---
    # Goal: Flatten the sequence by removing the growth of N! and finding the linear trend.
    print("Calculating optimal slope (Auto-Simkin)...")
    epsilon = 1e-9
    from scipy.special import gammaln
    
    # Use all training data to find the trend
    y_raw_log = np.log(np.abs(Y_FULL) + epsilon)
    y_log_minus_fact = y_raw_log - gammaln(X_FULL + 1)
    
    # Linear Regression (Slope only)
    # y = SLOPE * x + b
    # We want to find SLOPE that flattens y_log_minus_fact
    # Using simple covariance/variance formula for slope
    x_mean = np.mean(X_FULL)
    y_mean = np.mean(y_log_minus_fact)
    numerator = np.sum((X_FULL - x_mean) * (y_log_minus_fact - y_mean))
    denominator = np.sum((X_FULL - x_mean)**2)
    
    if abs(denominator) < 1e-9:
        SLOPE = 0.0
    else:
        SLOPE = numerator / denominator
        
    print(f"  [Auto-Simkin] Optimal flattened slope found: {SLOPE:.4f}")
    
    while True:
        iteration += 1
        
        # 1. Random Sampling
        # Ensure we pick at least MIN_SAMPLE_SIZE points
        # USER REQUEST: FULL DATASET (AI Physicist determines phases)
        pool_size = len(X_FULL)
        k = random.randint(min(MIN_SAMPLE_SIZE, pool_size), pool_size)
        
        indices = np.sort(np.random.choice(np.arange(pool_size), k, replace=False))
        
        x_sample = X_FULL[indices]
        y_sample = Y_FULL[indices]
        
        # USER REQUEST: Parity Split - Odds Only
        # We don't need x2 (parity) anymore as it's constant 1.
        # x0 = n
        # x1 = n % VAR_MOD_X1
        # x2 = n % VAR_MOD_X2 (Parity)
        # Shape: (k, 3)
        x_features = np.zeros((len(x_sample), 3), dtype=np.float64)
        x_features[:, 0] = x_sample
        x_features[:, 1] = x_sample % GpuGlobals.VAR_MOD_X1
        x_features[:, 2] = x_sample % GpuGlobals.VAR_MOD_X2
        
        print(f"[Iter {iteration}] Sampling {k} pts (Odds >=8)...", end=" ")
        
        # ... (Seeds Logic unchanged) ...
        # Prepare Seeds (Evolutionary Feedback + Mutations)
        extra_seeds = []
        if top_formulas:
            # W0..W4 -> Top 5 formulas
            candidates = top_formulas[:5]
            candidate_formulas = [c['formula'] for c in candidates]
            extra_seeds.extend(candidate_formulas)
            
            # ELITE PERTURBATION: Mutate the best ones to explore local neighborhood
            for f in candidate_formulas:
                extra_seeds.append(mutate_formula_structurally(f))
            
            print(f"+ {len(candidate_formulas)} Elites + {len(candidate_formulas)} Mutations")
        
        if pattern_memory:
            # IMPORTANCE SAMPLING: Patterns that worked before
            sorted_patterns = sorted(pattern_memory.items(), key=lambda x: x[1], reverse=True)
            candidate_pool = [p[0] for p in sorted_patterns[:10]] 
            if candidate_pool:
                # Add 3 variants from patterns
                for _ in range(3):
                    chosen_pattern = random.choice(candidate_pool)
                    # Patterns might have 'C', hybrid_solve handles them
                    extra_seeds.append(chosen_pattern)
                print(f"+ 3 Pattern variants")

        # ... (Search Logic unchanged) ...
        # Transform target
        # Use abs(y) just in case, though they are positive counts usually.
        # SIMKIN MANEUVER: Force slope correction +1.943 * x
        # We REMOVE the factorial and ADD the slope so the AI sees a flatter line.
        # Note: Use x_features[:, 0] (which is x0/n) for the linear correction
        epsilon = 1e-9
        from scipy.special import gammaln
        factorial_term = gammaln(x_sample + 1)
        
        # Transform target
        # Use abs(y) just in case, though they are positive counts usually.
        # SIMKIN MANEUVER 
        # 1. Simkin: +SLOPE * x0
        # 2. No Parity/Smoothing (Split Dataset)
        y_sample_flat = np.log(np.abs(y_sample) + epsilon) - factorial_term + (SLOPE * x_features[:, 0])
        
        # print first few to debug (in stdout)
        if iteration == 1:
             pass # print(f"Flat Y: {y_sample_flat[:3]}")

        # 2. Search (on FLATTENED target)
        try:
            # We use a relatively small beam width for speed, relying on many iterations
            result = hybrid_solve(
                x_features, y_sample_flat,  # PASS x_features (2 vars: x0, x1)
                MODEL, DEVICE, 
                beam_width=10, 
                gp_timeout=120, # Increased for 1M Pop (Allows ~40 gens)
                max_workers=1, # GPU Only (Save CPU heat)
                num_variables=3, # Used x0, x1, x2
                extra_seeds=extra_seeds,
                max_neural_seeds=1, # Restrict NN to just 1 seed for Worker 7
                random_seed_selection=True # User request: "random for infinity search"
            )
        except Exception as e:
            print(f"Search failed: {e}")
            continue
            
        if not result or not result.get('formula'):
            print("No formula found.")
            continue
            
        residual_formula_str = result['formula']
        final_formula_str = residual_formula_str 
        
        try:
            tree = ExpressionTree.from_infix(residual_formula_str)
            if tree.is_valid:
                initial_values = convert_and_extract_constants(tree.root)
                
                # Refine on Odds >= 8
                # 1. History Odds
                mask_hist_odd = (X_FULL >= 8) & (X_FULL % 2 != 0)
                x_hist = X_FULL[mask_hist_odd]
                y_hist = Y_FULL[mask_hist_odd]
                
                # 2. Target Odds
                mask_target_odd = (X_TARGETS % 2 != 0)
                x_targ = X_TARGETS[mask_target_odd]
                y_targ = Y_TARGETS[mask_target_odd]
                
                x_all = np.concatenate((x_hist, x_targ))
                y_all = np.concatenate((y_hist, y_targ))
                
                # Build x_all_features for refinement (3 variables)
                x_all_features = np.zeros((len(x_all), 3), dtype=np.float64)
                x_all_features[:, 0] = x_all
                x_all_features[:, 1] = x_all % GpuGlobals.VAR_MOD_X1
                x_all_features[:, 2] = x_all % GpuGlobals.VAR_MOD_X2
                
                factorial_term_all = gammaln(x_all + 1)
                # Apply Simkin Correction to Validation Target too (No parity correction)
                y_all_flat = np.log(np.abs(y_all) + epsilon) - factorial_term_all + (SLOPE * x_all_features[:, 0])
                
                if initial_values:
                    # optimization expects C-tree
                    # print(f"Refining on Odds N>=8...")
                    
                    # optimization expects C-tree
                    constants_dict, rmse_original = optimize_constants(tree, x_all_features, y_all_flat, initial_guess=initial_values)
                    
                    if constants_dict:
                         # 2.2 INTEGER SNAPPING (New Feature)
                         # Check if any constant is very close to an integer and snap it
                         snapped_dict = {}
                         snapped = False
                         for c_idx, v in constants_dict.items():
                             nearest_int = round(v)
                             if abs(v - nearest_int) < 0.02: # Tolerance 0.02
                                 snapped_dict[c_idx] = float(nearest_int)
                                 snapped = True
                             else:
                                 snapped_dict[c_idx] = v
                         
                         if snapped:
                             # SAFETY CHECK: Verify Snapped RMSE
                             try:
                                 # 1. Build temporary snapped string
                                 pos_tmp = tree.root.get_constant_positions()
                                 inf_tmp = tree.get_infix()
                                 snapped_str = substitute_constants(inf_tmp, snapped_dict, pos_tmp)
                                 
                                 # 2. Evaluate
                                 tree_snap = ExpressionTree.from_infix(snapped_str)
                                 if tree_snap.is_valid:
                                     # Ensure x_all_features is used for evaluation
                                     y_pred_snap = tree_snap.evaluate(x_all_features)
                                     # Calculate RMSE on FLAT space (same as optimization objective)
                                     mse_snap = np.mean((y_pred_snap - y_all_flat)**2)
                                     rmse_snap = np.sqrt(mse_snap)
                                    
                                     # 3. Compare: Allow up to 5% degradation for the sake of simplicity
                                     # Note: rmse_original might be nearly 0. Be careful with ratio.
                                     # Use absolute difference tolerance for very small errors?
                                     ratio = 1.05
                                     if rmse_original < 1e-6: # Ultra perfect fit
                                         ratio = 2.0 # Allow doubling error if it's tiny (1e-7 -> 2e-7 is fine)
                                         
                                     if rmse_snap <= rmse_original * ratio:
                                         # Accepted!
                                         constants_dict = snapped_dict
                                         # print(f"  [Snap] Accepted (RMSE: {rmse_original:.5f} -> {rmse_snap:.5f})")
                                     else:
                                         pass # print(f"  [Snap] Rejected (Degradation too high)")
                             except:
                                 pass # If verification fails, keeping original

                         # 3. Substitute back into residual
                         positions = tree.root.get_constant_positions()
                         infix_with_Cs = tree.get_infix() 
                         refined_residual = substitute_constants(infix_with_Cs, constants_dict, positions)
                         
                         # print(f"Refined Residual: {refined_residual}")
                         residual_formula_str = refined_residual
                         
        except Exception as e:
            print(f"Refinement failed: {e}") 
        
        # 3. RECONSTRUCTION & Transformation
        # Formula = exp( Residual + lgamma(x+1) - SLOPE*x )
        # No parity correction subtraction needed.
        full_formula_str = f"exp({residual_formula_str} - ({SLOPE:.4f} * x0) + lgamma(x0 + 1))"
        # print(f"Reconstructed Full Formula: {full_formula_str}")

        # 4. Evaluate and Save
        # This replaces the manual save logic to avoid inconsistencies (like the 'k' bug)
        evaluate_and_save(
            full_formula_str, 
            residual_formula_str, 
            top_formulas, 
            pattern_memory, 
            origin=f"Search(k={k})",
            time_taken=result.get('time', 0)
        )
        
        # 5. Phase Search (AI Physicist)
        run_phase_search(iteration, top_formulas, phase_manager, MODEL, DEVICE, pattern_memory)

if __name__ == "__main__":
    main()
