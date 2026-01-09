import time
import numpy as np
import torch
import pandas as pd
import os
import random
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from search.hybrid_search import hybrid_solve
from ui.app_core import get_model
from core.grammar import ExpressionTree
from utils.optimize_constants import optimize_constants, substitute_constants, convert_and_extract_constants

# --- CONFIGURATION ---
X_FULL = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25], dtype=np.float64)
Y_FULL = np.array([1,0,0,2,10,4,40,92,352,724,2680,14200,73712,365596,2279184,14772512,95815104,666090624,4968057848,39029188884,314666222712,2691008701644,2423393768440,227514171973736,2207893435808352], dtype=np.float64)

# Targets for Extrapolation
X_TARGETS = np.array([26, 27], dtype=np.float64)
Y_TARGETS = np.array([22317699616364044, 234907967154122528], dtype=np.float64)

CSV_FILE = "top_formulas.csv"
TOP_K = 5
MIN_SAMPLE_SIZE = 6 # > 5

def load_or_create_top_list():
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            return df.to_dict('records')
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

def main():
    print("--- Infinite Formula Search Script ---")
    
    # Check dependencies
    # Pandas is required


    # Load Model
    print("Loading Model...")
    try:
        MODEL, DEVICE = get_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    top_formulas = load_or_create_top_list()
    
    print(f"Starting infinite search loop... (Press Ctrl+C to stop)")
    iteration = 0
    
    while True:
        iteration += 1
        
        # 1. Random Sampling
        # Ensure we pick at least MIN_SAMPLE_SIZE points
        # USER REQUEST: Start from n > 4 (Indices >= 4, since X starts at 1)
        valid_indices = np.arange(4, len(X_FULL)) # 4, 5, ... end
        
        # Adjust k if valid pool is small
        pool_size = len(valid_indices)
        k = random.randint(min(MIN_SAMPLE_SIZE, pool_size), pool_size)
        
        indices = np.sort(np.random.choice(valid_indices, k, replace=False))
        
        x_sample = X_FULL[indices]
        y_sample = Y_FULL[indices]
        
        print(f"\n[Iter {iteration}] Sampling {k} points...")
        
        # Prepare Seeds (Evolutionary Feedback)
        # Prepare Seeds (Evolutionary Feedback)
        extra_seeds = []
        if top_formulas:
            # User request: "pick 3 samples from top 5" (Updated from top 3)
            candidates = top_formulas[:5]
            candidate_formulas = [c['formula'] for c in candidates]
            
            if candidate_formulas:
                # Sample 3 seeds with replacement 
                # (allows giving more compute to the very best if picked twice)
                chosen = random.choices(candidate_formulas, k=3)
                extra_seeds.extend(chosen)
                print(f"Feedback: Injected {len(chosen)} seeds from Top {len(candidates)} (Random Sample).")

        # 2. Search
        # 1.5 Flattening Transformation (The "Feynman" Trick)
        # y_flat = log(y) - lgamma(x + 1)
        # We use log1p for safety near 0, although y is usually large integerrs.
        # But wait, user said "log(target)". 
        # Since we reconstruct with exp, we must be consistent.
        # Shift y slightly to avoid log(0) if any y=0 exists (indices 1,2 are 0).
        # We'll use a small epsilon.
        epsilon = 1e-9
        # y_sample indices corresponds to x_sample values.
        # x_sample are values like 1, 2, ...
        
        # Calculate lgamma(n+1) which is log(n!)
        from scipy.special import gammaln
        factorial_term = gammaln(x_sample + 1)
        
        # Transform target
        # Use abs(y) just in case, though they are positive counts usually.
        y_sample_flat = np.log(np.abs(y_sample) + epsilon) - factorial_term
        
        # print first few to debug (in stdout)
        if iteration == 1:
            print(f"Sample X: {x_sample[:3]}")
            print(f"Sample Y: {y_sample[:3]}")
            print(f"Flat Y: {y_sample_flat[:3]}")

        # 2. Search (on FLATTENED target)
        try:
            # We use a relatively small beam width for speed, relying on many iterations
            result = hybrid_solve(
                x_sample, y_sample_flat,  # PASS FLAT Y
                MODEL, DEVICE, 
                beam_width=10, 
                gp_timeout=60, 
                max_workers=1, # GPU Only (0 CPU)
                num_variables=1,
                extra_seeds=extra_seeds
            )
        except Exception as e:
            print(f"Search failed: {e}")
            continue
            
        if not result or not result.get('formula'):
            print("No formula found in this iteration.")
            continue
            
        residual_formula_str = result['formula']
        print(f"Found residual candidate: {residual_formula_str}")
        
        # 2.5 INTELLIGENT REFINEMENT (BFGS) on Residual
        final_formula_str = residual_formula_str # Default if refinement fails
        
        try:
            # Parse residual tree
            tree = ExpressionTree.from_infix(residual_formula_str)
            if tree.is_valid:
                # 1. Convert hardcoded numbers to 'C' and get initial values
                initial_values = convert_and_extract_constants(tree.root)
                
                # Refine on ALL data (1-27) but FLATTENED
                x_all = np.concatenate((X_FULL, X_TARGETS))
                y_all = np.concatenate((Y_FULL, Y_TARGETS))
                
                factorial_term_all = gammaln(x_all + 1)
                y_all_flat = np.log(np.abs(y_all) + epsilon) - factorial_term_all
                
                if initial_values:
                    print(f"Refining {len(initial_values)} constants on FLAT surface...")
                    
                    # optimization expects C-tree
                    constants_dict, rmse = optimize_constants(tree, x_all, y_all_flat, initial_guess=initial_values)
                    
                    if constants_dict:
                         # 3. Substitute back into residual
                         positions = tree.root.get_constant_positions()
                         infix_with_Cs = tree.get_infix() 
                         refined_residual = substitute_constants(infix_with_Cs, constants_dict, positions)
                         
                         print(f"Refined Residual: {refined_residual}")
                         residual_formula_str = refined_residual
                         
        except Exception as e:
            print(f"Refinement failed: {e}") 
        
        # 3. RECONSTRUCTION & Transformation
        # Formula = exp( Residual + lgamma(x+1) )
        # We construct this string.
        # Note: lgamma(x+1) is 'lgamma(x+1)' in our language (or similar).
        # Our language has 'lgamma'. Input is 'x'.
        # So we string concat: "exp(" + residual + " + lgamma(x + 1))"
        
        # We need to be careful about parens.
        # FIX: lgamma in ExpressionTree adds +1 internally (lgamma(|x|+1)).
        # So we use lgamma(x) to represent lgamma(x+1) mathematically.
        full_formula_str = f"exp({residual_formula_str} + lgamma(x))"
        print(f"Reconstructed Full Formula: {full_formula_str}")

        # 4. Evaluate on FULL RANGE (History + Targets) w/ Reconstructed Formula
        try:
            tree = ExpressionTree.from_infix(full_formula_str)
            if not tree.is_valid:
                print("Invalid reconstructed tree.")
                # Fallback to evaluate residual directly? No, that's wrong scale.
                continue
            
            # Combine all points for validation
            x_all = np.concatenate((X_FULL, X_TARGETS))
            y_all = np.concatenate((Y_FULL, Y_TARGETS))
            
            y_pred_all = tree.evaluate(x_all)
            
            # Calculate RMSLE on ORIGINAL SPACE
            y_pred_safe = np.maximum(y_pred_all, 0) # Clip negative predictions
            
            # Validating log error
            # Handle potential overflow in y_pred if it's huge?
            # if y_pred is inf, log is inf.
            
            log_error = np.sqrt(np.mean((np.log1p(y_pred_safe) - np.log1p(y_all))**2))
            
            # Also calculate Extrapolation Absolute Error
            pred_26 = y_pred_all[-2]
            pred_27 = y_pred_all[-1]
            extrap_error_sum = abs(pred_26 - Y_TARGETS[0]) + abs(pred_27 - Y_TARGETS[1])
            
            print(f"RMSLE (1-27): {log_error:.6f} | Extrap Error: {extrap_error_sum:.2e}")
            
            # 5. Update Top List
            entry = {
                'formula': full_formula_str, # Store the FULL formula
                'residual': residual_formula_str, # Store residual for curiosity
                'rmsle_global': log_error, 
                'extrapolation_error': extrap_error_sum, 
                'pred_26': pred_26,
                'pred_27': pred_27,
                'sample_size': k,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add to list
            top_formulas.append(entry)
            
            # Deduplicate by formula
            unique_formulas = {d['formula']: d for d in top_formulas}
            top_formulas = list(unique_formulas.values())
            
            # Sort by RMSLE
            top_formulas.sort(key=lambda x: x.get('rmsle_global', float('inf')))
            
            # Keep Top 5
            if len(top_formulas) > TOP_K:
                top_formulas = top_formulas[:TOP_K]
            
            # Save
            save_top_list(top_formulas)
            
            current_best = top_formulas[0].get('rmsle_global', 999)
            print(f"Current Best RMSLE: {current_best:.6f}")
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            continue

if __name__ == "__main__":
    main()
