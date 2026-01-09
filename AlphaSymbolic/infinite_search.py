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
    # Sort by error (ascending)
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
        k = random.randint(MIN_SAMPLE_SIZE, len(X_FULL))
        indices = np.sort(np.random.choice(len(X_FULL), k, replace=False))
        
        x_sample = X_FULL[indices]
        y_sample = Y_FULL[indices]
        
        print(f"\n[Iter {iteration}] Sampling {k} points...")
        
        # Prepare Seeds (Evolutionary Feedback)
        extra_seeds = []
        if top_formulas and len(top_formulas) > 0:
            best_formula = top_formulas[0]['formula']
            extra_seeds.append(best_formula)
            print(f"Feedback: Injecting best formula as seed: {best_formula[:50]}...")

        # 2. Search
        try:
            # We use a relatively small beam width for speed, relying on many iterations
            result = hybrid_solve(
                x_sample, y_sample, 
                MODEL, DEVICE, 
                beam_width=10, 
                gp_timeout=10, # Keep it snappy
                max_workers=4,
                num_variables=1,
                extra_seeds=extra_seeds
            )
        except Exception as e:
            print(f"Search failed: {e}")
            continue
            
        if not result or not result.get('formula'):
            print("No formula found in this iteration.")
            continue
            
        formula_str = result['formula']
        print(f"Found candidate: {formula_str}")
        
        # 2.5 INTELLIGENT REFINEMENT (BFGS)
        try:
            # Parse tree
            tree = ExpressionTree.from_infix(formula_str)
            if tree.is_valid:
                # 1. Convert hardcoded numbers to 'C' and get initial values
                initial_values = convert_and_extract_constants(tree.root)
                
                if initial_values:
                    print(f"Refining {len(initial_values)} constants with BFGS...")
                    
                    # 2. Optimize using ALL data (1-27) to hit targets perfectly
                    x_all = np.concatenate((X_FULL, X_TARGETS))
                    y_all = np.concatenate((Y_FULL, Y_TARGETS))
                    
                    # optimization expects C-tree (which we created in step 1 by mutation)
                    constants_dict, rmse = optimize_constants(tree, x_all, y_all, initial_guess=initial_values)
                    
                    if constants_dict:
                         # 3. Substitute back
                         positions = tree.root.get_constant_positions()
                         infix_with_Cs = tree.get_infix() # Tree now has Cs
                         refined_formula = substitute_constants(infix_with_Cs, constants_dict, positions)
                         
                         print(f"Refined: {refined_formula}")
                         formula_str = refined_formula # Update for final eval
        except Exception as e:
            print(f"Refinement failed: {e}") 
        
        # 3. Evaluate on FULL RANGE (History + Targets) - "Holistic Intelligence"
        try:
            tree = ExpressionTree.from_infix(formula_str)
            if not tree.is_valid:
                print("Invalid tree.")
                continue
            
            # Combine all points for validation
            x_all = np.concatenate((X_FULL, X_TARGETS))
            y_all = np.concatenate((Y_FULL, Y_TARGETS))
            
            y_pred_all = tree.evaluate(x_all)
            
            # Calculate RMSLE (Root Mean Squared Log Error) allows comparing across 15 orders of magnitude
            # log1p handles 0s gracefully (log(1+0) = 0)
            # We clip predictions to be non-negative because log of negative is undefined
            y_pred_safe = np.maximum(y_pred_all, 0)
            
            log_error = np.sqrt(np.mean((np.log1p(y_pred_safe) - np.log1p(y_all))**2))
            
            # Also calculate Extrapolation Absolute Error (just for info)
            pred_26 = y_pred_all[-2]
            pred_27 = y_pred_all[-1]
            extrap_error_sum = abs(pred_26 - Y_TARGETS[0]) + abs(pred_27 - Y_TARGETS[1])
            
            print(f"RMSLE (1-27): {log_error:.6f} | Extrap Error: {extrap_error_sum:.2e}")
            
            # 4. Update Top List
            entry = {
                'formula': formula_str,
                'rmsle_global': log_error, # Primary Metric
                'extrapolation_error': extrap_error_sum, # Secondary info
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
            
            # Sort by Extrapolation Error (User's primary goal)
            # We keep RMSLE just for information/filtering if needed
            top_formulas.sort(key=lambda x: x['extrapolation_error'])
            
            # Keep Top 5
            if len(top_formulas) > TOP_K:
                top_formulas = top_formulas[:TOP_K]
            
            # Save
            save_top_list(top_formulas)
            
            current_best = top_formulas[0]['extrapolation_error']
            print(f"Current Best Extrap Error: {current_best:.2e}")
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            continue

if __name__ == "__main__":
    main()
