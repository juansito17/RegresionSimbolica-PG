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
        
        # 2. Search
        try:
            # We use a relatively small beam width for speed, relying on many iterations
            result = hybrid_solve(
                x_sample, y_sample, 
                MODEL, DEVICE, 
                beam_width=10, 
                gp_timeout=10, # Keep it snappy
                max_workers=4,
                num_variables=1
            )
        except Exception as e:
            print(f"Search failed: {e}")
            continue
            
        if not result or not result.get('formula'):
            print("No formula found in this iteration.")
            continue
            
        formula_str = result['formula']
        print(f"Found candidate: {formula_str}")
        
        # 3. Evaluate on Targets (Extrapolation)
        try:
            tree = ExpressionTree.from_infix(formula_str)
            if not tree.is_valid:
                print("Invalid tree.")
                continue
                
            y_pred_targets = tree.evaluate(X_TARGETS)
            
            # Calculate Error (Sum of Absolute Errors on Targets)
            # Using logs might be better due to scale, but user asked for "error mas bajo posible"
            # Given the huge numbers, absolute error will be massive. 
            # Let's use relative error or just absolute delta?
            # User said "With those that give the lowest possible error to these numbers respectively".
            # I will use direct Absolute Difference Sum.
            
            error_26 = abs(y_pred_targets[0] - Y_TARGETS[0])
            error_27 = abs(y_pred_targets[1] - Y_TARGETS[1])
            total_error = error_26 + error_27
            
            print(f"Extrapolation Error: {total_error:.2e} (26: {y_pred_targets[0]:.2e}, 27: {y_pred_targets[1]:.2e})")
            
            # 4. Update Top List
            entry = {
                'formula': formula_str,
                'extrapolation_error': total_error,
                'pred_26': y_pred_targets[0],
                'pred_27': y_pred_targets[1],
                'sample_size': k,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add to list
            top_formulas.append(entry)
            
            # Deduplicate by formula (keep lowest error if duplicates - though exact formula dupe is likely same error)
            # Simple dedup:
            unique_formulas = {d['formula']: d for d in top_formulas}
            top_formulas = list(unique_formulas.values())
            
            # Sort
            top_formulas.sort(key=lambda x: x['extrapolation_error'])
            
            # Keep Top 5
            if len(top_formulas) > TOP_K:
                top_formulas = top_formulas[:TOP_K]
            
            # Save
            save_top_list(top_formulas)
            
            # Show current top
            print(f"Current Best Error: {top_formulas[0]['extrapolation_error']:.2e}")
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            continue

if __name__ == "__main__":
    main()
