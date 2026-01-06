
import torch
import numpy as np
import pandas as pd
import time
from tabulate import tabulate

from ui.app_core import get_model, MODEL_PRESETS
from search.hybrid_search import hybrid_solve
from data.feynman_data import FEYNMAN_1D_SUBSET
from core.grammar import ExpressionTree

def evaluate_formula(formula_str, x_val):
    """Safely evaluates a formula string on x_val."""
    try:
        # Create a simple context for evaluation
        # constants
        pi = np.pi
        e = np.e
        
        # functions
        sin = np.sin
        cos = np.cos
        tan = np.tan
        exp = np.exp
        log = np.log
        sqrt = np.sqrt
        
        # Safe eval using locals
        # Replace 'x' in string with 'x_val' variable name if needed, but eval context handles 'x'
        x = x_val 
        
        # Handle 'C' if present (shouldn't be in ground truth, but just in case)
        # Ground truths in feynman_data don't have C, they are exact.
        
        res = eval(formula_str)
        return res
    except Exception as e:
        return np.full_like(x_val, np.nan)

def run_benchmark():
    print("\n" + "="*60)
    print("🔬 ALPHA SYMBOLIC: FEYNMAN BENCHMARK (PHYSICS)")
    print("="*60 + "\n")
    
    # 1. Load Model
    print("Loading Pro Model...")
    try:
        # Force load best available model (likely 'lite' but let's check app_core defaults)
        # In a real run we might want to let the user choose, but here we auto-load.
        model, device = get_model()
        print(f"Model Loaded on {device}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    results = []
    
    # 2. Iterate Problems
    for i, problem in enumerate(FEYNMAN_1D_SUBSET):
        print(f"\n[Problem {i+1}/{len(FEYNMAN_1D_SUBSET)}] {problem['name']} (ID: {problem['id']})")
        print(f"Target: {problem['formula']}")
        
        # 3. Generate Data
        # Generate clean data for the problem
        x_test = np.linspace(0.1, 5.0, 20) # Avoid 0 for potential singularities like 1/x
        y_test = evaluate_formula(problem['formula'], x_test)
        
        if np.any(np.isnan(y_test)) or np.any(np.isinf(y_test)):
            print("Skipping due to numerical issues in ground truth generation.")
            continue
            
        # 4. Solve
        start_time = time.time()
        try:
            # Using hybrid_solve with moderate beam width
            solution = hybrid_solve(
                x_test, 
                y_test, 
                model, 
                device, 
                beam_width=50,   # High effort
                gp_timeout=10,    # standard timeout
                gp_binary_path=None 
            )
            
            elapsed = time.time() - start_time
            
            if solution:
                pred_formula = solution.get('formula', "N/A")
                gp_rmse = solution.get('rmse', 999.0)
                
                # Verify RMSE independently if needed, but we trust the output mostly
                # Let's simple check error on the test set
                try:
                     # Parse prediction to evaluate
                    pred_tree = ExpressionTree.from_infix(pred_formula)
                    y_pred = pred_tree.evaluate(x_test)
                    real_rmse = np.sqrt(np.mean((y_test - y_pred)**2))
                    
                    is_solved = real_rmse < 0.01
                    status = "✅ SOLVED" if is_solved else "❌ FAILED"
                    
                except:
                    real_rmse = 999.0
                    status = "⚠️ ERROR"
                
                print(f"Result: {status} | RMSE: {real_rmse:.5f} | Time: {elapsed:.2f}s")
                print(f"Prediction: {pred_formula}")
                
                results.append({
                    "ID": problem['id'],
                    "Name": problem['name'],
                    "Target": problem['formula'],
                    "Prediction": pred_formula,
                    "RMSE": real_rmse,
                    "Time": elapsed,
                    "Status": status
                })
                
            else:
                print("Result: No solution found.")
                results.append({"ID": problem['id'], "Status": "NO_SOLUTION"})
                
        except Exception as e:
            print(f"Error executing solve: {e}")
            results.append({"ID": problem['id'], "Status": "CRASH"})

    # 5. Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    df = pd.DataFrame(results)
    if not df.empty:
        # Clean up for display
        display_cols = ["Name", "Target", "Prediction", "RMSE", "Status"]
        print(tabulate(df[display_cols], headers="keys", tablefmt="grid", showindex=False))
        
        # Stats
        solved_count = df[df["Status"].str.contains("SOLVED")].shape[0]
        total = len(df)
        print(f"\nSuccess Rate: {solved_count}/{total} ({solved_count/total*100:.1f}%)")
        
        # Save results
        df.to_csv("feynman_results.csv", index=False)
        print("Results saved to feynman_results.csv")
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_benchmark()
