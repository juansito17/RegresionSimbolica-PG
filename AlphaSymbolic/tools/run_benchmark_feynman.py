
import torch
import numpy as np
import pandas as pd
import time
from tabulate import tabulate
from ui.app_core import get_model, load_model, MODEL_PRESETS
from search.hybrid_search import hybrid_solve
from data.expanded_benchmarks import load_expanded_feynman_subset, evaluate_projected_formula
from core.grammar import ExpressionTree

def evaluate_dynamic(problem, x_val):
    """Wrapper to evaluate dynamic problems with fixed contexts."""
    return evaluate_projected_formula(
        problem['original_formula'], 
        problem['target_var'], 
        x_val, 
        problem['fixed_context']
    )

def run_benchmark():
    print("\n" + "="*80)
    print("🔬 ALPHA SYMBOLIC: EXPANDED FEYNMAN BENCHMARK (LITE vs PRO)")
    print("="*80 + "\n")
    
    # LOAD DATASETS
    print("Loading Feynman Dataset (FULL)...")
    problems = load_expanded_feynman_subset(limit=None) 
    if not problems:
        print("No problems loaded. Check data/benchmarks/FeynmanEquations.csv")
        return

    presets_to_test = ['lite', 'pro']
    all_results = []
    
    summary_comparison = []

    for preset in presets_to_test:
        print(f"\n>>> LOADING MODEL: {preset.upper()} <<<")
        try:
            status, info = load_model(preset_name=preset)
            print(f"Status: {status} | Device: {info}")
            model, device = get_model()
        except Exception as e:
            print(f"Failed to load {preset}: {e}")
            continue

        preset_results = []
        
        # Iterate Problems
        for i, problem in enumerate(problems):
            print(f"\n[{preset.upper()}] Problem {i+1}/{len(problems)}: {problem['name']}")
            print(f"Target: {problem['original_formula']}")
            print(f"Desc: {problem['description']}")
            
            # Generate Data
            x_test = np.linspace(0.1, 5.0, 20)
            y_test = evaluate_dynamic(problem, x_test)
            
            if np.any(np.isnan(y_test)) or np.any(np.isinf(y_test)):
                print("Skipping due to numerical issues.")
                continue
                
            # Solve
            start_time = time.time()
            try:
                solution = hybrid_solve(
                    x_test, 
                    y_test, 
                    model, 
                    device, 
                    beam_width=50,
                    gp_timeout=10,
                    max_workers=6
                )
                
                elapsed = time.time() - start_time
                
                if solution:
                    pred_formula = solution.get('formula', "N/A")
                    
                    # Verify RMSE
                    try:
                        # For RMSE check we treat prediction as function of x
                        # The ground truth y_test is already correct
                        pred_tree = ExpressionTree.from_infix(pred_formula)
                        y_pred = pred_tree.evaluate(x_test)
                        real_rmse = np.sqrt(np.mean((y_test - y_pred)**2))
                        is_solved = real_rmse < 0.05 # Relaxed slightly for complex physics
                        status_text = "✅ SOLVED" if is_solved else "❌ FAILED"
                    except:
                        real_rmse = 999.0
                        status_text = "⚠️ ERROR"
                    
                    print(f"Result: {status_text} | RMSE: {real_rmse:.4f} | Time: {elapsed:.2f}s")
                    
                    all_results.append({
                        "Model": preset,
                        "ID": problem['id'],
                        "Name": problem['name'],
                        "Target": problem['original_formula'],
                        "Prediction": pred_formula,
                        "RMSE": real_rmse,
                        "Time": elapsed,
                        "Status": status_text
                    })
                    
                    preset_results.append({
                        "ID": problem['id'],
                        "Status": status_text,
                        "Time": elapsed
                    })

                else:
                    print("Result: No solution found.")
                    all_results.append({"Model": preset, "ID": problem['id'], "Name": problem['name'], "Status": "NO_SOLUTION", "RMSE": 999.0, "Time": elapsed})
                    preset_results.append({"ID": problem['id'], "Status": "NO_SOLUTION", "Time": elapsed})
                    
            except Exception as e:
                print(f"Error executing solve: {e}")
                all_results.append({"Model": preset, "ID": problem['id'], "Name": problem['name'], "Status": "CRASH", "RMSE": 999.0, "Time": 0.0})
                preset_results.append({"ID": problem['id'], "Status": "CRASH", "Time": 0.0})

        summary_comparison.append({"Model": preset, "Results": preset_results})

    # Final Comparative Report
    print("\n" + "="*80)
    print("🏆 FINAL COMPARISON REPORT (EXPANDED)")
    print("="*80)
    
    # Pivot results for side-by-side view
    comparison_rows = []
    
    lite_map = {r['ID']: r for r in summary_comparison[0]['Results']} if len(summary_comparison) > 0 else {}
    pro_map = {r['ID']: r for r in summary_comparison[1]['Results']} if len(summary_comparison) > 1 else {}
    
    for problem in problems:
        pid = problem['id']
        name = problem['name']
        
        l_res = lite_map.get(pid, {"Status": "N/A", "Time": 0.0})
        p_res = pro_map.get(pid, {"Status": "N/A", "Time": 0.0})
        
        comparison_rows.append({
            "ID": pid,
            "LITE Status": l_res['Status'],
            "LITE Time": f"{l_res['Time']:.2f}s",
            "PRO Status": p_res['Status'],
            "PRO Time": f"{p_res['Time']:.2f}s"
        })
        
    df_compare = pd.DataFrame(comparison_rows)
    print(tabulate(df_compare, headers="keys", tablefmt="grid", showindex=False))
    
    # Save CSV
    pd.DataFrame(all_results).to_csv("feynman_expanded_results.csv", index=False)
    print("\nDetailed results saved to 'feynman_expanded_results.csv'")

if __name__ == "__main__":
    run_benchmark()
