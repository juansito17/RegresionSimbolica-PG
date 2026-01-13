
import numpy as np
import time
import pandas as pd
import torch
import sys
import os

# Ensure we can import from local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.app_core import get_model, DEVICE
from search.hybrid_search import hybrid_solve
from core.grammar import ExpressionTree

def generate_data(problem_type, n_samples=100):
    """Generate synthetic data for benchmark problems."""
    rng = np.random.RandomState(42)
    
    if problem_type == "easy":
        # x0^2 + x1 - 1
        X = rng.uniform(-5, 5, (n_samples, 2))
        y = X[:, 0]**2 + X[:, 1] - 1
        formula_gt = "x0^2 + x1 - 1"
        
    elif problem_type == "medium":
        # x0^2 + 3*sin(x1) + x2 - x3 (Multivariable + Noise variable x4)
        X = rng.uniform(-5, 5, (n_samples, 5))
        y = X[:, 0]**2 + 3*np.sin(X[:, 1]) + X[:, 2] - X[:, 3]
        # x4 is noise
        formula_gt = "x0^2 + 3*sin(x1) + x2 - x3"
        
    elif problem_type == "hard":
        # exp(-x0^2) * sin(x1) + x2/x3 (Nested + Division)
        X = rng.uniform(0.1, 5, (n_samples, 4)) # Avoid div by zero
        y = np.exp(-X[:, 0]**2) * np.sin(X[:, 1]) + X[:, 2] / X[:, 3]
        formula_gt = "exp(-x0^2) * sin(x1) + x2/x3"
    
    return X, y, formula_gt

def run_alphasymbolic(X, y, timeout=30):
    print(f"   [AlphaSymbolic] Running (Timeout={timeout}s)...")
    
    # Load model (LITE is fine for benchmark)
    model, device = get_model()
    
    start_time = time.time()
    
    # Run Hybrid Search
    # Using beam_width=10 for speed, similar to PySR's population
    try:
        result = hybrid_solve(
            X, y, model, device, 
            beam_width=20, 
            gp_timeout=timeout,
            num_variables=X.shape[1],
            max_workers=4
        )
        
        elapsed = time.time() - start_time
        
        formula = result.get('formula', "Failed")
        rmse = result.get('rmse', 999.0)
        
        return formula, rmse, elapsed
        
    except Exception as e:
        print(f"   [AlphaSymbolic] Error: {e}")
        return "Error", 999.0, time.time() - start_time

def run_pysr(X, y, timeout=30):
    print(f"   [PySR] Running (Timeout={timeout}s)...")
    
    try:
        from pysr import PySRRegressor
    except ImportError:
        return "PySR Not Installed", None, 0.0

    try:
        # Configure PySR to match our settings roughly
        model = PySRRegressor(
            niterations=100000,  # Unlimited, controlled by timeout
            timeout_in_seconds=timeout,
            binary_operators=["+", "-", "*", "/", "pow"],
            unary_operators=["sin", "cos", "exp", "log"],
            verbosity=0,
            temp_equation_file=True,
            delete_tempfiles=True,
            random_state=42,
            procs=4,
            multithreading=True
        )
        
        start_time = time.time()
        model.fit(X, y)
        elapsed = time.time() - start_time
        
        # Get best
        best = model.get_best()
        formula = str(best.equation)
        
        # Calculate RMSE manually to ensure fairness
        y_pred = model.predict(X)
        rmse = np.sqrt(np.mean((y_pred - y)**2))
        
        return formula, rmse, elapsed
        
    except Exception as e:
        print(f"   [PySR] Error: {e}")
        return f"Error: {str(e)[:50]}", 999.0, 0.0

def main():
    problems = ["easy", "medium", "hard"]
    results = []
    
    print("\n" + "="*60)
    print("🥊 BENCHMARK: AlphaSymbolic vs PySR 🥊")
    print("="*60 + "\n")
    
    # Check PySR
    try:
        import pysr
        print("✅ PySR detectado. ¡Que comience la batalla!")
    except ImportError:
        print("⚠️ PySR NO detectado. Solo se ejecutará AlphaSymbolic.")
        print("   (pip install pysr && python -c 'import pysr; pysr.install()')")
    
    for prob in problems:
        print(f"\n>>> ROUND: {prob.upper()} <<<")
        X, y, gt = generate_data(prob)
        print(f"Target: {gt}")
        print(f"Data: {X.shape} samples")
        
        # 1. AlphaSymbolic
        alpha_form, alpha_rmse, alpha_time = run_alphasymbolic(X, y)
        results.append({
            "Problem": prob,
            "Method": "AlphaSymbolic",
            "Formula": alpha_form,
            "RMSE": alpha_rmse,
            "Time": alpha_time
        })
        print(f"   -> Alpha: RMSE={alpha_rmse:.5f}, Time={alpha_time:.2f}s")
        
        # 2. PySR
        if "PySR Not Installed" not in run_pysr(X[:2], y[:2], timeout=1)[0]: # Quick check
             pysr_form, pysr_rmse, pysr_time = run_pysr(X, y)
             if pysr_rmse is not None:
                results.append({
                    "Problem": prob,
                    "Method": "PySR",
                    "Formula": pysr_form,
                    "RMSE": pysr_rmse,
                    "Time": pysr_time
                })
                print(f"   -> PySR:  RMSE={pysr_rmse:.5f}, Time={pysr_time:.2f}s")
        else:
             print("   -> PySR: Skipped (Not Installed)")

    # Print Summary Table
    print("\n" + "="*60)
    print("📊 RESULTADOS FINALES")
    print("="*60)
    df = pd.DataFrame(results)
    # Reorder columns
    df = df[["Problem", "Method", "RMSE", "Time", "Formula"]]
    print(df.to_markdown(index=False))
    print("="*60)

if __name__ == "__main__":
    main()
