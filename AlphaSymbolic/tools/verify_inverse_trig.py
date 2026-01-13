import sys
import os
import numpy as np
import warnings

# Add path to AlphaSymbolic
sys.path.append(os.path.join(os.getcwd(), 'AlphaSymbolic'))

from search.hybrid_search import hybrid_solve
from ui.app_core import get_model
from core.grammar import ExpressionTree

def verify_inverse_trig():
    print("=== Verifying Inverse Trigonometric Operators End-to-End ===\n")
    
    # 1. Generate Data for asin(x)
    # Domain of asin is [-1, 1]. Let's use [-0.9, 0.9] to be safe.
    x = np.linspace(-0.9, 0.9, 20)
    y = np.arcsin(x)
    
    print(f"Target Formula: asin(x)")
    print(f"Data Points: {len(x)}\n")
    
    # 2. Load Model (Mocking if needed, but hybrid_solve uses real model)
    # We need to make sure we don't crash if model loading fails or takes too long.
    # But hybrid_solve needs it.
    print("Loading Model...")
    try:
        model, device = get_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    # 3. Solve using Hybrid Search
    # This will call the C++ engine.
    print("Running Hybrid Search (this invokes C++ GP)...")
    try:
        # Increase timeout to ensure GP has time, though asin(x) is simple.
        # But GP needs to find 'S' operator.
        result = hybrid_solve(x, y, model, device, beam_width=50, gp_timeout=20)
    except Exception as e:
        print(f"Hybrid Search Crashed: {e}")
        return

    if not result:
        print("❌ Hybrid Search failed to return any result.")
        return

    print(f"\nResult: {result}")
    
    formula_str = result.get('formula', '')
    print(f"Found Formula: {formula_str}")
    
    # 4. Check if formula contains 'asin' OR 'atan(x/sqrt(1-x^2))' etc.
    # Ideally it should be exactly asin(x) or close.
    # The C++ engine uses 'S' for asin. py bridge converts back.
    
    if 'asin' in formula_str:
        print("✅ Success: 'asin' found in formula.")
    elif 'atan' in formula_str or 'acos' in formula_str:
        print("⚠️  Partial Success: Found other inverse trig functions, might be equivalent.")
    else:
        print("❌ Failure: Inverse trig function NOT found.")
        
    # 5. Evaluate Error
    try:
        tree = ExpressionTree.from_infix(formula_str)
        y_pred = tree.evaluate(x)
        rmse = np.sqrt(np.mean((y_pred - y)**2))
        print(f"RMSE: {rmse:.6f}")
        
        if rmse < 1e-4:
            print("✅ Accuracy: Excellent.")
        else:
            print("❌ Accuracy: Poor.")
            
    except Exception as e:
        print(f"Error evaluating result: {e}")

if __name__ == "__main__":
    verify_inverse_trig()
