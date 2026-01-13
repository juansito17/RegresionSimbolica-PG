
import numpy as np
import math
import sys
import os

# Add project root to path to import core
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'AlphaSymbolic'))

from AlphaSymbolic.core.grammar import ExpressionTree
from scipy.special import gammaln

def test_diagnosis():
    print("--- DIAGNOSTIC TEST ---")
    
    # 1. Test basic lgamma values
    x_test = np.array([26.0, 27.0])
    
    # Manual calc
    # Note: Our code uses lgamma(|x|+1)
    manual_26 = gammaln(abs(26.0) + 1.0)
    manual_27 = gammaln(abs(27.0) + 1.0)
    
    print(f"Manual lgamma(26+1): {manual_26}")
    print(f"Manual lgamma(27+1): {manual_27}")
    
    # 2. Test ExpressionTree lgamma
    formula_lgamma = "lgamma(x)"
    tree_lgamma = ExpressionTree.from_infix(formula_lgamma)
    
    if not tree_lgamma.is_valid:
        print("ERROR: Could not parse 'lgamma(x)'")
    else:
        pred_lgamma = tree_lgamma.evaluate(x_test)
        print(f"Tree lgamma(x) at 26, 27: {pred_lgamma}")
        
    # 3. Test Full Reconstruction logic with dummy residual
    # Case A: residual = 0
    residual_str = "0"
    full_formula = f"exp({residual_str} + lgamma(x))"
    print(f"\nTesting Formula: {full_formula}")
    
    tree_full = ExpressionTree.from_infix(full_formula)
    if not tree_full.is_valid:
         print(f"ERROR: Could not parse '{full_formula}'")
    else:
         pred_full = tree_full.evaluate(x_test)
         print(f"Prediction at 26: {pred_full[0]}")
         print(f"Prediction at 27: {pred_full[1]}")
         
         # Check if it equals e
         if abs(pred_full[0] - math.e) < 1e-4:
             print("CRITICAL: Prediction is e (2.718...)!")
             
    # 4. Test User Hypothesis: Eval failure returning default?
    # In evaluate:
    # if val == 'e': return np.e
    # If parsing somehow resulted in just 'e'?
    
    print("\n--- End Diagnosis ---")

if __name__ == "__main__":
    try:
        test_diagnosis()
    except Exception as e:
        print(f"Runtime Error: {e}")
        import traceback
        traceback.print_exc()
