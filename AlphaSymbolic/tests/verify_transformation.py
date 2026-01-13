
import numpy as np
from scipy.special import gammaln
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.grammar import ExpressionTree, OPERATORS

def test_transformation():
    print("Testing Transformation Logic...")
    
    # 1. Setup Data: y = n!
    x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    # y = [1, 2, 6, 24, 120]
    y = np.array([1, 2, 6, 24, 120], dtype=np.float64)
    
    # 2. Transformation (as implemented)
    epsilon = 1e-9
    factorial_term = gammaln(x + 1)
    y_flat = np.log(np.abs(y) + epsilon) - factorial_term
    
    print(f"X: {x}")
    print(f"Y (Original): {y}")
    print(f"Factorial term: {factorial_term}")
    print(f"Y Flat (Should be near 0): {y_flat}")
    
    # Check if flat is small
    if np.allclose(y_flat, 0, atol=1e-5):
        print("SUCCESS: Transformation flattened factorial data to zero.")
    else:
        print(f"FAILURE: Transformation did not flatten data. Max diff: {np.max(np.abs(y_flat))}")

    # 3. Reconstruction with constant residual 0
    # Formula: exp(0 + lgamma(x)) - ExpressionTree adds +1 internally
    residual_str = "0"
    full_formula_str = f"exp({residual_str} + lgamma(x))"
    
    print(f"Reconstructed Formula: {full_formula_str}")
    
    # 4. Evaluate using ExpressionTree
    try:
        tree = ExpressionTree.from_infix(full_formula_str)
        if not tree.is_valid:
            print("FAILURE: Tree invalid.")
            return
            
        y_pred = tree.evaluate(x)
        print(f"Pred: {y_pred}")
        
        if np.allclose(y_pred, y, rtol=1e-5):
             print("SUCCESS: Reconstruction matched original data.")
        else:
             print(f"FAILURE: Reconstruction mismatch. Max diff: {np.max(np.abs(y_pred - y))}")
             
    except Exception as e:
        print(f"FAILURE: Exception during tree evaluation: {e}")

if __name__ == "__main__":
    test_transformation()
