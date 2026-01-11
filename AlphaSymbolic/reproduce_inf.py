
import numpy as np
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.grammar import ExpressionTree

def test_inf_formula():
    # Problem: Multivariable Interaction
    # formula: lambda x: x[:,0] * x[:,1] + np.sin(x[:,2])
    # lower: -2.0, upper: 2.0, points: 100, vars: 3
    
    print("Generating Data...")
    np.random.seed(42)
    X = np.random.uniform(-2.0, 2.0, (100, 3)).astype(np.float64)
    Y = X[:,0] * X[:,1] + np.sin(X[:,2])
    
    formula_str = "((x1 * x0) + x2)" # Formula found by GPU
    
    print(f"Testing Formula: {formula_str}")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    
    try:
        tree = ExpressionTree.from_infix(formula_str)
        # Benchmark logic: tree.evaluate(X.T)
        # X.T shape: (3, 100)
        y_pred = tree.evaluate(X.T)
        
        print(f"y_pred shape: {y_pred.shape}")
        print(f"y_pred range: [{np.min(y_pred)}, {np.max(y_pred)}]")
        print(f"y_pred NaNs: {np.isnan(y_pred).any()}")
        print(f"y_pred Infs: {np.isinf(y_pred).any()}")
        
        mse = np.mean((y_pred - Y)**2)
        print(f"MSE: {mse}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")

if __name__ == "__main__":
    test_inf_formula()
