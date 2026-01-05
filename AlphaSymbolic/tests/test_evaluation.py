
import sys
import os
import numpy as np
sys.path.append(os.getcwd())

from core.grammar import ExpressionTree

def test_evaluation():
    print("Testing Formula Evaluation Correctness...")
    
    x_val = np.array([1.0, 2.0, 3.0, 0.5, -1.0])
    
    test_cases = [
        # (Tokens, Expected Lambda, Name)
        (['+', 'x', '1'], lambda x: x + 1, "x + 1"),
        (['-', 'x', 'x'], lambda x: x - x, "x - x (Zero)"),
        (['*', 'x', '2'], lambda x: x * 2, "2x"),
        (['pow', 'x', '2'], lambda x: x**2, "x^2"),
        (['sin', 'x'], lambda x: np.sin(x), "sin(x)"),
        (['exp', 'x'], lambda x: np.exp(x), "exp(x)"),
        # Nested: sin(x^2)
        (['sin', 'pow', 'x', '2'], lambda x: np.sin(x**2), "sin(x^2)"),
        # Log (handle domain safety manually for test inputs, but implementation handles it)
        (['log', 'x'], lambda x: np.log(np.abs(x) + 1e-10), "log(|x|)"),
    ]

    all_passed = True
    
    for tokens, func, name in test_cases:
        try:
            tree = ExpressionTree(tokens)
            y_pred = tree.evaluate(x_val)
            y_true = func(x_val)
            
            # Check difference
            diff = np.abs(y_pred - y_true)
            max_diff = np.max(diff)
            
            if max_diff < 1e-5:
                print(f"✅ {name}: OK (Max Diff: {max_diff:.1e})")
            else:
                print(f"❌ {name}: FAILED (Max Diff: {max_diff:.1e})")
                print(f"   Pred: {y_pred}")
                print(f"   True: {y_true}")
                all_passed = False
                
        except Exception as e:
            print(f"❌ {name}: CRASH ({e})")
            all_passed = False

    return all_passed

if __name__ == "__main__":
    if test_evaluation():
        print("\nALL EVALUATIONS CORRECT.")
    else:
        print("\nSOME EVALUATIONS FAILED.")
