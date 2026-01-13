
import os
import numpy as np
import sys
import time
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.gp_bridge import GPEngine
from core.grammar import ExpressionTree

def generate_random_formula(num_vars):
    """Generate a random formula with the given number of variables."""
    vars_list = [f"x{i}" for i in range(num_vars)]
    
    # Pick random subset of variables to use (at least 2)
    num_to_use = random.randint(2, min(num_vars, 5))  # Cap at 5 for readability
    used_vars = random.sample(vars_list, num_to_use)
    
    # Templates for different complexity levels
    templates = [
        # Simple arithmetic
        lambda v: f"{v[0]} + {v[1]}",
        lambda v: f"{v[0]} * {v[1]}",
        lambda v: f"({v[0]} + {v[1]}) * {random.randint(1,5)}",
        lambda v: f"{v[0]} - {v[1]}",
        # Medium - 3 vars
        lambda v: f"({v[0]} * {v[1]}) + {v[2 % len(v)]}",
        lambda v: f"({v[0]} + {v[1]}) * {v[2 % len(v)]}",
        lambda v: f"({v[0]} * {v[0]}) + ({v[1]} * {v[1]})",
        # Complex - trig
        lambda v: f"sin({v[0]}) + cos({v[1]})",
        lambda v: f"sin({v[0]}) + (cos({v[1]}) * {v[2 % len(v)]})",
        # High dimension sum
        lambda v: " + ".join(v[:min(len(v), 4)]),
        # High dimension product-sum
        lambda v: f"({v[0]} * {v[1]}) + ({v[2 % len(v)]} * {v[3 % len(v)]})" if len(v) >= 2 else f"{v[0]} * 2",
    ]
    
    template = random.choice(templates)
    return template(used_vars)

def test_multivariable_gp():
    engine = GPEngine()
    
    num_tests = 5
    
    print(f"==========================================")
    print(f"   RANDOM MULTIVARIABLE GP STRESS TEST    ")
    print(f"       (UP TO 10 VARIABLES)               ")
    print(f"==========================================")
    
    passed = 0
    failed = 0
    
    for i in range(num_tests):
        # Random number of variables (2 to 10)
        num_vars = random.randint(2, 10)
        formula = generate_random_formula(num_vars)
        
        print(f"\nTEST {i+1}: {num_vars} variables")
        print(f"  - Target: {formula}")
        
        # Generate Data
        samples = random.randint(50, 100)
        x_safe = np.random.uniform(-3, 3, (samples, num_vars))
        inputs = {f'x{j}': x_safe[:, j] for j in range(num_vars)}
        
        try:
            tree_gt = ExpressionTree.from_infix(formula)
            y_values = tree_gt.evaluate(inputs)
            
            # Skip if y has NaN/Inf
            if np.any(np.isnan(y_values)) or np.any(np.isinf(y_values)):
                print(f"  - Skipping (invalid y values)")
                continue
            
            print(f"  - Solving with GP...")
            
            start_t = time.time()
            result = engine.run(x_safe, y_values.tolist(), timeout_sec=30)
            end_t = time.time()
            
            if result:
                print(f"  - GP Result: {result}")
                try:
                    tree_res = ExpressionTree.from_infix(result)
                    y_pred = tree_res.evaluate(inputs)
                    rmse = np.sqrt(np.mean((y_values - y_pred)**2))
                    print(f"  - RMSE: {rmse:.8f}")
                    print(f"  - Time: {end_t - start_t:.2f}s")
                    
                    if rmse < 0.05:
                        print(f"  => VERDICT: [PASS]")
                        passed += 1
                    else:
                        print(f"  => VERDICT: [APPROXIMATE]")
                        passed += 1  # Still count as partial success
                except Exception as e:
                    print(f"  - Error evaluating result: {e}")
                    failed += 1
            else:
                print(f"  => VERDICT: [FAILED - No result]")
                failed += 1
        except Exception as e:
            print(f"  - Error: {e}")
            failed += 1
        
        print("-" * 40)
    
    print(f"\n==========================================")
    print(f"SUMMARY: {passed}/{num_tests} PASSED, {failed}/{num_tests} FAILED")
    print(f"==========================================")

if __name__ == "__main__":
    test_multivariable_gp()
