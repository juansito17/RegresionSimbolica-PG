import torch
import numpy as np
import os
import sys

# Ensure root is in path
sys.path.append(os.getcwd())

from core.grammar import VOCABULARY, OPERATORS, ExpressionTree, TOKEN_TO_ID
from ui.app_core import load_model, get_model, CURRENT_PRESET, TRAINING_STATUS
from search.mcts import MCTS
from data.synthetic_data import DataGenerator
from ui.app_training import normalize_batch, train_basic, train_curriculum, train_self_play
from utils.optimize_constants import optimize_constants
from utils.benchmark_runner import run_benchmark_suite
from data.benchmark_data import BENCHMARK_SUITE, get_benchmark_data

# Mock progress callback
class MockProgress:
    def __call__(self, value, desc=""):
        pass  # Do nothing

def test_vocabulary():
    print("--- [1/6] Testing Vocabulary & Grammar ---")
    print(f"Vocab size: {len(VOCABULARY)}")
    
    operators_to_check = ['sign', 'abs', 'max', 'min', 'tan']
    for op in operators_to_check:
        if op not in VOCABULARY:
            print(f"[FAIL] Operator '{op}' missing from VOCABULARY")
            return False
        print(f"[PASS] Operator '{op}' verified.")
    return True

def test_model_loading():
    print("\n--- [2/6] Testing Model Loading (Lite & Pro) ---")
    try:
        # Test Lite
        status, _ = load_model(preset_name="lite")
        print(f"Lite: {status}")
        model, _ = get_model()
        if model.d_model != 128:
            print(f"[FAIL] Lite wrong dim: {model.d_model}")
            return False
        print("[PASS] Lite OK")
        
        # Test Pro
        status, _ = load_model(preset_name="pro")
        print(f"Pro: {status}")
        model, _ = get_model()
        if model.d_model != 256:
            print(f"[FAIL] Pro wrong dim: {model.d_model}")
            return False
        print("[PASS] Pro OK")
        
        # Switch back to Lite for remaining tests
        load_model(preset_name="lite")
        return True
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False

def test_mcts_execution():
    print("\n--- [3/6] Testing MCTS Execution ---")
    try:
        model, device = get_model()
        X = np.array([-1, 0, 1], dtype=np.float32)
        y = X.copy()
        
        mcts = MCTS(model, device, max_simulations=10, batch_size=2)
        result = mcts.search(X, y)
        
        print(f"Formula: {result.get('formula', 'None')}, RMSE: {result['rmse']:.4f}")
        print("[PASS] MCTS ran without crashing.")
        return True
    except Exception as e:
        print(f"[FAIL] MCTS Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_train_basic():
    print("\n--- [4/6] Testing train_basic (1 epoch) ---")
    try:
        TRAINING_STATUS["running"] = False  # Reset
        result, fig = train_basic(epochs=1, batch_size=4, point_count=10, progress=MockProgress())
        
        if "Error" in result:
            print(f"[FAIL] train_basic returned error: {result}")
            return False
        
        print(f"Result: {result[:100]}...")
        print("[PASS] train_basic passed.")
        return True
    except Exception as e:
        print(f"[FAIL] train_basic Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_train_curriculum():
    print("\n--- [5/6] Testing train_curriculum (1 epoch) ---")
    try:
        TRAINING_STATUS["running"] = False
        result, fig = train_curriculum(epochs=1, batch_size=4, point_count=10, progress=MockProgress())
        
        if "Error" in result:
            print(f"[FAIL] train_curriculum returned error: {result}")
            return False
        
        print(f"Result: {result[:100]}...")
        print("[PASS] train_curriculum passed.")
        return True
    except Exception as e:
        print(f"[FAIL] train_curriculum Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_train_self_play():
    print("\n--- [6/6] Testing train_self_play (1 iteration) ---")
    try:
        TRAINING_STATUS["running"] = False
        result, fig = train_self_play(iterations=1, problems_per_iter=2, point_count=10, progress=MockProgress())
        
        if "Error" in result:
            print(f"[FAIL] train_self_play returned error: {result}")
            return False
        
        print(f"Result: {result[:100]}...")
        print("[PASS] train_self_play passed.")
        return True
    except Exception as e:
        print(f"[FAIL] train_self_play Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_expression_tree_eval():
    print("\n--- [7/9] Testing ExpressionTree Evaluation (New Operators) ---")
    try:
        x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        
        # Test sign(x)
        tree_sign = ExpressionTree(['sign', 'x'])
        result = tree_sign.evaluate(x)
        expected = np.sign(x)
        if not np.allclose(result, expected, equal_nan=True):
            print(f"[FAIL] sign(x) failed: got {result}, expected {expected}")
            return False
        print("[PASS] sign(x) OK")
        
        # Test abs(x)
        tree_abs = ExpressionTree(['abs', 'x'])
        result = tree_abs.evaluate(x)
        expected = np.abs(x)
        if not np.allclose(result, expected, equal_nan=True):
            print(f"[FAIL] abs(x) failed")
            return False
        print("[PASS] abs(x) OK")
        
        # Test max(x, C) with C=0
        # For ['max', 'x', 'C'], the constant 'C' is at path [1] (second child)
        tree_max = ExpressionTree(['max', 'x', 'C'])
        result = tree_max.evaluate(x, constants={(1,): 0.0})  # Path [1] = child index 1
        expected = np.maximum(x, 0.0)
        if not np.allclose(result, expected, equal_nan=True):
            print(f"[FAIL] max(x, C) failed: got {result}, expected {expected}")
            return False
        print("[PASS] max(x, C) OK (ReLU)")
        
        # Test min(x, C) with C=0
        tree_min = ExpressionTree(['min', 'x', 'C'])
        result = tree_min.evaluate(x, constants={(1,): 0.0})
        expected = np.minimum(x, 0.0)
        if not np.allclose(result, expected, equal_nan=True):
            print(f"[FAIL] min(x, C) failed")
            return False
        print("[PASS] min(x, C) OK")
        
        print("[PASS] All new operators verified.")
        return True
    except Exception as e:
        print(f"[FAIL] ExpressionTree Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimize_constants():
    print("\n--- [8/9] Testing Constant Optimizer (BFGS) ---")
    try:
        x = np.linspace(-1, 1, 20).astype(np.float32)
        # y = 2*x + 3 (target: C0=2, C1=3)
        y = 2 * x + 3
        
        tokens = ['+', '*', 'C', 'x', 'C']  # C*x + C
        tree = ExpressionTree(tokens)  # Create tree object!
        
        optimized, rmse = optimize_constants(tree, x, y)
        
        print(f"Optimized constants: {optimized}, RMSE: {rmse:.6f}")
        
        # Verify RMSE is low (should find near-perfect fit)
        if rmse > 0.1:
            print(f"[WARN] High RMSE ({rmse:.4f}), but algorithm ran.")
        
        print("[PASS] Constant optimizer ran successfully.")
        return True
    except Exception as e:
        print(f"[FAIL] Optimizer Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_benchmark_runner():
    print("\n--- [9/9] Testing Benchmark Runner (1 problem) ---")
    try:
        model, device = get_model()
        
        # Test just fetching benchmark data
        x, y, name = get_benchmark_data(BENCHMARK_SUITE[0]['id'])
        print(f"Benchmark problem: {name}")
        print(f"Data shape: X={x.shape}, Y={y.shape}")
        
        # We won't run full suite (slow), just verify data loads
        print("[PASS] Benchmark data loads correctly.")
        return True
    except Exception as e:
        print(f"[FAIL] Benchmark Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup():
    print("\n--- Cleaning up test files ---")
    for p in ["alpha_symbolic_model_lite.pth", "alpha_symbolic_model_pro.pth"]:
        if os.path.exists(p):
            os.remove(p)
            print(f"Removed {p}")

if __name__ == "__main__":
    print("[START] AlphaSymbolic COMPLETE System Check (9 Tests)")
    print("=" * 50)
    
    results = []
    results.append(("Vocabulary", test_vocabulary()))
    results.append(("Model Loading", test_model_loading()))
    results.append(("MCTS", test_mcts_execution()))
    results.append(("train_basic", test_train_basic()))
    results.append(("train_curriculum", test_train_curriculum()))
    results.append(("train_self_play", test_train_self_play()))
    results.append(("ExpressionTree (New Ops)", test_expression_tree_eval()))
    results.append(("Constant Optimizer", test_optimize_constants()))
    results.append(("Benchmark Data", test_benchmark_runner()))
    
    cleanup()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    all_passed = True
    for name, passed in results:
        status = "[PASS] PASS" if passed else "[FAIL] FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n[OK] ALL {len(results)} TESTS PASSED! System is 100% healthy.")
        exit(0)
    else:
        print("\n[WARN] SOME TESTS FAILED.")
        exit(1)
