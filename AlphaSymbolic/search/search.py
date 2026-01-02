"""
AlphaSymbolic Search Script - Enhanced Version
Combines MCTS with Constant Optimization and Simplification.
"""
import torch
import numpy as np
import time
from core.model import AlphaSymbolicModel
from search.mcts import MCTS
from core.grammar import VOCABULARY, ExpressionTree
from utils.optimize_constants import optimize_constants
from utils.simplify import simplify_tree

def solve_problem(target_x, target_y, model_path="alpha_symbolic_model.pth", simulations=500):
    """
    Solve a symbolic regression problem using AlphaSymbolic.
    
    Pipeline:
    1. MCTS finds best formula structure (with 'C' placeholders)
    2. Constant Optimization fills in optimal values for 'C'
    3. Simplification cleans up the formula
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VOCAB_SIZE = len(VOCABULARY)
    
    print("="*50)
    print("AlphaSymbolic Symbolic Regression Engine")
    print("="*50)
    print(f"Device: {DEVICE}")
    print(f"Data points: {len(target_x)}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    
    # Load Model
    model = AlphaSymbolicModel(vocab_size=VOCAB_SIZE + 1, d_model=64).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.eval()
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("⚠ Model not found - using random weights (results will be poor)")
    except Exception as e:
        print(f"⚠ Model load warning: {e}")
    
    # Initialize MCTS
    mcts = MCTS(model, DEVICE)
    
    print("\n--- Phase 1: Neural-Guided Tree Search ---")
    start_time = time.time()
    
    # Run Search
    best_sequence = mcts.search(target_x, target_y, num_simulations=simulations)
    
    mcts_time = time.time() - start_time
    
    # Build tree
    tree = ExpressionTree(best_sequence)
    raw_formula = tree.get_infix()
    
    print(f"Time: {mcts_time:.2f}s")
    print(f"Found structure: {raw_formula}")
    print(f"Constants to optimize: {tree.count_constants()}")
    
    # Phase 2: Constant Optimization
    print("\n--- Phase 2: Constant Optimization ---")
    opt_start = time.time()
    
    constants, rmse_before_opt = optimize_constants(tree, target_x, target_y)
    
    opt_time = time.time() - opt_start
    
    # Evaluate with optimized constants
    y_pred = tree.evaluate(target_x, constants=constants)
    final_rmse = np.sqrt(np.mean((y_pred - target_y)**2))
    
    print(f"Time: {opt_time:.2f}s")
    print(f"Optimized constants: {len(constants)}")
    print(f"RMSE after optimization: {final_rmse:.6f}")
    
    # Phase 3: Simplification
    print("\n--- Phase 3: Algebraic Simplification ---")
    simplified_formula = simplify_tree(tree)
    
    # Build final formula string with constants substituted
    final_formula = raw_formula
    if constants:
        positions = tree.root.get_constant_positions()
        for pos in positions:
            key = tuple(pos)
            if key in constants:
                val = constants[key]
                if abs(val - round(val)) < 0.001:
                    val_str = str(int(round(val)))
                else:
                    val_str = f"{val:.4f}"
                final_formula = final_formula.replace('C', val_str, 1)
    
    # Final Report
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Total Time: {mcts_time + opt_time:.2f}s")
    print(f"Raw Formula: {raw_formula}")
    print(f"With Constants: {final_formula}")
    print(f"Simplified: {simplified_formula}")
    print(f"Final RMSE: {final_rmse:.6f}")
    
    print("\n--- Predictions vs Targets ---")
    for i in range(min(5, len(target_x))):
        diff = abs(y_pred[i] - target_y[i])
        print(f"x={target_x[i]:8.2f} | Pred={y_pred[i]:12.4f} | Target={target_y[i]:12.4f} | Δ={diff:.4f}")
    
    if len(target_x) > 5:
        print(f"... ({len(target_x) - 5} more points)")
    
    return {
        'raw_formula': raw_formula,
        'final_formula': final_formula,
        'simplified': simplified_formula,
        'rmse': final_rmse,
        'constants': constants,
        'tokens': best_sequence
    }


if __name__ == "__main__":
    # Test Case 1: y = x^2 + 1
    print("\n" + "#"*60)
    print("# TEST 1: y = x^2 + 1")
    print("#"*60)
    x_test = np.linspace(-5, 5, 20).astype(np.float64)
    y_test = x_test**2 + 1
    result1 = solve_problem(x_test, y_test, simulations=200)
    
    # Test Case 2: y = 2*x + 3 (linear)
    print("\n" + "#"*60)
    print("# TEST 2: y = 2*x + 3 (Linear)")
    print("#"*60)
    x_test2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
    y_test2 = 2 * x_test2 + 3
    result2 = solve_problem(x_test2, y_test2, simulations=200)
