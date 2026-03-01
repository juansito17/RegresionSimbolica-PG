"""
AlphaSymbolic Pro Search - Integrated Full Pipeline
Combines all improvements:
- Pattern Detection (initial hints)
- Beam Search OR MCTS
- Constant Optimization
- Simplification
- Pareto Front
- Pattern Memory
"""
import torch
import numpy as np
import time
import argparse
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AlphaSymbolic.core.model import AlphaSymbolicModel
from AlphaSymbolic.core.grammar import VOCABULARY, ExpressionTree
from AlphaSymbolic.search.beam_search import beam_solve
from AlphaSymbolic.search.mcts import MCTS
from AlphaSymbolic.utils.optimize_constants import optimize_constants
from AlphaSymbolic.utils.simplify import simplify_tree
from AlphaSymbolic.search.pareto import ParetoFront
from data.pattern_memory import PatternMemory
from AlphaSymbolic.utils.detect_pattern import detect_pattern, summarize_pattern


def solve_pro(target_x, target_y, 
              model_path="alpha_symbolic_model.pth",
              method="beam",  # "beam" or "mcts"
              beam_width=15,
              mcts_simulations=200,
              use_memory=True,
              verbose=True):
    """
    Professional formula solver with all bells and whistles.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VOCAB_SIZE = len(VOCABULARY)
    
    if verbose:
        print("="*60)
        print("AlphaSymbolic Pro - Full Pipeline")
        print("="*60)
        print(f"Device: {DEVICE}")
        print(f"Method: {method.upper()}")
        print(f"Data points: {len(target_x)}")
    
    # --- Phase 0: Pattern Detection ---
    if verbose:
        print("\n--- Phase 0: Pattern Detection ---")
    pattern_info = detect_pattern(target_x, target_y)
    if verbose:
        print(f"Detected: {pattern_info['type']} (confidence: {pattern_info['confidence']:.1%})")
        print(f"Suggested ops: {pattern_info['suggested_ops']}")
    
    # --- Load Model ---
    model = AlphaSymbolicModel(vocab_size=VOCAB_SIZE + 1, d_model=128).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.eval()
        if verbose:
            print("\n✓ Model loaded")
    except:
        if verbose:
            print("\n⚠ Model not found, using random weights")
    
    # --- Load Pattern Memory ---
    memory = PatternMemory() if use_memory else None
    
    # --- Phase 1: Search ---
    if verbose:
        print(f"\n--- Phase 1: {method.upper()} Search ---")
    
    start_time = time.time()
    
    all_results = []
    
    if method == "beam":
        results = beam_solve(target_x, target_y, model, DEVICE, 
                            beam_width=beam_width, max_length=25)
        if results:
            all_results.extend(results)
    else:
        mcts = MCTS(model, DEVICE)
        best_seq = mcts.search(target_x, target_y, num_simulations=mcts_simulations)
        tree = ExpressionTree(best_seq)
        if tree.is_valid:
            constants, rmse = optimize_constants(tree, target_x, target_y)
            all_results.append({
                'tokens': best_seq,
                'rmse': rmse,
                'constants': constants,
                'formula': tree.get_infix()
            })
    
    search_time = time.time() - start_time
    
    if verbose:
        print(f"Time: {search_time:.2f}s")
        print(f"Candidates found: {len(all_results)}")
    
    # --- Phase 2: Pareto Front ---
    if verbose:
        print("\n--- Phase 2: Pareto Analysis ---")
    
    pareto = ParetoFront()
    pareto.add_from_results(all_results)
    
    if verbose:
        print(f"Non-dominated solutions: {len(pareto.solutions)}")
    
    # --- Phase 3: Select Best Solutions ---
    if verbose:
        print("\n--- Phase 3: Final Selection ---")
    
    best_rmse = pareto.get_best_by_rmse()
    simplest = pareto.get_simplest()
    balanced = pareto.get_balanced(alpha=0.6)  # Slight preference for accuracy
    
    # --- Phase 4: Simplify Winners ---
    if verbose:
        print("\n--- Phase 4: Simplification ---")
    
    final_results = {}
    
    if best_rmse:
        tree = ExpressionTree(best_rmse.tokens)
        simplified = simplify_tree(tree) 
        final_results['best_accuracy'] = {
            'formula': best_rmse.formula,
            'simplified': simplified,
            'rmse': best_rmse.rmse,
            'complexity': best_rmse.complexity
        }
        
        # Record in pattern memory
        if memory:
            memory.record(best_rmse.tokens, best_rmse.rmse, best_rmse.formula)
    
    if simplest and simplest != best_rmse:
        tree = ExpressionTree(simplest.tokens)
        simplified = simplify_tree(tree)
        final_results['simplest'] = {
            'formula': simplest.formula,
            'simplified': simplified,
            'rmse': simplest.rmse,
            'complexity': simplest.complexity
        }
    
    if balanced and balanced != best_rmse and balanced != simplest:
        tree = ExpressionTree(balanced.tokens)
        simplified = simplify_tree(tree)
        final_results['balanced'] = {
            'formula': balanced.formula,
            'simplified': simplified,
            'rmse': balanced.rmse,
            'complexity': balanced.complexity
        }
    
    # Save memory
    if memory:
        memory.save()
    
    # --- Final Report ---
    if verbose:
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        for key, res in final_results.items():
            print(f"\n[{key.upper()}]")
            print(f"  Formula: {res['formula']}")
            print(f"  Simplified: {res['simplified']}")
            print(f"  RMSE: {res['rmse']:.6f}")
            print(f"  Complexity: {res['complexity']} nodes")
        
        # Predictions comparison
        if best_rmse:
            tree = ExpressionTree(best_rmse.tokens)
            y_pred = tree.evaluate(target_x, constants=best_rmse.constants)
            
            print(f"\n--- Predictions (Best Accuracy) ---")
            for i in range(min(5, len(target_x))):
                diff = abs(y_pred[i] - target_y[i])
                print(f"x={target_x[i]:8.2f} | Pred={y_pred[i]:12.4f} | Target={target_y[i]:12.4f} | Δ={diff:.4f}")
    
    return final_results, pareto


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaSymbolic Pro Solver")
    parser.add_argument("--method", choices=["beam", "mcts"], default="beam")
    parser.add_argument("--beam-width", type=int, default=15)
    parser.add_argument("--mcts-sims", type=int, default=200)
    args = parser.parse_args()
    
    # Test 1: Linear
    print("\n" + "#"*60)
    print("# TEST 1: y = 2x + 3")
    print("#"*60)
    x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
    y1 = 2 * x1 + 3
    solve_pro(x1, y1, method=args.method, beam_width=args.beam_width)
    
    # Test 2: Quadratic
    print("\n" + "#"*60)
    print("# TEST 2: y = x^2 + 1")
    print("#"*60)
    x2 = np.linspace(-5, 5, 15).astype(np.float64)
    y2 = x2**2 + 1
    solve_pro(x2, y2, method=args.method, beam_width=args.beam_width)
