import torch
import numpy as np
import time
from model import AlphaSymbolicModel
from mcts import MCTS
from grammar import VOCABULARY, ExpressionTree

def solve_problem(target_x, target_y, model_path="alpha_symbolic_model.pth", simulations=1000):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VOCAB_SIZE = len(VOCABULARY)
    
    # Load Model
    # Note: Training used VOCAB_SIZE + 1 (for SOS).
    model = AlphaSymbolicModel(vocab_size=VOCAB_SIZE + 1, d_model=64).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Warning: Model file not found. Using untrained random weights (for testing only).")
    
    # Initialize MCTS
    mcts = MCTS(model, DEVICE)
    
    print("\nStarting Search (Deep Thinking)...")
    print(f"Target Y: {target_y[:5]}...")
    
    start_time = time.time()
    
    # Run Search
    best_sequence = mcts.search(target_x, target_y, num_simulations=simulations)
    
    elapsed = time.time() - start_time
    
    # Evaluate Result
    tree = ExpressionTree(best_sequence)
    formula_str = tree.get_infix()
    
    y_pred = tree.evaluate(target_x)
    rmse = np.sqrt(np.mean((y_pred - target_y)**2))
    
    print("\n" + "="*40)
    print("SEARCH COMPLETE")
    print("="*40)
    print(f"Time: {elapsed:.2f}s")
    print(f"Simulations: {simulations}")
    print(f"Found Formula: {formula_str}")
    print(f"RMSE: {rmse:.6f}")
    
    print("\nPredictions vs Targets:")
    for i in range(min(5, len(target_x))):
        print(f"X={target_x[i]:.2f} | Pred={y_pred[i]:.2f} | Target={target_y[i]:.2f}")
        
    return formula_str, rmse

if __name__ == "__main__":
    # Test Case: x^2 + 1
    # Generate data
    x_test = np.linspace(-5, 5, 20)
    # y = x^2 + 1
    y_test = x_test**2 + 1
    
    solve_problem(x_test, y_test, simulations=500)
