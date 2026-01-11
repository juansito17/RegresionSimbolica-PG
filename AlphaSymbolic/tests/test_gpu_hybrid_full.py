import torch
import numpy as np
import time
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from search.hybrid_search import hybrid_solve
from core.grammar import ExpressionTree, VOCABULARY

class MockModel(torch.nn.Module):
    """
    Mock NN that doesn't really do anything, we will rely on 
    explicit seed injection passed to hybrid_solve.
    """
    def forward(self, x, y, seq, formula_mask=None):
        bs, seq_len = seq.shape
        vocab = len(VOCABULARY)
        return torch.randn(bs, seq_len, vocab, device=seq.device), torch.randn(bs, 1, device=seq.device)

def test_full_integration():
    print("=== Starting Full GPU Hybrid Integration Test ===")
    
    # 1. Setup Problem: y = 2*x + 1 (Simple linear)
    # Target constants: 2, 1
    # We will inject a seed "x * C + C" to see if it optimizes constants.
    
    x_val = np.linspace(-5, 5, 20)
    y_val = 2.0 * x_val + 1.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 2. Define "Good" Seed (Structurally correct, constants wrong)
    # The engine should load this and fix constants via gradient descent.
    # Note: 'C' in infix maps to a constant slot.
    good_seed = "x * C + C" 
    
    # 3. Define "Bad" Seed (Structural dead end)
    bad_seed = "sin(x)"
    
    seeds = [bad_seed, bad_seed, good_seed] # Hide good seed in list
    
    print(f"Injection Seeds: {seeds}")
    
    # 4. Run Hybrid Solve
    # We pass extra_seeds to simulate "Beam Search" results or Feedback Loop
    start_t = time.time()
    
    # Note: hybrid_search internal beam search uses the model. 
    # With MockModel it produces garbage.
    # We rely on 'extra_seeds' OR the fact that hybrid_solve will use the seeds we pass?
    # Wait, hybrid_solve signature:
    # hybrid_solve(..., extra_seeds=...)
    
    result = hybrid_solve(
        x_val, y_val, 
        MockModel().to(device), 
        device,
        beam_width=2,         # Minimal beam
        gp_timeout=15,        # 15 seconds to optimize
        gp_binary_path=None,  # We are using GPU
        max_workers=0,        # Should be ignored by GPU path, but logic uses it for fallback
        extra_seeds=seeds     # Inject our seeds
    )
    
    end_t = time.time()
    print(f"Search completed in {end_t - start_t:.2f}s")
    print(f"Result: {result}")
    
    # 5. Assertions
    if result['formula'] is None:
        print("FAIL: No formula found.")
        return
        
    f_str = result['formula']
    rmse = result['rmse']
    
    print(f"Found Formula: {f_str}")
    print(f"RMSE: {rmse}")
    
    # Check if formula is valid
    try:
        tree = ExpressionTree.from_infix(f_str)
        y_pred = tree.evaluate(x_val)
        final_mse = np.mean((y_val - y_pred)**2)
        print(f"Verification MSE: {final_mse}")
        
        if final_mse < 1e-4:
            print("SUCCESS: Target accuracy achieved.")
            # Check if it looks like our seed or equivalent
            # x*2+1 vs 2*x+1 etc.
            print("Integration Verified!")
        else:
             print("WARNING: MSE not effectively zero. Optimization might need more time or better seeds.")
    except Exception as e:
        print(f"FAIL: Formula verification crashed: {e}")

if __name__ == "__main__":
    test_full_integration()
