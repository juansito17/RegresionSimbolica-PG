import torch
import numpy as np
import time
from search.hybrid_search import hybrid_solve

def test_optimization():
    print("--- Verifying Optimization Configuration ---")
    
    # 1. Setup Simple Data (y = 2x)
    x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    y = x * 2
    
    print(f"Dataset: X={x}, Y={y}")
    
    # 2. Configure Parameters
    # Using small population and NO log transform
    pop_size = 50_000 
    use_log = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Params: Pop={pop_size}, UseLog={use_log}")
    
    # 3. Run Search
    print("\nRunning Hybrid Search...")
    start_time = time.time()
    
    # Mock Model (None) to force GP only
    result = hybrid_solve(
        x, y, 
        model=None, 
        device=device,
        beam_width=10,
        gp_timeout=10, # Short timeout
        pop_size=pop_size,
        use_log=use_log,
        num_variables=1
    )
    
    elapsed = time.time() - start_time
    print(f"\nElapsed Time: {elapsed:.2f}s")
    
    # 4. Assertions
    if result['formula']:
        print(f"SUCCESS! Formula found: {result['formula']}")
        print(f"RMSE: {result['rmse']}")
        
        # Verify correctness
        # Formula might be (x + x) or (2 * x) or similar
        # Since we don't have a parser here handy without importing more, 
        # we rely on the RMSE being close to 0.
        if result['rmse'] < 1e-5:
            print("Verdict: ACCURATE")
        else:
            print("Verdict: INACCURATE (RMSE too high)")
            
    else:
        print("FAILURE: No formula found.")

    print("------------------------------------------")

if __name__ == "__main__":
    test_optimization()
