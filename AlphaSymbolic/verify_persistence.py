
import numpy as np
import time
import torch
from search.hybrid_search import hybrid_solve
from core.gpu.engine import TensorGeneticEngine

def verify():
    # Linear data: y = 2*x
    x = np.linspace(0, 10, 20).reshape(-1, 1)
    y = 2 * x
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run 1: Should be slow
    print("\n--- Run 1 (Cold Start) ---")
    start = time.time()
    res1 = hybrid_solve(x, y, model=None, device=device, gp_timeout=0.001)
    end = time.time()
    print(f"Time: {end - start:.4f}s")
    
    # Run 2: Should be fast
    print("\n--- Run 2 (Persistent Cache) ---")
    start = time.time()
    res2 = hybrid_solve(x, y, model=None, device=device, gp_timeout=0.001)
    end = time.time()
    print(f"Time: {end - start:.4f}s")

    # Run 3: Different data, same engine
    print("\n--- Run 3 (Persistent Cache + Different Data) ---")
    x2 = np.linspace(0, 10, 20).reshape(-1, 1)
    y2 = 3 * x + 5
    start = time.time()
    res3 = hybrid_solve(x2, y2, model=None, device=device, gp_timeout=0.001)
    end = time.time()
    print(f"Time: {end - start:.4f}s")

if __name__ == "__main__":
    verify()
