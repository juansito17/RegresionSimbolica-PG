
import torch
import numpy as np
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.gpu.engine import TensorGeneticEngine
from core.gpu.config import GpuGlobals

def test_convergence():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing Convergence on {device}")
    
    # Force settings
    GpuGlobals.POP_SIZE = 10000
    GpuGlobals.GENERATIONS = 50
    GpuGlobals.USE_NANO_PSO = True
    
    # Problem: x^2 + x + 1
    # Simple, 1 var.
    X = np.linspace(-5, 5, 100).reshape(1, 100)
    Y = X**2 + X + 1
    
    x_t = torch.tensor(X, dtype=torch.float64, device=device)
    y_t = torch.tensor(Y, dtype=torch.float64, device=device)
    
    engine = TensorGeneticEngine(
        pop_size=GpuGlobals.POP_SIZE,
        num_variables=1,
        device=device
    )
    
    print("Initial Population Analysis:")
    pop = engine.initialize_population()
    
    # Check variable frequency
    # Assuming standard grammar: 0=PAD, vars around start
    # We need to know token IDs for 'x0'
    id_x0 = engine.grammar.token_to_id['x0']
    
    flat = pop.flatten()
    count_x0 = (flat == id_x0).sum().item()
    count_total = flat.numel()
    print(f"x0 Tokens: {count_x0} / {count_total} ({count_x0/count_total:.2%})")
    
    if count_x0 == 0:
        print("CRITICAL: No variables in initial population!")
    
    print("\nRunning Evolution...")
    
    # Custom callback to track diversity
    def progress(gen, rmse, best_rpn, best_const, improved, _):
        if gen % 10 == 0 or improved:
            # Decode best
            if best_rpn is not None:
                s = engine.rpn_to_infix(best_rpn, best_const)
                print(f"Gen {gen}: RMSE {rmse:.4f} | {s}")
            else:
                print(f"Gen {gen}: RMSE {rmse:.4f}")
                
    engine.run(x_t, y_t, callback=progress)

if __name__ == "__main__":
    test_convergence()
