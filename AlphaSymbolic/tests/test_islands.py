
import torch
import numpy as np
from core.gpu.engine import TensorGeneticEngine
from core.gpu.config import GpuGlobals

# Mock callback to capture island reporting
class MockCallback:
    def __init__(self):
        self.reports = []

    def __call__(self, gen, best_rmse, best_rpn, best_consts, is_new_best, island_idx):
        if is_new_best:
            self.reports.append(island_idx)
            print(f"Callback recieved: Best found on Island {island_idx}")

def test_islands_reporting():
    print("Testing Island Reporting...")
    
    # Force multi-island config
    GpuGlobals.POP_SIZE = 100
    GpuGlobals.NUM_ISLANDS = 4
    GpuGlobals.GENERATIONS = 5
    
    engine = TensorGeneticEngine(pop_size=100, n_islands=4)
    
    # Dummy data
    x = torch.linspace(1, 10, 10).view(-1, 1)
    y = x ** 2
    
    cb = MockCallback()
    
    print(f"Engine initialized with {engine.n_islands} islands.")
    
    # Run
    engine.run(x, y, callback=cb)
    
    print("\nResults:")
    unique_islands = set(cb.reports)
    print(f"Islands reported in callback: {unique_islands}")
    
    if unique_islands == {0}:
        print("FAIL: Only Island 0 was reported (likely hardcoded).")
    elif len(unique_islands) > 1:
        print("SUCCESS: Multiple islands reported.")
    else:
        print("INCONCLUSIVE: No best found or weird reporting.")

if __name__ == "__main__":
    test_islands_reporting()
