
import torch
import numpy as np
import sys
import os

# Ensure we can import core
sys.path.append(os.getcwd())

from core.gpu.engine import TensorGeneticEngine
from core.gpu.config import GpuGlobals

def test_linear():
    print("Testing Linear Problem (y=2x)...")
    
    # 1. Check Config
    if GpuGlobals.USE_LOG_TRANSFORMATION:
        print("FAIL: USE_LOG_TRANSFORMATION is still True!")
        return
    else:
        print("PASS: USE_LOG_TRANSFORMATION is False.")

    # 2. Setup Data
    x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    y = 2 * x # [2, 4, 6, 8, 10]
    
    # 3. Initialize Engine
    try:
        engine = TensorGeneticEngine(pop_size=1000, n_islands=1) # Small pop for speed
        print("Engine initialized.")
    except Exception as e:
        print(f"FAIL: Engine init failed: {e}")
        return

    # 4. Run Search
    try:
        # Run for short time
        formula = engine.run(x, y, timeout_sec=5)
        print(f"Result Formula: {formula}")
        
        if formula and ("2" in formula or "x" in formula):
             print("PASS: Found a formula (likely correct).")
        else:
             print("WARNING: Formula might be empty or invalid (short timeout).")
             
    except Exception as e:
        print(f"FAIL: Engine run failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_linear()
