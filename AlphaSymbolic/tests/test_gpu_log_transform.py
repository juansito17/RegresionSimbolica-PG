
import torch
import numpy as np
from AlphaSymbolic.core.gpu.engine import TensorGeneticEngine
from AlphaSymbolic.core.gpu.config import GpuGlobals

def test_log_transformation():
    print("Testing USE_LOG_TRANSFORMATION implementation...")
    
    # 1. Setup
    GpuGlobals.USE_LOG_TRANSFORMATION = True
    engine = TensorGeneticEngine(num_variables=1)
    
    # Exponential growth data: y = exp(x)
    x_vals = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_vals = [1.0, 2.718281828, 7.389056099, 20.08553692, 54.59815003]
    
    # Add some garbage to test filtering
    x_vals += [-1.0, -2.0]
    y_vals += [0.0, -5.0]
    
    print(f"Original Y: {y_vals}")
    
    # We want to see if run transforms y to [0, 1, 2, 3, 4]
    # Since run is a loop, we can wrap it or just mock it.
    # But better to just check the preprocessing step if possible.
    # We'll just run a very short search (timeout 1s) and check if it finds 'x' 
    # (since log(exp(x)) = x)
    
    print("Running short search with log transform...")
    best_formula = engine.run(x_vals, y_vals, seeds=[], timeout_sec=2)
    
    print(f"Best formula found: {best_formula}")
    
    # Also verify the console output was correct (manual check of the log)
    print("Test finished. Verify above output for 'Info: Log Transformation is ON' and 'Warning: Filtering out...'")

if __name__ == "__main__":
    test_log_transformation()
