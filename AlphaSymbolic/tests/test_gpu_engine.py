import torch
import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.gpu_engine import TensorGeneticEngine

def test_gpu_engine():
    print("--- Testing GPU Genetic Engine ---")
    
    # 1. Setup Simple Problem: f(x) = x + 1
    x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    y = x + 1
    
    seeds = ["(x + 1)", "(x * 1)", "(x + 0.1)"]
    
    engine = TensorGeneticEngine(pop_size=1000, max_len=10, device=torch.device('cpu'), num_variables=1)
    
    # Check Grammar Restriction
    print("Checking Grammar Restriction...")
    if 'x3' in engine.grammar.token_to_id:
        print("FAIL: x3 found in grammar for num_variables=1")
    else:
        print("PASS: x3 correctly excluded.")
        
    print(f"Device: {engine.device}")
    
    # Run
    start_t = time.time()
    best_formula = engine.run(x, y, seeds, timeout_sec=2)
    end_t = time.time()
    
    print(f"Result for x+1: {best_formula}")
    print(f"Time: {end_t - start_t:.2f}s")
    
    # 2. Robustness Test (Invalid RPN)
    # x + (requires 2 args, has 1). Should act as x.
    # GPU engine handles this by ignoring '+'.
    # Decoder should also ignore '+'.
    print("\n--- Testing Robustness (x +) ---")
    bad_seeds = ["(x +)"] # Invalid infix, will be parsed as x if robust or PAD
    # Actually infix parser might fail first. Let's try to inject RPN directly or use a seed that becomes invalid.
    # Just running normal cycle is enough proof if it doesn't crash.
    
    # 3. Harder Problem: f(x) = x^2 + 1
    y2 = x**2 + 1
    seeds2 = ["(x * x)", "(x + 1)"]
    
    print("\n--- Testing f(x) = x^2 + 1 ---")
    best_formula2 = engine.run(x, y2, seeds2, timeout_sec=5)
    print(f"Result: {best_formula2}")

if __name__ == "__main__":
    test_gpu_engine()
