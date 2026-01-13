
import torch
import time
import numpy as np
from core.gpu import TensorGeneticEngine
from core.gpu.config import GpuGlobals

def benchmark(n_islands, pop_size=100000, generations=10):
    print(f"\n--- Benchmarking n_islands={n_islands} (Pop={pop_size:,}) ---")
    
    # Init Engine
    engine = TensorGeneticEngine(
        pop_size=pop_size, 
        n_islands=n_islands,
        max_len=30,
        num_variables=3
    )
    
    # Reset Globals for test
    GpuGlobals.GENERATIONS = generations
    GpuGlobals.PROGRESS_REPORT_INTERVAL = 1000 # Silence report
    
    # Dummy Data
    x = torch.linspace(1, 10, 100).view(-1, 1).to(engine.device)
    y = x ** 2
    
    # Warmup
    # print("Warmup...")
    # engine.initialize_population()
    
    start_time = time.perf_counter()
    
    # Run
    # callback=None to avoid print overhead
    engine.run(x, y, timeout_sec=None) 
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    evals = pop_size * generations
    eps = evals / duration
    
    print(f"Time: {duration:.4f}s")
    print(f"Speed: {eps:,.0f} Evals/sec")
    return eps

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Test 1: Baseline (Single Island / Panmictic)
        # Note: My refactor uses the loop 'for island in range(n_islands)'.
        # If n=1, it loops once. This mimics the 'old' behavior but with 1 loop overhead.
        speed_1 = benchmark(n_islands=1, pop_size=100000, generations=20)
        
        # Test 2: 40 Islands
        speed_40 = benchmark(n_islands=40, pop_size=100000, generations=20)
        
        # Comparison
        diff = (speed_1 - speed_40) / speed_1 * 100
        print(f"\nOverhead: {diff:.2f}% slowdown")
    else:
        print("Skipping benchmark (No GPU).")
