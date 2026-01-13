
import time
import torch
import numpy as np
import pandas as pd
from core.gp_bridge import GPEngine
from core.gpu.engine import TensorGeneticEngine

def benchmark():
    print("="*50)
    print("      BENCHMARK: GPU vs C++ GP Engine")
    print("="*50)

    # 1. Setup Data
    # Simple x^2 + x + 1
    x_test = np.linspace(-10, 10, 100)
    y_test = x_test**2 + x_test + 1
    
    # 2. Benchmark C++ (Baseline)
    print("\n[1] Running C++ Engine (Baseline)...")
    cpp_engine = GPEngine()
    
    start_time = time.time()
    # Using a simple seed to ensure it doesn't take forever searching if hard
    # But for benchmarking performance, let's let it run search.
    # We pass a short timeout so we measure speed of iterations or just total time for search?
    # C++ engine usually runs for fixed generations (e.g. 100/500).
    # We'll set a timeout that allows it to finish or at least start.
    # main.cpp says GENERATIONS is fixed (often 100 or 500).
    # We'll measure wall clock time.
    
    cpp_res = cpp_engine.run(x_test.tolist(), y_test.tolist(), seeds=[], timeout_sec=60)
    cpp_time = time.time() - start_time
    
    # Try to parse population size from output?
    # GPEngine.run returns formula string. It swallows stdout unless we modify it or capture it.
    # The current GPEngine swallows stdout but prints warnings.
    # We can rely on defaults or what we know. Global default is usually 1000-20000.
    # Let's assume C++ Pop is around 2000-10000.
    # We will simply report the TIME it took.
    print(f"    C++ Time: {cpp_time:.4f}s")
    print(f"    C++ Result: {cpp_res}")

    # 3. Benchmark GPU with varying sizes
    print("\n[2] Running GPU Engine (TensorGeneticEngine)...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"    Device: {device}")
    
    pop_sizes = [1000, 5000, 10000, 20000, 50000, 100000]
    results = []

    # Warmup
    print("    Warming up GPU...")
    warmup_eng = TensorGeneticEngine(device=device, pop_size=100, n_islands=1)
    warmup_eng.run(x_test, y_test, timeout_sec=2)

    for pop in pop_sizes:
        print(f"\n    --- Population: {pop} ---")
        # Ensure islands divide pop
        n_islands = 1
        if pop >= 10000: n_islands = 5
        
        eng = TensorGeneticEngine(device=device, pop_size=pop, n_islands=n_islands)
        
        # We want to measure Generations per Second or Total Time for Fixed Generations.
        # TensorGeneticEngine config normally runs for 500 gens.
        # But we can limit by timeout or we can hack Globals to reduce generations for test?
        # We'll use timeout to stop it if it's too long, but ideally we want to see speed.
        # Let's set a fixed short timeout (e.g. 5s) and invoke a callback to count generations!
        
        gen_count = [0]
        def callback(gen, rmse, rpn, consts, improved, _):
            gen_count[0] = gen
            
        t0 = time.time()
        # Ensure it doesn't stop immediately on finding solution by giving a harder target?
        # No, finding solution fast is also "performance". 
        # But for throughput (gens/sec), finding it instantly is bad metric.
        # Let's use a slightly noisy target so it searches.
        y_noisy = y_test + np.random.normal(0, 0.01, size=y_test.shape)
        
        eng.run(x_test, y_noisy, timeout_sec=5.0, callback=callback)
        dt = time.time() - t0
        
        gens = gen_count[0]
        gens_per_sec = gens / dt if dt > 0 else 0
        total_evals = gens * pop
        evals_per_sec = total_evals / dt if dt > 0 else 0
        
        print(f"    [GPU] Pop={pop}: {gens} gens in {dt:.2f}s => {gens_per_sec:.1f} gen/s | {evals_per_sec:.2e} eval/s")
        results.append({
            "Population": pop,
            "Generations": gens,
            "Time": dt,
            "Gen/Sec": gens_per_sec,
            "Eval/Sec": evals_per_sec
        })

    # 4. Report
    print("\n" + "="*50)
    print("Results Summary")
    print("="*50)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    print("\nFor comparison:")
    print(f"C++ Engine ran in {cpp_time:.4f}s (Population unknown/fixed, typically 1000-5000)")
    
if __name__ == "__main__":
    benchmark()
