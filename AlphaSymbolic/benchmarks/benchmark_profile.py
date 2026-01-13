
import time
import torch
import numpy as np
import pandas as pd
from core.gpu.engine import TensorGeneticEngine

def benchmark_profile():
    print("="*50)
    print("      BENCHMARK PROFILE: GPU Engine")
    print("="*50)

    x_test = np.linspace(-10, 10, 100)
    y_test = x_test**2 + x_test + 1
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # pop_sizes = [1000, 10000, 50000]
    pop_sizes = [1000, 5000, 20000] # Smaller set for quick profile
    
    for pop in pop_sizes:
        print(f"\n--- Population: {pop} ---")
        n_islands = 1
        if pop >= 10000: n_islands = 5
        
        # 1. Initialization Time
        t0 = time.time()
        eng = TensorGeneticEngine(device=device, pop_size=pop, n_islands=n_islands)
        init_pop = eng.initialize_population()
        # Force constant init too as it happens in run
        pop_constants = torch.randn(pop, eng.max_constants, device=eng.device, dtype=torch.float64)
        if device == 'cuda': torch.cuda.synchronize()
        t_init = time.time() - t0
        print(f"  Init Time: {t_init:.4f}s")
        
        # 2. Components Profile (Single Pass)
        # Data
        x_t = torch.tensor(x_test, dtype=torch.float64, device=device).view(-1, 1)
        y_t = torch.tensor(y_test, dtype=torch.float64, device=device).view(-1)
        
        # A. Evaluation
        t0 = time.time()
        rmse = eng.evaluate_batch(init_pop, x_t, y_t, pop_constants)
        if device == 'cuda': torch.cuda.synchronize()
        t_eval = time.time() - t0
        print(f"  Eval Batch: {t_eval:.5f}s ({(pop/t_eval):.1f} inds/s)")
        
        # B. Constant Optimization (Top 200) - DISABLED
        t_opt = 0.0
        print(f"  Opt Consts: DISABLED (0.00000s)")

        # C. Operators (Crossover/Mutation)
        # Simulating generation step
        t0 = time.time()
        # Just call mutate for whole pop to see speed
        mut_pop = eng.operators.mutate_population(init_pop, 0.3)
        if device == 'cuda': torch.cuda.synchronize()
        t_ops = time.time() - t0
        print(f"  Mutation (Full Pop): {t_ops:.5f}s")
        
        # D. Deduplication
        t0 = time.time()
        eng.operators.deduplicate_population(init_pop, pop_constants)
        if device == 'cuda': torch.cuda.synchronize()
        t_dedup = time.time() - t0
        print(f"  Deduplication: {t_dedup:.5f}s")
        
        # E. Simplification (Sample of 100)
        # Note: Simplification is usually run on the whole population or huge chunks?
        # In engine.py run(), simplify_population is called periodically or on final?
        # Let's check a small sample to see unit cost.
        t0 = time.time()
        # Create simplifier temporarily if not init
        from core.gpu.simplification import GPUSimplifier
        simplifier = GPUSimplifier(eng.grammar, device=eng.device)
        # Simplify just 100 individuals to estimate
        sample_size = 100
        simplifier.simplify_population(init_pop[:sample_size], pop_constants[:sample_size])
        t_simp_100 = time.time() - t0
        est_simp_full = t_simp_100 * (pop / sample_size)
        print(f"  Simplification (Est. Full Pop): {est_simp_full:.5f}s (Based on 100 samples: {t_simp_100:.5f}s)")
        
        # 3. Full Step Estimate
        est_gen_time = t_eval + t_opt + t_ops + t_dedup # + est_simp_full (usually optional/periodic)
        print(f"  > Est. Time/Gen: {est_gen_time:.4f}s")
        print(f"  > Est. Gens/Sec: {(1.0/est_gen_time if est_gen_time>0 else 0):.2f}")
        
if __name__ == "__main__":
    benchmark_profile()
