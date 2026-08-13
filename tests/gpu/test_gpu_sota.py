import torch
import time
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from warpsymbolic.gpu.grammar import GPUGrammar
from warpsymbolic.gpu.engine import TensorGeneticEngine

def test_jump_table_speed():
    """Benchmark the eval kernel to ensure jump table speed is fast."""
    print("--- Testing Jump Table Dispatch Speed ---")
    
    # 1. Setup Data
    num_vars = 2
    n_cases = 10000
    x = torch.rand((n_cases, num_vars), dtype=torch.float32, device="cuda") * 10 - 5
    # Dummy Target Y = V0 * 2.5 + sin(V1)
    y_target = x[:, 0] * 2.5 + torch.sin(x[:, 1])
    
    # 2. Setup Engine
    engine = TensorGeneticEngine(
        pop_size=5000, 
        max_len=30, 
        num_variables=num_vars,
        max_constants=5, 
        n_islands=1
    )
    
    # 3. Generating a population
    pop = engine.operators.generate_random_population(engine.pop_size)
    consts = torch.empty(engine.pop_size, engine.max_constants, dtype=torch.float32, device="cuda").uniform_(-10, 10)
    
    # 4. Warmup
    _ = engine.evaluator.evaluate_batch(pop, x.T.contiguous(), y_target, constants=consts)
    
    # 5. Timing
    n_iters = 50
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(n_iters):
        _ = engine.evaluator.evaluate_batch(pop, x.T.contiguous(), y_target, constants=consts)
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    duration = end_time - start_time
    evals_per_sec = (engine.pop_size * n_cases * n_iters) / duration
    print(f"Time for {n_iters} iterations: {duration:.4f} seconds")
    print(f"Evaluations per second: {evals_per_sec:.2e}")
    assert duration < 5.0, "Execution took too long; jump table likely not optimized correctly."
    print("Jump Table Speed Test Passed!\n")

def test_sbx_crossover():
    """Test the newly added SBX constant crossover functionality without crashing."""
    print("--- Testing SBX Constant Crossover ---")
    
    num_vars = 1
    n_cases = 100
    x = torch.rand((n_cases, num_vars), dtype=torch.float32, device="cuda") * 2
    y_target = x[:, 0] * 3.14
    
    # Enable all adaptive features to stress test them
    import warpsymbolic.gpu.config
    warpsymbolic.gpu.config.GpuGlobals.USE_CUDA_ORCHESTRATOR = True
    warpsymbolic.gpu.config.GpuGlobals.BASE_MUTATION_RATE = 0.5
    
    engine = TensorGeneticEngine(pop_size=1000, max_len=15, num_variables=num_vars, max_constants=3, n_islands=1)
    
    print("Running for 2 seconds to ensure SBX and adaptive parameters execute flawlessly...")
    best_formula = engine.run(
        x, 
        y_target,
        timeout_sec=2
    )
    
    print(f"Execution completed! Final Best Formula (gen 5): {best_formula}")
    print("SBX Crossover Test Passed!\n")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_jump_table_speed()
        test_sbx_crossover()
        print("All SOTA GPU Core tests passed successfully!")
    else:
        print("CUDA not available. Skipping GPU SOTA tests.")
