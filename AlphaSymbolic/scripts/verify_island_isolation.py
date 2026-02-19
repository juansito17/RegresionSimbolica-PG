
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.gpu.engine import TensorGeneticEngine
from core.gpu.config import GpuGlobals

def verify_isolation():
    print("--- Verifying Island Isolation ---")
    
    # Setup
    pop_size = 1000
    n_islands = 10
    island_size = pop_size // n_islands
    
    # Initialize Engine
    engine = TensorGeneticEngine(pop_size=pop_size, n_islands=n_islands, device='cuda')
    engine.initialize_population()
    
    # Force C++ Orchestrator
    GpuGlobals.USE_CUDA_ORCHESTRATOR = True
    GpuGlobals.CUDA_ORCHESTRATOR_ONLY = True
    
    # 1. Create Distinct Species
    # We will set the first token of each individual to a unique value per island.
    # Token 0 is chosen to be a constant index for simplicity, or just a dummy op.
    # Let's use the 'constants' vector.
    # Island i gets constant vector filled with value i.
    
    # Use Buffer A as initial population
    population = engine.pop_buffer_A
    pop_constants = engine.const_buffer_A
    
    pop_constants.fill_(0.0)
    for i in range(n_islands):
        start = i * island_size
        end = start + island_size
        # Set constant 0 to float(i)
        pop_constants[start:end, 0] = float(i)
        
    # 2. Run Evolve Generation (Prohibit Mutation to track Selection/Crossover only)
    # We want to see if parents form different islands were selected.
    # If Island 0 has parents from Island 1, the constant will be 1.0.
    
    # We need to disable mutation and crossover mixing?
    # Crossover between two parents from Island 0 -> Child has constant 0.
    # Crossover between Island 0 and Island 1 -> Child has constant 0 or 1 (or mix).
    # If isolation works, Parent 1 and Parent 2 MUST be from same island.
    # So if we are in Island 0, we should ONLY see constant 0.0.
    
    print("Running 1 generation with mutation_rate=0, crossover=1.0...")
    
    # Mock data
    x = torch.randn(1, 10).cuda()
    y = x
    engine.fitness_rmse = torch.ones(pop_size, device='cuda') # Equal fitness to ensure random selection
    engine.abs_errors = torch.zeros(pop_size, 10).cuda()
    
    # Run
    # We access the internal evolve_generation_cuda directly to bypass migration logic in engine.evolve()
    print("Calling evolve_generation_cuda...")
    print(f"Shapes: Pop={population.shape}, Const={pop_constants.shape}, Fit={engine.fitness_rmse.shape}, Err={engine.abs_errors.shape}")
    print(f"Stats: n_islands={n_islands}, tourn={3}")
    
    result = engine.evolve_generation_cuda(
        population,
        pop_constants,
        engine.fitness_rmse,
        engine.abs_errors,
        x, y,
        None, # mutation bank
        mutation_rate=0.0, # No mutation
        crossover_rate=1.0, # Full crossover
        tournament_size=3,
        pso_steps=0,         # Disable PSO
        pso_particles=0      # Disable PSO
    )
    
    new_pop, new_c, new_fit = result
    
    if new_c is None:
        print("Test Failed: CUDA Orchestrator did not run (returned None).")
        return

    # 3. Verify Species Intactness
    errors = 0
    for i in range(n_islands):
        start = i * island_size
        end = start + island_size
        
        # Check constants in this island of the NEW population
        # They should all be close to float(i)
        # Note: If crossover happened, constants might be swapped, but both parents are species i, so constants are i.
        # Unless constants are mutated? 
        # rpn_kernels.cu mutation might mutate constants? 
        # We set mutation_rate=0.
        
        island_c = new_c[start:end, 0]
        
        # Count how many are NOT i
        # Use a tolerance for float comparison
        mask_bad = (island_c - float(i)).abs() > 0.1
        n_bad = mask_bad.sum().item()
        
        if n_bad > 0:
            print(f"Island {i}: Found {n_bad} aliens! (Expected {i}.0)")
            # Show some examples
            print(f"  Examples: {island_c[mask_bad][:5].tolist()}")
            errors += n_bad
        else:
            print(f"Island {i}: 100% Pure.")

    if errors == 0:
        print("SUCCESS: Island Isolation Verified!")
    else:
        print(f"FAILURE: {errors} cross-island contaminations detected.")

if __name__ == "__main__":
    verify_isolation()
