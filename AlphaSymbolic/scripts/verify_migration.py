
import sys
import os
import torch
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from AlphaSymbolic.core.gpu.engine import TensorGeneticEngine
from AlphaSymbolic.core.gpu.config import GpuGlobals

def verify_migration():
    print("--- Verifying Island Migration ---")
    
    # 1. Setup Engine with Migration Enabled
    pop_size = 1000
    n_islands = 10
    island_size = pop_size // n_islands
    # Migration happens every generation for this test
    GpuGlobals.MIGRATION_INTERVAL = 1 
    # Migration size: 10 individuals
    GpuGlobals.MIGRATION_SIZE = 10 
    
    engine = TensorGeneticEngine(pop_size=pop_size, n_islands=n_islands, device='cuda')
    engine.initialize_population()
    
    # buffers
    population = engine.pop_buffer_A
    pop_constants = engine.const_buffer_A
    
    # 2. Initialize Distinct "Species" (Constants) per Island
    print(f"Initializing {n_islands} islands with distinct species IDs (0.0 to {n_islands-1}.0)...")
    pop_constants.fill_(0.0)
    for i in range(n_islands):
        start = i * island_size
        end = start + island_size
        # Assign species ID = i.0 to all constants in this island
        pop_constants[start:end] = float(i)

    # Verify Initial State (Purity)
    print("Verifying initial purity...")
    for i in range(n_islands):
        start = i * island_size
        end = start + island_size
        island_consts = pop_constants[start:end, 0] # Check first constant
        if not torch.allclose(island_consts, torch.tensor(float(i), device='cuda')):
            print(f"FAIL: Initial setup failed for Island {i}")
            return
            
    print("Initial state pure. Running migration test...")
    
    # 3. Simulate Migration
    # We can call engine.migrate_islands directly to test the logic in isolation
    # Set fake fitness: Island i has constant i. 
    # We want Island i to be "Good" so it sends migrants?
    # Migration selects Best from Source.
    # Let's give Island i fitness = 0.0 (Perfect) so checked individuals are candidates.
    
    # Actually, migration picks top-k (smallest RMSE).
    # All individuals have same fitness?
    # No, we need to fake fitness.
    
    # Let's randomize fitness slightly so top-k is deterministic or random but valid.
    fake_fitness = torch.rand(pop_size, device='cuda') 
    
    # Perform Migration
    # Ring: i -> (i+1)%N
    # Island 0 should send to Island 1.
    # Island 1 should have some 0.0 constants afterwards.
    
    engine.migrate_islands(population, pop_constants, fake_fitness)
    
    # 4. Check for Migrants
    print("\nChecking for migrants in Target Islands...")
    total_migrants_found = 0
    
    for i in range(n_islands):
        target_island = (i + 1) % n_islands
        expected_species = float(i) # Comes from Source i
        
        start = target_island * island_size
        end = start + island_size
        
        island_consts = pop_constants[start:end, 0]
        
        # Count how many have expected_species
        # We need tolerance float check
        matches = (torch.abs(island_consts - expected_species) < 0.1).sum().item()
        
        print(f"Island {target_island} (Target) contains {matches} migrants from Island {i} (Source).")
        
        if matches > 0:
            total_migrants_found += matches
            
    if total_migrants_found > 0:
        print(f"\nSUCCESS: Found {total_migrants_found} migrants total.")
        print("Migration Logic is WORKING.")
    else:
        print("\nFAIL: No migrants moved. Migration Logic BROKEN.")

if __name__ == "__main__":
    verify_migration()
