
import torch
import numpy as np
import time
import sys
import os
import gc

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.gpu.config import GpuGlobals
try:
    from core.gpu import TensorGeneticEngine
    from core.grammar import ExpressionTree
except ImportError as e:
    print(f"Error importing core modules: {e}")
    sys.exit(1)

def get_vram_info(device_idx=0):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device_idx) / 1024**3
        reserved = torch.cuda.memory_reserved(device_idx) / 1024**3
        return allocated, reserved
    return 0, 0

def format_mem(gb):
    return f"{gb:.2f} GB"

def run_stress_test():
    print("--- Optimized GPU Stress Test (Equilibrium Search) ---")
    
    if not torch.cuda.is_available():
        print("CUDA not available.")
        sys.exit(1)
        
    device_idx = 0
    device = torch.device(f'cuda:{device_idx}')
    device_props = torch.cuda.get_device_properties(device_idx)
    total_mem_gb = device_props.total_memory / 1024**3
    
    # SAFETY THRESHOLD: 3.8GB (Fixed for 4GB Card to prevent Swap)
    # PyTorch might allocate more (shared memory) but perform slowly.
    SAFETY_LIMIT_GB = 3.80
    
    print(f"Device: {torch.cuda.get_device_name(device_idx)}")
    print(f"Total VRAM: {format_mem(total_mem_gb)}")
    print(f"Safety Limit: {format_mem(SAFETY_LIMIT_GB)} (Hard Cutoff for Speed)")
    
    # Dummy Data for Eval
    x_val = np.arange(1, 26, dtype=np.float64)
    y_val = x_val ** 2
    
    # Incremental State
    cached_rpn_cpu = None
    cached_consts_cpu = None
    current_pop_size = 0
    final_max_stable_pop = 0

    # Steps to test
    pop_steps = [100_000, 500_000, 1_000_000, 1_500_000, 2_000_000, 2_250_000, 2_500_000, 2_750_000, 3_000_000, 3_500_000, 4_000_000, 5_000_000]
    
    print("\n--- Starting Equilibrium Search ---")
    print("Testing: Eval -> Islands -> Cycle (Mutation/Crossover)")

    for target_pop in pop_steps:
        print(f"\n[Testing Population: {target_pop:,}]")
        
        # 1. Check Pre-Conditions
        alloc, res = get_vram_info()
        if res > SAFETY_LIMIT_GB:
             print(f"  ABORT: Starting memory {format_mem(res)} exceeds limit.")
             break

        engine = None
        full_pop = None
        full_consts = None
        
        try:
            # 2. Setup / Expand Population
            # Clean GPU
            torch.cuda.empty_cache()
            
            # Init Engine with target size for buffers
            engine = TensorGeneticEngine(
                device=device,
                pop_size=target_pop,
                n_islands=40, # Test with target 40 islands
                max_len=30,
                num_variables=3,
                max_constants=5
            )

            # Incremental Gen Logic
            delta = target_pop - current_pop_size
            if delta > 0:
                # Alloc RAM for full
                # We do this carefully.
                
                # 1. Generate Delta on GPU
                # print(f"  (Generating {delta} new...)", end=" ")
                new_pop = engine.operators.generate_random_population(delta)
                new_const = torch.randn(delta, 5, device=device, dtype=torch.float64)
                
                # 2. Merge with Cache (on GPU)
                if cached_rpn_cpu is not None:
                     full_pop = torch.empty((target_pop, 30), dtype=torch.long, device=device)
                     full_consts = torch.empty((target_pop, 5), dtype=torch.float64, device=device)
                     
                     full_pop[:current_pop_size] = cached_rpn_cpu.to(device)
                     full_consts[:current_pop_size] = cached_consts_cpu.to(device)
                     full_pop[current_pop_size:] = new_pop
                     full_consts[current_pop_size:] = new_const
                     del new_pop, new_const
                else:
                    full_pop = new_pop
                    full_consts = new_const
            else:
                full_pop = cached_rpn_cpu[:target_pop].to(device)
                full_consts = cached_consts_cpu[:target_pop].to(device)

            # Update cache (CPU) immediately to save progress
            cached_rpn_cpu = full_pop.cpu()
            cached_consts_cpu = full_consts.cpu()
            current_pop_size = target_pop
            
            # ========================
            # STAGE 1: EVALUATION
            # ========================
            x_t = torch.tensor(x_val, device=device).unsqueeze(1)
            y_t = torch.tensor(y_val, device=device)
            
            t0 = time.perf_counter()
            engine.evaluate_batch(full_pop, x_t, y_t, full_consts)
            engine.evaluator.evaluate_batch_full(full_pop, x_t, y_t, full_consts) # Lexicase check
            t1 = time.perf_counter()
            
            alloc, res = get_vram_info()
            if res > SAFETY_LIMIT_GB:
                print(f"  FAIL: Eval usage {format_mem(res)} > Limit.")
                # Don't break loop, just stop this path? No, bigger pops will also fail. Break.
                break
            
            print(f"  [Eval] PASS ({format_mem(res)}, {(t1-t0)*1000:.1f}ms)", end=" | ")
            
            # ========================
            # STAGE 2: ISLANDS (Migration)
            # ========================
            # Test migration overhead
            fitness = torch.rand(target_pop, device=device)
            engine.migrate_islands(full_pop, full_consts, fitness)
            
            alloc, res = get_vram_info()
            if res > SAFETY_LIMIT_GB:
                print(f"\n  FAIL: Island Migration usage {format_mem(res)} > Limit.")
                break
            # print(f"Islands OK", end=" | ")

            # ========================
            # STAGE 3: CYCLE (Reproduction Peak)
            # ========================
            # Optimized "Micro-Batching" Simulation
            t0 = time.perf_counter()
            
            # 1. Chunked Crossover
            n_cross = int(target_pop * 0.5)
            # We don't need to store all offspring for the test, just prove we can generate them without OOM
            # But to be fair to memory usage, we should allocate the full result buffer (pre-allocation) implies
            # we hold the result.
            
            # Alloc destination (Pre-allocation simulation)
            # In engine we have next_pop. Here we simulate holding 2 full pops (Parent + Child)
            try:
                offspring_buffer = torch.empty((target_pop, 30), dtype=torch.long, device=device)
            except RuntimeError:
                print(f"\n  FAIL: Cannot allocate Offspring Buffer.")
                break
                
            chunk_size = 50000
            
            # Crossover Fill
            for i in range(0, n_cross, chunk_size):
                curr = min(chunk_size, n_cross - i)
                parents = full_pop[i : i+curr] # Dummy selection
                offspring_chunk = engine.operators.crossover_population(parents, 1.0)
                offspring_buffer[i : i+curr] = offspring_chunk
                del offspring_chunk, parents
                
            # Mutation Fill
            # The rest of buffer
            n_mut = target_pop - n_cross
            start_mut = n_cross
            
            for i in range(0, n_mut, chunk_size):
                curr = min(chunk_size, n_mut - i)
                parents = full_pop[start_mut + i : start_mut + i + curr]
                offspring_chunk = engine.operators.mutate_population(parents, 0.5)
                offspring_buffer[start_mut + i : start_mut + i + curr] = offspring_chunk
                del offspring_chunk, parents

            t1 = time.perf_counter()
            
            alloc, res = get_vram_info()
            
            if res > SAFETY_LIMIT_GB:
                print(f"\n  FAIL: Cycle Peak usage {format_mem(res)} > Limit.")
                break
            
            print(f"[Cycle] PASS ({format_mem(res)}, {(t1-t0)*1000:.1f}ms)")
            
            # SUCCESS
            final_max_stable_pop = target_pop
            
            # Cleanup step
            del engine
            del full_pop
            del full_consts
            del offspring_buffer
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  FAIL: OOM Exception.")
            else:
                print(f"\n  FAIL: {e}")
            break
        except Exception as e:
            print(f"\n  FAIL: {e}")
            break
            
    print(f"\n>>> FINAL EQUILIBRIUM: {final_max_stable_pop:,} Population <<<")
    
    # ========================
    # STAGE 4: MAX ISLANDS (Post-Equilibrium)
    # ========================
    if final_max_stable_pop > 0:
        print(f"\n--- Phase 2: Finding Max Islands for Pop {final_max_stable_pop:,} ---")
        
        # Incremental Island Steps
        island_steps = [40, 50, 64, 80, 100, 128, 200, 256, 500]
        max_stable_islands = 40
        
        # Ensure we have the data loaded (reuse cache)
        if cached_rpn_cpu is not None:
             full_pop = cached_rpn_cpu[:final_max_stable_pop].to(device)
             full_consts = cached_consts_cpu[:final_max_stable_pop].to(device)
             fitness = torch.rand(final_max_stable_pop, device=device)
        else:
             # Should not happen if loop ran, but safety fallback
             full_pop = torch.randint(0, 10, (final_max_stable_pop, 30), device=device)
             full_consts = torch.randn(final_max_stable_pop, 5, device=device)
             fitness = torch.rand(final_max_stable_pop, device=device)

        for n_isl in island_steps:
            if n_isl == 40 and max_stable_islands == 40: 
                print(f"Islands: {n_isl}... (Already Passed)")
                continue
                
            print(f"Testing Islands: {n_isl}...", end=" ")
            try:
                # Setup Engine (Migration uses buffers based on n_islands)
                engine = TensorGeneticEngine(
                    device=device,
                    pop_size=final_max_stable_pop,
                    n_islands=n_isl, 
                    max_len=30,
                    num_variables=3,
                    max_constants=5
                )
                
                t0 = time.perf_counter()
                
                # Test Migration (The memory stressor here involves index buffers)
                engine.migrate_islands(full_pop, full_consts, fitness)
                
                t1 = time.perf_counter()
                
                alloc, res = get_vram_info()
                if res > SAFETY_LIMIT_GB:
                    print(f"FAIL (Usage {format_mem(res)} > Limit)")
                    break
                
                print(f"PASS ({format_mem(res)} | Mig Time: {(t1-t0)*1000:.1f}ms)")
                max_stable_islands = n_isl
                del engine
                
            except Exception as e:
                print(f"FAIL ({e})")
                break
                
        print(f"\n>>> RECOMMENDED CONFIGURATION <<<")
        print(f"POP_SIZE = {final_max_stable_pop}")
        print(f"NUM_ISLANDS = {max_stable_islands}")

if __name__ == "__main__":
    run_stress_test()
