
import torch
import numpy as np
import time
import sys
import os
import gc
import psutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.gpu.config import GpuGlobals
try:
    # Use the same TensorGeneticEngine but we will initialize with device='cpu'
    from core.gpu import TensorGeneticEngine 
    from core.grammar import ExpressionTree
except ImportError as e:
    print(f"Error importing core modules: {e}")
    sys.exit(1)

def get_ram_info():
    """Returns (Used_GB, Total_GB, Percent)"""
    mem = psutil.virtual_memory()
    used_gb = mem.used / 1024**3
    total_gb = mem.total / 1024**3
    available_gb = mem.available / 1024**3
    return used_gb, total_gb, mem.percent, available_gb

def format_mem(gb):
    return f"{gb:.2f} GB"

def run_stress_test_cpu():
    print("--- Optimized CPU Stress Test (System RAM Capacity) ---")
    
    # Force CPU
    device = torch.device('cpu')
    
    # System Info
    used, total, percent, avail = get_ram_info()
    
    # Safety: Leave at least 2GB available
    SAFETY_THRESHOLD_AVAIL_GB = 2.0 
    
    print(f"Device: CPU (System RAM)")
    print(f"Total RAM: {format_mem(total)}")
    print(f"Available RAM: {format_mem(avail)}")
    print(f"Safety Cutoff: Stop if Available < {format_mem(SAFETY_THRESHOLD_AVAIL_GB)}")
    
    if avail < SAFETY_THRESHOLD_AVAIL_GB:
        print("WARNING: Low memory available. Test might fail early.")

    # Dummy Data for Eval
    # CPU is slower, so we keep points low to focus on memory stress?
    # No, we want realistic load. 25 points is fine.
    x_val = np.arange(1, 26, dtype=np.float64)
    y_val = x_val ** 2
    
    # Incremental State
    cached_rpn_cpu = None
    cached_consts_cpu = None
    current_pop_size = 0
    final_max_stable_pop = 0

    # Steps to test (RAM supports much more than VRAM)
    # 4GB VRAM -> 4M Pop.
    # 10GB Avail RAM -> ~10M Pop? (Maybe less due to Python overhead vs raw CUDA tensors)
    # Let's verify.
    pop_steps = [100_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 6_000_000, 8_000_000, 10_000_000, 12_000_000, 15_000_000]
    
    print("\n--- Starting Equilibrium Search (CPU) ---")
    print("Testing: Eval -> Islands -> Cycle (Mutation/Crossover)")

    for target_pop in pop_steps:
        print(f"\n[Testing Population: {target_pop:,}]")
        
        # 1. Check Pre-Conditions
        used, _, _, avail = get_ram_info()
        if avail < SAFETY_THRESHOLD_AVAIL_GB:
             print(f"  ABORT: Starting available memory {format_mem(avail)} is dangerously low.")
             break

        engine = None
        full_pop = None
        full_consts = None
        
        try:
            # 2. Setup / Expand Population
            gc.collect() # Python GC
            
            # Init Engine
            engine = TensorGeneticEngine(
                device=device,
                pop_size=target_pop,
                n_islands=40, # Same topology
                max_len=30,
                num_variables=3,
                max_constants=5
            )

            # Incremental Gen Logic (CPU RAM to CPU RAM is just resize/copy)
            delta = target_pop - current_pop_size
            
            t_gen_start = time.perf_counter()
            if delta > 0:
                print(f"  (Generating {delta} new...)", end=" ")
                # Generate new chunk
                new_pop = engine.operators.generate_random_population(delta)
                new_const = torch.randn(delta, 5, device=device, dtype=torch.float64)
                
                if cached_rpn_cpu is not None:
                     # Merge
                     full_pop = torch.cat([cached_rpn_cpu, new_pop], dim=0)
                     full_consts = torch.cat([cached_consts_cpu, new_const], dim=0)
                     del new_pop, new_const
                else:
                    full_pop = new_pop
                    full_consts = new_const
            else:
                full_pop = cached_rpn_cpu[:target_pop]
                full_consts = cached_consts_cpu[:target_pop]

            # Cache Update
            cached_rpn_cpu = full_pop
            cached_consts_cpu = full_consts
            current_pop_size = target_pop
            
            t_gen_end = time.perf_counter()
            # print(f"Gen Time: {(t_gen_end - t_gen_start):.2f}s")
            
            # Check Memory Post-Allocation
            used, _, _, avail = get_ram_info()
            if avail < SAFETY_THRESHOLD_AVAIL_GB:
                print(f"  FAIL: Allocation usage left {format_mem(avail)} < Limit.")
                break
            
            # ========================
            # STAGE 1: EVALUATION
            # ========================
            x_t = torch.tensor(x_val, device=device).unsqueeze(1)
            y_t = torch.tensor(y_val, device=device)
            
            t0 = time.perf_counter()
            # Standard Eval (Chunked internally in engine)
            engine.evaluate_batch(full_pop, x_t, y_t, full_consts)
            t1 = time.perf_counter()
            
            used, _, _, avail = get_ram_info()
            if avail < SAFETY_THRESHOLD_AVAIL_GB:
                print(f"  FAIL: Eval usage left {format_mem(avail)} < Limit.")
                break
            
            print(f"  [Eval] PASS (Avail: {format_mem(avail)}, {(t1-t0)*1000:.1f}ms)", end=" | ")
            
            # ========================
            # STAGE 2: ISLANDS
            # ========================
            fitness = torch.rand(target_pop, device=device)
            t0 = time.perf_counter()
            engine.migrate_islands(full_pop, full_consts, fitness)
            t1 = time.perf_counter()
            
            used, _, _, avail = get_ram_info()
            if avail < SAFETY_THRESHOLD_AVAIL_GB:
                print(f"\n  FAIL: Migration usage left {format_mem(avail)} < Limit.")
                break
            # print(f"Islands OK ({(t1-t0)*1000:.1f}ms)", end=" | ")

            # ========================
            # STAGE 3: CYCLE (Reproduction)
            # ========================
            t0 = time.perf_counter()
            
            # Alloc destination
            try:
                # We allocate buffer for next gen
                offspring_buffer = torch.empty((target_pop, 30), dtype=torch.long, device=device)
            except RuntimeError:
                print(f"\n  FAIL: Cannot allocate Offspring Buffer (OOM).")
                break
                
            n_cross = int(target_pop * 0.5)
            chunk_size = 50000
            
            # Crossover Fill
            for i in range(0, n_cross, chunk_size):
                curr = min(chunk_size, n_cross - i)
                parents = full_pop[i : i+curr]
                offspring_chunk = engine.operators.crossover_population(parents, 1.0)
                offspring_buffer[i : i+curr] = offspring_chunk
                del offspring_chunk, parents
                
            # Mutation Fill
            n_mut = target_pop - n_cross
            start_mut = n_cross
            for i in range(0, n_mut, chunk_size):
                curr = min(chunk_size, n_mut - i)
                parents = full_pop[start_mut + i : start_mut + i + curr]
                offspring_chunk = engine.operators.mutate_population(parents, 0.5)
                offspring_buffer[start_mut + i : start_mut + i + curr] = offspring_chunk
                del offspring_chunk, parents
                
            t1 = time.perf_counter()
            
            # Cleanup
            del offspring_buffer
            gc.collect()
            
            used, _, _, avail = get_ram_info()
            if avail < SAFETY_THRESHOLD_AVAIL_GB:
                print(f"\n  FAIL: Cycle usage left {format_mem(avail)} < Limit.")
                break
                
            print(f"  [Cycle] PASS (Avail: {format_mem(avail)}, {(t1-t0)*1000:.1f}ms)", end="") # No newline to keep line compact?
            
            final_max_stable_pop = target_pop
            
        except MemoryError:
            print(f"\n  FAIL: Python MemoryError.")
            break
        except Exception as e:
            print(f"\n  FAIL: {e}")
            break
        finally:
            # Partial cleanup if needed
            pass
            
    print(f"\n\n>>> MAX STABLE CPU POPULATION: {final_max_stable_pop:,} <<<")
    print(f"(Safety Cutoff: {format_mem(SAFETY_THRESHOLD_AVAIL_GB)} available RAM)")

if __name__ == "__main__":
    run_stress_test_cpu()
