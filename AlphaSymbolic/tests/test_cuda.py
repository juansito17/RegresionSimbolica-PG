
import torch
import time
from core.gpu.engine import TensorGeneticEngine
from core.gpu.config import GpuGlobals

def test_cuda_vm():
    print("Initializing Engine...")
    engine = TensorGeneticEngine(pop_size=2000, n_islands=1)
    
    # Random Data
    print("Generating Data...")
    B = 2000
    Vars = 2
    Samples = 1000
    
    x = torch.randn(Vars, Samples, device='cuda', dtype=torch.float64)
    # y = sin(x0) + x1
    y_target = torch.sin(x[0]) + x[1]
    
    print("Initializing Population...")
    pop = engine.initialize_population()
    consts = torch.randn(B, 5, device='cuda', dtype=torch.float64)
    
    print("Running Evaluate Batch Full (Check for shape mismatch)...")
    start = time.time()
    
    try:
        # User reported crash in evaluate_batch_full
        # This function returns [Pop, Samples] error matrix, unlike evaluate_batch which returns [Pop] RMSE.
        abs_errs = engine.evaluator.evaluate_batch_full(pop, x, y_target, consts)
        print(f"Full Eval Shape: {abs_errs.shape}")
        assert abs_errs.shape == (B, Samples)
        print("Evaluate Batch Full Passed.")
        
        # Also check Differentiable
        loss = engine.evaluator.evaluate_differentiable(pop, consts, x, y_target)
        print(f"Diff Loss Shape: {loss.shape}")
        
    except Exception as e:
        print(f"Test Failed: {e}")
        raise e
        
    torch.cuda.synchronize()
    
    end = time.time()
    
    print(f"Execution Time (2k pop, 1k samples): {end - start:.4f} sec")
    print("SUCCESS: CUDA VM Evaluate Full Execution completed without errors.")

def stress_test():
    print("\n--- STRESS TEST: Finding Max Population ---")
    # We rely on chunking in evaluation.py (max_chunk_inds=2000)
    # The limit is mainly the Population tensor itself residing in VRAM.
    
    engine = TensorGeneticEngine(pop_size=2000, n_islands=1) # dummy
    Vars = 2
    Samples = 1000
    L = 50
    x = torch.randn(Vars, Samples, device='cuda', dtype=torch.float64)
    y_target = torch.sin(x[0]) + x[1]
    
    # Try increasing sizes
    sizes = [10_000, 100_000, 500_000, 1_000_000]
    
    for B in sizes:
        print(f"\nTesting Population: {B:,}")
        try:
            # Create Pop
            # In real engine, pop is int64 (long)
            pop = torch.randint(0, 10, (B, L), device='cuda', dtype=torch.long)
            consts = torch.randn(B, 5, device='cuda', dtype=torch.float64)
            
            torch.cuda.synchronize()
            t0 = time.time()
            
            # evaluate_batch handles chunking internally
            rmse = engine.evaluate_batch(pop, x, y_target, consts)
            
            torch.cuda.synchronize()
            dt = time.time() - t0
            
            print(f" -> SUCCESS. Time: {dt:.4f}s")
            
            # Check memory?
            # print(torch.cuda.memory_summary())
            
            del pop, consts, rmse
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f" -> OOM detected at {B:,}!")
            else:
                print(f" -> FAILED: {e}")
            break

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_cuda_vm()
        stress_test()
    else:
        print("CUDA not available.")
