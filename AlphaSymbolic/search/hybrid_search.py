import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional
import concurrent.futures
import os

from core.gp_bridge import GPEngine
from search.beam_search import BeamSearch, beam_solve
from core.grammar import ExpressionTree
try:
    from core.gpu_engine import TensorGeneticEngine
except ImportError:
    TensorGeneticEngine = None
    print("Warning: Could not import TensorGeneticEngine (PyTorch/CUDA missing?)")

def _run_gp_worker(args):
    """
    Worker function for Parallel GP execution.
    args: (x_list, y_list, seeds_chunk, gp_timeout, gp_binary_path)
    """
    x_list, y_list, seeds, timeout, binary_path = args
    import numpy as np
    from core.grammar import ExpressionTree
    engine = GPEngine(binary_path=binary_path)
    # Give each worker a slight timeout variance to avoid file lock collisions if using temp files
    # or just to spread load. But GPEngine handles unique tmp files so it should be fine.
    
    # Run GP
    result = engine.run(x_list, y_list, seeds, timeout_sec=timeout)
    
    # Evaluate immediately if result found to return RMSE for comparison
    if result:
        try:
             # Basic RMSE check for the worker's champion
            tree = ExpressionTree.from_infix(result)
            y_pred = tree.evaluate(np.array(x_list))
            mse = np.mean((np.array(y_list) - y_pred)**2)
            rmse = np.sqrt(mse)
            return {'formula': result, 'rmse': rmse, 'status': 'success'}
        except:
            return {'formula': result, 'rmse': 999.0, 'status': 'eval_error'}
    
    return {'formula': None, 'rmse': 1e9, 'status': 'failed'}

def hybrid_solve(
    x_values: np.ndarray,
    y_values: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    beam_width: int = 50,
    gp_timeout: int = 10,
    gp_binary_path: Optional[str] = None,
    max_workers: int = 4,
    num_variables: int = 1,
    extra_seeds: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Solves Symbolic Regression using a Hybrid Neuro-Evolutionary approach with Parallel GP.
    """
    
    print(f"--- Starting Alpha-GP Hybrid Search (Parallel Workers={max_workers}, Vars={num_variables}) ---")
    start_time = time.time()
    
    # 1. Neural Beam Search (Phase 1)
    print(f"[Phase 1] Neural Beam Search (Width={beam_width})...")
    neural_results = beam_solve(x_values, y_values, model, device, beam_width=beam_width, num_variables=num_variables)
    
    seeds = []
    
    # Inject Extra Seeds (Feedback Loop)
    if extra_seeds:
        print(f"[Phase 1] Injecting {len(extra_seeds)} external seeds (Feedback Loop).")
        seeds.extend(extra_seeds)
        
    if neural_results:
        print(f"[Phase 1] Found {len(neural_results)} candidates.")
        seen_formulas = set()
        for res in neural_results:
            f_str = res['formula']
            if f_str.startswith("Partial"): continue
            if f_str not in seen_formulas:
                seeds.append(f_str)
                seen_formulas.add(f_str)
        
        print(f"[Phase 1] Generated {len(seeds)} unique seeds for GP.")
        if len(seeds) > 0:
            print(f"Top Seed: {seeds[0]}")
    else:
        print("[Phase 1] No valid candidates found. Falling back to pure GP.")

    # 2. GP Refinement (Phase 2 - Heterogeneous CPU + GPU)
    print(f"[Phase 2] Genetic Refinement (Timeout={gp_timeout}s)...")
    
    x_list = x_values.tolist() if hasattr(x_values, 'tolist') else list(x_values)
    y_list = y_values.tolist() if hasattr(y_values, 'tolist') else list(y_values)
    
    # Determine Resources
    # Determine Resources
    use_gpu = (TensorGeneticEngine is not None) and (max_workers > 0)
    
    if use_gpu:
        num_cpu_workers = max(0, max_workers - 1)
    else:
        num_cpu_workers = max_workers
    
    print(f"[Phase 2] Resources: {num_cpu_workers} CPU Workers + {'1 GPU Worker' if use_gpu else '0 GPU Workers'}")

    results = []
    futures = []

    # A. Launch CPU Workers (Background)
    # ---------------------------------
    # Prepare chunks (Sniper strategy for CPU)
    cpu_seeds = list(seeds) # Copy
    cpu_chunks = []
    top_seeds = []
    
    if num_cpu_workers > 0:
        if not cpu_seeds:
            cpu_chunks = [[] for _ in range(num_cpu_workers)]
        else:
            top_seeds = cpu_seeds[:num_cpu_workers]
            for i in range(num_cpu_workers):
                seed_idx = i % len(top_seeds)
                cpu_chunks.append([top_seeds[seed_idx]])

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, num_cpu_workers)) as executor:
        if num_cpu_workers > 0:
            for chunk in cpu_chunks:
                args = (x_list, y_list, chunk, gp_timeout, gp_binary_path)
                futures.append(executor.submit(_run_gp_worker, args))
                
        # B. Run GPU Worker (Main Thread / Concurrent)
        # ------------------------------------------
        gpu_result = None
        if use_gpu:
            try:
                print("[Phase 2] Launching GPU Engine...")
                # We can run this in the main thread while threads handle CPU subprocesses
                # Or submit to executor if we want? 
                # Better to run in main thread to ensure CUDA context is happy and we monitor it.
                gpu_engine = TensorGeneticEngine(pop_size=50000, max_len=30, num_variables=num_variables)
                
                # Give GPU diverse seeds (ALL seeds, not just top 1)
                # GPU excels at breadth.
                gpu_best_formula = gpu_engine.run(x_list, y_list, seeds, timeout_sec=gp_timeout)
                
                if gpu_best_formula:
                    # Evaluate fitness (RMSE) using CPU logic for fairness/consistency
                    try:
                         tree = ExpressionTree.from_infix(gpu_best_formula)
                         y_pred = tree.evaluate(np.array(x_list))
                         mse = np.mean((np.array(y_list) - y_pred)**2)
                         rmse = np.sqrt(mse)
                         gpu_result = {'formula': gpu_best_formula, 'rmse': rmse, 'status': 'success', 'worker': 'GPU'}
                         print(f"[GPU Result] {gpu_best_formula} (RMSE: {rmse:.5f})")
                    except Exception as e:
                         print(f"[GPU Error] Eval failed: {e}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[GPU Error] Engine failed: {e}")

        # C. Collect CPU Results
        # ----------------------
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if res['status'] == 'success' or res['status'] == 'eval_error':
                    res['worker'] = 'CPU'
                    results.append(res)
            except Exception as e:
                print(f"CPU Worker exception: {e}")
                
        if gpu_result:
            results.append(gpu_result)

    total_time = time.time() - start_time

    # Find best result across all workers
    best_result = None
    best_rmse = float('inf')
    
    for res in results:
        if res['formula'] and res['rmse'] < best_rmse:
            best_rmse = res['rmse']
            best_result = res['formula']
            
    if best_result:
        print(f"--- Hybrid Search Completed in {total_time:.2f}s ---")
        print(f"Best Formula (Parallel): {best_result} (RMSE: {best_rmse:.5f})")
        
        return {
            'formula': best_result,
            'rmse': best_rmse,
            'source': 'Alpha-GP Hybrid',
            'time': total_time,
            'seeds_tried': top_seeds if seeds else []
        }
    else:
        print(f"--- Hybrid Search Failed (All workers failed) ---")
        return {
            'formula': None,
            'rmse': 1e9,
            'source': 'Alpha-GP Hybrid',
            'time': total_time,
            'seeds_tried': top_seeds if seeds else []
        }

if __name__ == "__main__":
    # Test
    class MockModel(torch.nn.Module):
        def forward(self, x, y, seq):
            bs, seq_len = seq.shape
            vocab = 20
            return torch.randn(bs, seq_len, vocab), None

    print("Testing Parallel Hybrid Search...")
    x = np.linspace(-5, 5, 20)
    y = x**2 - 5
    try:
        # Important: must protect entry point for multiprocessing on Windows
        res = hybrid_solve(x, y, MockModel(), torch.device("cpu"), beam_width=5, max_workers=2)
        print(res)
    except Exception as e:
        print(f"Test failed: {e}")
