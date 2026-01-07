import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional
import concurrent.futures
import os

from core.gp_bridge import GPEngine
from search.beam_search import BeamSearch, beam_solve
from core.grammar import ExpressionTree

def _run_gp_worker(args):
    """
    Worker function for Parallel GP execution.
    args: (x_list, y_list, seeds_chunk, gp_timeout, gp_binary_path)
    """
    x_list, y_list, seeds, timeout, binary_path = args
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
    max_workers: int = 4
) -> Dict[str, Any]:
    """
    Solves Symbolic Regression using a Hybrid Neuro-Evolutionary approach with Parallel GP.
    """
    
    print(f"--- Starting Alpha-GP Hybrid Search (Parallel Workers={max_workers}) ---")
    start_time = time.time()
    
    # 1. Neural Beam Search (Phase 1)
    print(f"[Phase 1] Neural Beam Search (Width={beam_width})...")
    neural_results = beam_solve(x_values, y_values, model, device, beam_width=beam_width)
    
    seeds = []
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

    # 2. GP Refinement (Phase 2 - Parallel)
    print(f"[Phase 2] Parallel GPU Genetic Refinement (Timeout={gp_timeout}s)...")
    
    x_list = x_values.tolist() if hasattr(x_values, 'tolist') else list(x_values)
    y_list = y_values.tolist() if hasattr(y_values, 'tolist') else list(y_values)
    
    # Prepare chunks for workers
    # STRATEGY: "Sniper Mode" (Top-K Seeding)
    # Instead of giving chunks of 50 seeds to each worker, we give the BEST unique seed to each worker.
    # This allows each worker to dedicate 100% of its resources to refining a high-probability candidate.
    
    if not seeds:
        # Fallback: Random Init for all workers
        chunks = [[] for _ in range(max_workers)]
    else:
        # Take Top-N seeds where N = max_workers
        top_seeds = seeds[:max_workers]
        
        # If we have fewer seeds than workers, repeat the best ones!
        # E.g. Worker 1 gets Top 1, Worker 2 gets Top 2... Worker 6 gets Top 1 again (to try a different evolutionary path)
        chunks = []
        for i in range(max_workers):
            seed_idx = i % len(top_seeds)
            # Important: Wrap in list because GP engine expects a list of seeds
            chunks.append([top_seeds[seed_idx]])

    print(f"[Phase 2] Sniper Strategy Active: Distributed {len(chunks)} top seeds across {max_workers} workers.")

    # Prepare arguments for each worker
    worker_args = []
    for chunk in chunks:
        worker_args.append((x_list, y_list, chunk, gp_timeout, gp_binary_path))
        
    results = []
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor
    # Since GPEngine uses subprocess.run, it releases GIL while waiting for the external binary.
    # This avoids Windows multiprocessing spawning issues (KeyError: __main__) with Gradio.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_worker = {executor.submit(_run_gp_worker, args): args for args in worker_args}
        
        for future in concurrent.futures.as_completed(future_to_worker):
            try:
                res = future.result()
                if res['status'] == 'success' or res['status'] == 'eval_error':
                    results.append(res)
            except Exception as e:
                print(f"Worker generated an exception: {e}")

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
