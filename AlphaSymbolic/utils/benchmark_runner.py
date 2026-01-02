import torch
import numpy as np
import time
import traceback
from search.mcts import MCTS
from data.benchmark_data import BENCHMARK_SUITE, get_benchmark_data
from utils.optimize_constants import optimize_constants

def run_benchmark_suite(model, device, progress_callback=None):
    """
    Runs the full benchmark suite.
    Args:
        model: Loaded AlphaSymbolic model
        device: Torch device
        progress_callback: Function(float, string) to update UI
        
    Returns:
        results: List of result dicts
        summary: Dict with aggregated stats
    """
    results = []
    
    # Configure MCTS for benchmark (balanced speed/accuracy)
    # 500 simulations is decent for benchmarking
    mcts = MCTS(model, device, max_simulations=500, lambda_mix=0.5, batch_size=32)
    
    total = len(BENCHMARK_SUITE)
    solved_count = 0
    
    for i, problem in enumerate(BENCHMARK_SUITE):
        if progress_callback:
            progress_callback(i / total, f"Testing: {problem['name']}...")
            
        x, y, _ = get_benchmark_data(problem['id'])
        
        start_time = time.time()
        
        # Run Search
        try:
            search_result = mcts.search(x, y)
             # Determine success
            # Success threshold: RMSE < 0.01 (or 1% relative error)
            rmse = search_result['rmse']
            is_solved = rmse < 0.05 # Looser threshold for general regression
            
            # Special check for exact integer symbolic match? No, RMSE is ground truth.
            
            elapsed = time.time() - start_time
            
            if is_solved:
                solved_count += 1
                status = "✅ SOLVED"
            else:
                status = "❌ FAILED"
                
            results.append({
                'id': problem['id'],
                'name': problem['name'],
                'level': problem['level'],
                'rmse': rmse,
                'time': elapsed,
                'status': status,
                'found_formula': search_result.get('formula', '???'),
                'is_solved': is_solved
            })
            
        except Exception as e:
            print(f"Error in benchmark {problem['name']}:")
            traceback.print_exc()
            results.append({
                'id': problem['id'],
                'name': problem['name'],
                'level': problem['level'],
                'rmse': 1e9,
                'time': 0,
                'status': "⚠️ ERROR",
                'found_formula': "Error",
                'is_solved': False
            })

    # Summary
    if progress_callback:
        progress_callback(1.0, "Done!")
        
    score = (solved_count / total) * 100
    summary = {
        'total': total,
        'solved': solved_count,
        'score': score,
        'avg_time': np.mean([r['time'] for r in results]) if results else 0
    }
    
    return results, summary
