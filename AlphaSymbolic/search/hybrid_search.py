import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional

from core.gp_bridge import GPEngine
from search.beam_search import BeamSearch, beam_solve

def hybrid_solve(
    x_values: np.ndarray,
    y_values: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    beam_width: int = 50,
    gp_timeout: int = 10,
    gp_binary_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Solves Symbolic Regression using a Hybrid Neuro-Evolutionary approach.
    
    Phase 1: Neural Beam Search (The Brain)
             - Rapidly scans the search space.
             - Generates diverse, high-likelihood formula skeletons.
             
    Phase 2: Genetic Programming Refinement (The Muscle)
             - Takes the best skeletons from Phase 1.
             - Uses GPU-accelerated evolution to optimize constants and structure.
             - Runs for `gp_timeout` seconds.
             
    Returns:
        Best found formula result dict.
    """
    
    print(f"--- Starting Alpha-GP Hybrid Search ---")
    start_time = time.time()
    
    # 1. Neural Beam Search (Phase 1)
    print(f"[Phase 1] Neural Beam Search (Width={beam_width})...")
    # We use a larger beam width to ensure diversity for the GP
    # If the user requests beam_width=X, we might want to multiply it for the "seeds"
    # But let's stick to what is passed.
    
    neural_results = beam_solve(x_values, y_values, model, device, beam_width=beam_width)
    
    seeds = []
    if neural_results:
        print(f"[Phase 1] Found {len(neural_results)} candidates.")
        # Extract formulas tokens/string
        # neural_results is a list of dicts with 'formula' key (infix string)
        # GPEngine expects infix strings (e.g. "((x*x)+2)")
        
        # Filter for uniqueness and validity
        seen_formulas = set()
        for res in neural_results:
            f_str = res['formula']
            # Basic validation: must verify it's not a Partial result
            if f_str.startswith("Partial"): continue
            
            if f_str not in seen_formulas:
                seeds.append(f_str)
                seen_formulas.add(f_str)
        
        print(f"[Phase 1] Generated {len(seeds)} unique seeds for GP.")
        if len(seeds) > 0:
            print(f"Top Seed: {seeds[0]}")
    else:
        print("[Phase 1] No valid candidates found (Beam Search failed).")
        print("[Phase 1] Falling back to pure GP (Random Initialization).")
        seeds = []

    # 2. GP Refinement (Phase 2)
    print(f"[Phase 2] GPU Genetic Refinement (Timeout={gp_timeout}s)...")
    gp_engine = GPEngine(binary_path=gp_binary_path)
    
    # Run GP
    # We pass the seeds. GP engine handles the rest.
    # Ensure x_values and y_values are lists for gp_engine
    x_list = x_values.tolist() if hasattr(x_values, 'tolist') else list(x_values)
    y_list = y_values.tolist() if hasattr(y_values, 'tolist') else list(y_values)
    gp_result_str = gp_engine.run(x_list, y_list, seeds, timeout_sec=gp_timeout)
    
    total_time = time.time() - start_time
    
    if gp_result_str:
        print(f"--- Hybrid Search Completed in {total_time:.2f}s ---")
        print(f"Best Formula: {gp_result_str}")
        
        # Construct a result dict similar to Beam Search for consistency
        # Ideally we would evaluate it here to get RMSE, but GP output doesn't give us RMSE directly in a structured way (only stdout).
        # We can implement a quick evaluator if needed, or assume the user trusts the string.
        # For UI display, we probably want RMSE.
        
        return {
            'formula': gp_result_str,
            'rmse': 0.0, # Placeholder, will be evaluated by UI if needed or we can do it here
            'source': 'Alpha-GP Hybrid',
            'time': total_time
        }
    else:
        print(f"--- Hybrid Search Failed (GP did not return valid result) ---")
        return None

if __name__ == "__main__":
    # Test
    # Mock Model
    class MockModel(torch.nn.Module):
        def forward(self, x, y, seq):
            # Return random logits
            bs, seq_len = seq.shape
            vocab = 20
            return torch.randn(bs, seq_len, vocab), None

    print("Testing Hybrid Search...")
    x = np.linspace(-5, 5, 10)
    y = x**2
    try:
        res = hybrid_solve(x, y, MockModel(), torch.device("cpu"), beam_width=5)
        print(res)
    except Exception as e:
        print(f"Test failed: {e}")
