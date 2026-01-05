"""
Diagnostic script for Hall of Shame 'Error' status.
"""
import torch
import numpy as np
from core.grammar import VOCABULARY, TOKEN_TO_ID
from core.model import AlphaSymbolicModel
from search.beam_search import BeamSearch

def debug_search():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VOCAB_SIZE = len(VOCABULARY)
    
    # Initialize model
    model = AlphaSymbolicModel(vocab_size=VOCAB_SIZE + 1, d_model=128).to(DEVICE)
    model.eval()
    
    # Mock data (Simple y = x + 0)
    x_val = np.linspace(-5, 5, 10).astype(np.float64)
    y_val = x_val + 0
    
    curriculum_stage = 0
    
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Stage: {curriculum_stage}")
    
    bs = BeamSearch(model, DEVICE, beam_width=1, max_length=20, curriculum_stage=curriculum_stage)
    
    print("\nRunning search...")
    res = bs.search(x_val, y_val)
    
    print(f"\nResult length: {len(res)}")
    if not res:
        print("FAIL: Search returned empty list!")
        # Investigate why. Let's trace one step of search.
        
        # Check token mask
        if bs.token_mask is not None:
            print(f"Mask counts: {torch.isfinite(bs.token_mask).sum().item()} allowed, "
                  f"{torch.isinf(bs.token_mask).sum().item()} disallowed")
            
            allowed_tokens = [VOCABULARY[i] for i in range(VOCAB_SIZE) if torch.isfinite(bs.token_mask[i])]
            print(f"Allowed tokens snippet: {allowed_tokens[:10]}...")
    else:
        print(f"SUCCESS: Found {res[0]['formula']}")

if __name__ == "__main__":
    debug_search()
