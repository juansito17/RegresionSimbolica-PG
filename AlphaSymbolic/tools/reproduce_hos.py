
import torch
import numpy as np
import sys
import os
import traceback

# Ensure we can import from local dirs
sys.path.append(os.getcwd())

from ui.app_core import get_model
from search.beam_search import BeamSearch
from core.grammar import TOKEN_TO_ID, VOCABULARY, OPERATORS

def debug_trace(model, device, x_val, y_val):
    print("\n--- DEBUG TRACE (Greedy) ---")
    sos_id = len(VOCABULARY)
    seq = []
    open_count = 1
    
    print(f"Start: seq=[], open={open_count}")
    
    with torch.no_grad():
        for step in range(30): # Increase steps to see if it Loops
            # Prepare input
            input_seq = [sos_id] + [TOKEN_TO_ID[t] for t in seq]
            input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)
            x_tensor = torch.tensor(x_val, dtype=torch.float32).unsqueeze(0).to(device)
            y_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(0).to(device)
            
            logits, _ = model(x_tensor, y_tensor, input_tensor)
            last_logits = logits[0, -1, :len(VOCABULARY)]
            
            # Grease decode
            token_id = torch.argmax(last_logits).item()
            token = VOCABULARY[token_id]
            
            # Get top 3 probs
            probs = torch.softmax(last_logits, dim=0)
            top3_v, top3_i = torch.topk(probs, 3)
            top3_str = [f"{VOCABULARY[i.item()]}:{v.item():.2f}" for v, i in zip(top3_v, top3_i)]
            
            print(f"Step {step}: Pred '{token}' | Top3: {top3_str}")
            
            seq.append(token)
            if token in OPERATORS:
                open_count += OPERATORS[token] - 1
            else:
                open_count -= 1
            
            print(f"  -> seq={seq}, open={open_count}")
            
            if open_count == 0:
                print("  -> Completed!")
                break
            if open_count < 0:
                print("  -> Invalid (Too many closers)")
                break

def reproduce():
    print("Loading model...")
    try:
        model, device = get_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Model loaded on {device}")
    
    cases = [
        ("x - 5", lambda x: x - 5),
        ("x - 3", lambda x: x - 3),
        ("x - 1", lambda x: x - 1),
        ("2 * x", lambda x: 2 * x),
    ]
    
    x_val = np.linspace(-5, 5, 10).astype(np.float64)
    
    # Use stage 0 (Arithmetic) mask if tested model is pre-trained
    bs = BeamSearch(model, device, beam_width=1, max_length=20, curriculum_stage=0)
    
    for name, func in cases:
        print(f"\nTesting Case: {name}")
        y_val = func(x_val)
        
        try:
            results = bs.search(x_val, y_val, return_partial=True)
            
            if not results:
                print(f"RESULT: Search Empty for {name}")
                debug_trace(model, device, x_val, y_val)
            else:
                print(f"RESULT: {results[0]['formula']}")
        except Exception as e:
            print(f"EXCEPTION: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    reproduce()
