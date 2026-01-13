
import torch
import numpy as np
import sys
from unittest.mock import MagicMock

# Needs to be able to import core
sys.path.append(".")

from core.gpu import TensorGeneticEngine
from core.model import AlphaSymbolicModel

def test_neural_integration():
    print("=== Testing Neural Integration ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Create Dummy Model
    vocab_size = 30 # Rough guess
    model = AlphaSymbolicModel(vocab_size=vocab_size)
    model.to(device)
    model.eval()
    
    # 2. Mock beam_solve to return known candidates
    # Because running real beam search might be slow or return garbage with untrained model
    # We want to test the ENGINE logic, not the model accuracy here.
    
    # We need to mock 'search.beam_search.beam_solve'
    # But it's imported inside the method 'neural_flash_injection'.
    # So we can't easily mock it via sys.modules unless we catch it.
    
    # Actually, let's just run it! 
    # The dummy model will output random tokens.
    # beam_solve might fail to parse them or produce "Partial".
    # engine handles that.
    
    # Let's verify it doesn't CRASH.
    
    x = np.linspace(1, 10, 10).astype(np.float32)
    y = x * 2 # Simple
    
    # Small pop, short run
    engine = TensorGeneticEngine(pop_size=100, max_len=20, device=device, model=model)
    
    # Force neural flash every 1 gen for testing
    # We can't easily change the hardcoded '% 50'.
    # But wait, we can just call the method manually to test it!
    
    print("Manual Neural Flash Test...")
    x_t = torch.tensor(x, device=device)
    y_t = torch.tensor(y, device=device)
    
    try:
        pop, consts = engine.neural_flash_injection(x_t, y_t)
        if pop is None:
            print("Neural Flash returned None (Expected with dummy model giving garbage tokens)")
        else:
            print(f"Neural Flash returned {len(pop)} candidates.")
            
    except Exception as e:
        print(f"FAILED: Neural Flash raised exception: {e}")
        import traceback
        traceback.print_exc()
        
    print("Run Loop Test (checking 50 gens logic, though we won't wait that long usually)")
    # Just run for 2 secs
    engine.run(x, y, timeout_sec=2)
    print("Run Loop finished without crash.")

if __name__ == "__main__":
    test_neural_integration()
