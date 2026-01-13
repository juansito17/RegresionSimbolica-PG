
import torch
import numpy as np
import sys
from unittest.mock import MagicMock

# Needs to be able to import core
sys.path.append(".")

from core.gpu import TensorGeneticEngine
from core.model import AlphaSymbolicModel

def test_mcts_integration():
    print("=== Testing MCTS Integration (Alpha Mode) ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Create Dummy Model
    # MCTS needs a model that outputs (logits, value)
    vocab_size = 30 
    model = AlphaSymbolicModel(vocab_size=vocab_size)
    model.to(device)
    model.eval()
    
    # 2. Setup Engine
    engine = TensorGeneticEngine(pop_size=100, max_len=20, device=device, model=model)
    
    # 3. Setup Data
    # Simple x^2
    x = np.linspace(-5, 5, 20).astype(np.float32)
    y = x**2
    
    x_t = torch.tensor(x, device=device)
    y_t = torch.tensor(y, device=device)
    
    # 4. Run Manual MCTS Refinement
    print("\n--- Manual Call to alpha_mcts_refinement ---")
    try:
        # This will internally import MCTS and run it for 50 simulations
        # Since model is untrained, it will produce random formulas or fail valid check
        # We just want to ensure it RUNS and returns something (None or Formula)
        formula = engine.alpha_mcts_refinement(x_t, y_t)
        
        if formula:
            print(f"MCTS returned formula: {formula}")
        else:
            print("MCTS returned None (Expected with random model not finding solution)")
            
        print("SUCCESS: Method executed without crash.")
        
    except Exception as e:
        print(f"FAILED: MCTS Execution crashed: {e}")
        import traceback
        traceback.print_exc()

    # 5. Verify Engine Loop Integration
    # We can't easily wait 100 generations, but we can verify config
    from core.gpu.config import GpuGlobals
    print(f"\nConfig Check: USE_PARETO_SELECTION = {GpuGlobals.USE_PARETO_SELECTION}")
    
    if GpuGlobals.USE_PARETO_SELECTION:
        print("SUCCESS: Pareto Selection is ENABLED.")
    else:
        print("WARNING: Pareto Selection is DISABLED.")

if __name__ == "__main__":
    test_mcts_integration()
