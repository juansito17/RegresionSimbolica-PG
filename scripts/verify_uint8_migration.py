
import torch
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../AlphaSymbolic')))

from AlphaSymbolic.core.gpu.engine import TensorGeneticEngine
from AlphaSymbolic.core.gpu.config import GpuGlobals

def verify_uint8_migration():
    print("Starting uint8 migration verification...")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("CUDA not available! Cannot verify GPU implementation.")
        return

    # Initialize Engine (Small config for speed)
    print("Initializing Engine...")
    GpuGlobals.POP_SIZE = 100
    GpuGlobals.NUM_ISLANDS = 2
    GpuGlobals.GENERATIONS = 5
    
    engine = TensorGeneticEngine()
    
    # 1. Verify Grammar dtype
    print(f"Grammar dtype: {engine.grammar.dtype}")
    assert engine.grammar.dtype == torch.uint8, f"Expected torch.uint8, got {engine.grammar.dtype}"
    
    # 2. Verify Population Buffer dtype
    print(f"Population Buffer A dtype: {engine.pop_buffer_A.dtype}")
    assert engine.pop_buffer_A.dtype == torch.uint8, f"Expected torch.uint8, got {engine.pop_buffer_A.dtype}"
    
    print(f"Population Buffer B dtype: {engine.pop_buffer_B.dtype}")
    assert engine.pop_buffer_B.dtype == torch.uint8, f"Expected torch.uint8, got {engine.pop_buffer_B.dtype}"

    # 3. Create Dummy Data
    X = torch.rand(1, 50, device='cuda') # 1 variable, 50 samples (Must be <= 64 for Fused PSO)
    Y = X * 2 + 1 # Target: 2x + 1
    
    # 4. Run Evolution
    print("Running evolution for 5 generations...")
    # run returns the best formula string/object
    best_prog = engine.run(X, Y)
    
    print("Evolution complete.")
    print("Best Program:", best_prog)
    
    print("Verification SUCCESS!")

if __name__ == "__main__":
    try:
        verify_uint8_migration()
    except Exception as e:
        print(f"Verification FAILED: {e}")
        import traceback
        traceback.print_exc()
