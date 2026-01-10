
import torch
import numpy as np
import sys
# Ensure we can import core
sys.path.append('.')

from core.gpu.engine import TensorGeneticEngine, GpuGlobals

def test_penalty():
    print("Testing Invalid Penalty...")
    # Initialize with device='cpu' or default if cuda available
    # The engine detects device automatically
    engine = TensorGeneticEngine(num_variables=1)
    
    # 1. Valid Formula: x0
    # RPN: x0
    # IDs: 1
    # Stack: [x0] -> sp=1 -> Valid
    
    # 2. Invalid Formula: x0 + (not enough operands)
    # RPN: x0 +
    # IDs: 1, 3 (assuming + is 3, checking grammar)
    
    op_add = engine.grammar.token_to_id['+']
    var_x0 = engine.grammar.token_to_id['x0']
    pad_id = engine.grammar.token_to_id['<PAD>']
    
    # Create population of 2: [Valid, Invalid]
    max_len = 5
    # Initialize with PAD
    pop = torch.full((2, max_len), pad_id, dtype=torch.long)
    
    # Individual 0: x0 (Valid)
    pop[0, 0] = var_x0
    
    # Individual 1: x0 + (Invalid, needs 2 ops, has 1)
    pop[1, 0] = var_x0
    pop[1, 1] = op_add
    
    # Dummy Data
    x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    y = x.clone() # Target = Input (Identity)
    
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        pop = pop.cuda()
    
    # Evaluate
    # Should get: [0.0, 1e150] (RMSE)
    # 1e150 is from sqrt(1e300)
    
    rmse = engine.evaluate_batch(pop, x, y)
    
    print(f"RMSE: {rmse}")
    
    success = True
    if rmse[0] < 1e-5:
        print("Valid formula score: OK")
    else:
        print(f"Valid formula score FAIL: {rmse[0]}")
        success = False
        
    if rmse[1] > 1e100:
        print("Invalid formula score: OK (Penalized)")
    else:
        print(f"Invalid formula score FAIL: {rmse[1]} (Should be huge)")
        success = False
        
    if success:
        print("TEST PASSED")
    else:
        print("TEST FAILED")

if __name__ == "__main__":
    test_penalty()
