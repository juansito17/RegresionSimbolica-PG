
import torch
import numpy as np
from core.gpu.engine import TensorGeneticEngine

def test_nan_penalty():
    print("Testing NaN Penalty vs Large Scalar Error...")
    
    # 1. Setup Engine
    engine = TensorGeneticEngine(pop_size=10, num_variables=1)
    
    # 2. Setup Data: Large targets (like in user request)
    # Target max ~ 2.2e15
    y_target = torch.tensor([2e15], device=engine.device, dtype=torch.float64)
    x = torch.tensor([1.0], device=engine.device, dtype=torch.float64)
    
    # 3. Create dummy population
    # Individual 0: Constant '2' (Valid but bad)
    # Individual 1: NaN (Invalid)
    
    # We need to manually construct the stack or just mock the evaluate_batch?
    # Easier to mock evaluate_batch internal behavior by creating a population that produces these.
    # Individual 0: '2' -> RPN: [id('2')]
    # Individual 1: '0/0' -> RPN: [id('0'), id('0'), id('/')] (Assumes we have 0, but we have 1, 2..?)
    # Let's use 'sqrt(-1)' if available, or just '0'/'0' if we can make 0. Start 'x'-'x' = 0?
    # engine.grammar has '1', '2'. '1'-'1' = 0.
    # So '1', '1', '-', '1', '1', '-', '/' should be 0/0.
    
    # Let's just construct the tensors manually for 'evaluate_batch'
    # Wait, evaluate_batch takes population tensor.
    
    ids = engine.grammar.token_to_id
    
    # Indiv 0: "2"
    ind0 = [ids['2']] + [0]*(engine.max_len-1)
    
    # Indiv 2: "1 1" -> sp=2 (Invalid)
    ind2 = [ids['1'], ids['1']] + [0]*(engine.max_len-2)
    
    pop = torch.tensor([ind0, ind2], device=engine.device, dtype=torch.long)
    
    # Evaluate
    rmse = engine.evaluate_batch(pop, x, y_target)
    
    print(f"Target: {y_target.item()}")
    print(f"Indiv 0 (Valid '2'): RMSE = {rmse[0].item()}")
    print(f"Indiv 1 (Invalid Stack): RMSE = {rmse[1].item()}")
    
    if rmse[1] < rmse[0]:
        print("\nFAILURE: Invalid formula has BETTER (lower) fitness than valid formula.")
    else:
        print("\nSUCCESS: Invalid formula has WORSE (higher) fitness.")
        
if __name__ == "__main__":
    try:
        test_nan_penalty()
    except Exception as e:
        print(f"Error: {e}")
