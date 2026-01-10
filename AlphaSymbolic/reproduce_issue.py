
import torch
from core.gpu import TensorGeneticEngine

def reproduce():
    # Initialize engine
    engine = TensorGeneticEngine(num_variables=1)
    
    # Create a tensor representing: x0, PAD, x0, +
    # This should be equivalent to x0 + x0 = 2*x0
    # But current rpn_to_infix might show just "x0"
    
    x0_id = engine.grammar.token_to_id['x0']
    pad_id = 0
    plus_id = engine.grammar.token_to_id['+']
    
    # [x0, PAD, x0, +]
    rpn = torch.tensor([x0_id, pad_id, x0_id, plus_id], dtype=torch.long)
    
    print(f"RPN Tensor: {rpn}")
    
    infix = engine.rpn_to_infix(rpn)
    
    print(f"Decoded Infix: '{infix}'")
    
    expected = "(x0 + x0)"
    if infix == "x0":
        print("ISSUE REPRODUCED: Infix truncated at PAD.")
    elif infix == expected:
        print("ISSUE FIXED: Infix skipped PAD correctly.")
    else:
        print(f"Unexpected result: {infix}")

if __name__ == "__main__":
    reproduce()
