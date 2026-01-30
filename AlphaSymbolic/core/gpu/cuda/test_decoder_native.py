
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.stdout = open('julio.log', 'w')
sys.stderr = sys.stdout

try:
    import rpn_cuda_native as rpn_cuda_final
    print("[TEST] Import successful!")
except ImportError as e:
    print(f"[TEST] Import failed: {e}")
    sys.exit(1)

def test_decoder():
    print("[TEST] Testing Native Decoder...")
    
    # Mock Grammar
    vocab = ["sin", "+", "x", "C", "log", "PAD"]
    # Arities: sin=1, +=2, x=0, C=0, log=1, PAD=0
    arities = [1, 2, 0, 0, 1, 0]
    PAD_ID = 5
    
    # 1. Test "sin(x + C)" -> RPN: x C + sin
    # Indices: 2, 3, 1, 0
    pop = torch.tensor([[2, 3, 1, 0, 5, 5]], dtype=torch.long)
    consts = torch.tensor([[3.14]], dtype=torch.float64) # C = 3.14
    
    decoded = rpn_cuda_final.decode_rpn(pop, consts, vocab, arities, PAD_ID)
    print(f"[TEST] Result: {decoded[0]}")
    
    expected = "sin_(x_0 + 3.1400)" # Using the formatting we defined in decoder.cpp
    # Note: decoder.cpp uses "sin(" directly, variable "x" maps to "x0".
    # Expected: "sin((x0 + 3.1400))" probably or similar based on parentheses logic
    
    # decoder.cpp logic:
    # x -> x0
    # C -> 3.1400
    # + -> (x0 + 3.1400)
    # sin -> sin((x0 + 3.1400))
    
    # Let's see actual output
    
    # 2. Test Invalid Stack
    # x + (requires 2 args, stack has 1)
    pop_inv = torch.tensor([[2, 1, 5, 5]], dtype=torch.long)
    decoded_inv = rpn_cuda_final.decode_rpn(pop_inv, consts, vocab, arities, PAD_ID)
    print(f"[TEST] Invalid Result: {decoded_inv[0]}")
    
    # 3. Test Batch
    pop_batch = torch.cat([pop, pop], dim=0)
    const_batch = torch.cat([consts, consts], dim=0)
    decoded_batch = rpn_cuda_final.decode_rpn(pop_batch, const_batch, vocab, arities, PAD_ID)
    print(f"[TEST] Batch Size: {len(decoded_batch)}")

if __name__ == "__main__":
    test_decoder()
