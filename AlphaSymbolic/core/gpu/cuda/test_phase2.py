
import torch
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.stdout = open('julio_phase2.log', 'w')
sys.stderr = sys.stdout

import rpn_cuda_native as rpn_cuda
from core.gpu.grammar import GPUGrammar, PAD_ID

def test_phase2():
    print("[TEST] Starting Phase 2 Kernel Tests...")
    device = torch.device('cuda')
    
    # Mock Grammar
    # 0: x0 (Term)
    # 1: + (Op2)
    # 2: sin (Op1)
    # 3: 1 (Term)
    # 4: PAD
    grammar_vocab = ['x0', '+', 'sin', '1']
    arities = [0, 2, 1, 0] # x0, +, sin, 1
    
    token_arities = torch.tensor(arities + [0], dtype=torch.int32, device=device) # +PAD
    # PAD_ID is 4
    PAD = 4
    
    # --- Test 1: Find Subtree Ranges ---
    print("\n[TEST] 1. Find Subtree Ranges")
    # Expression: sin(x0 + 1)
    # RPN: x0, 1, +, sin
    # IDs: 0, 3, 1, 2
    # Subtrees:
    # 0 (x0): range [0, 0]
    # 3 (1): range [1, 1]
    # 1 (+): needs 2 args. Arg2 (1) is [1,1]. Arg1 (x0) is [0,0]. Range [0, 2]
    # 2 (sin): needs 1 arg. Arg1 (+) is [0,2]. Range [0, 3]
    
    pop = torch.tensor([[0, 3, 1, 2, PAD]], dtype=torch.long, device=device)
    B, L = pop.shape
    starts = torch.full_like(pop, -1)
    
    rpn_cuda.find_subtree_ranges(pop, token_arities, starts, PAD)
    
    print(f"Population: {pop.tolist()}")
    print(f"Starts: {starts.tolist()}")
    
    expected = [0, 1, 0, 0, -1] # Last is PAD
    if starts[0].tolist() == expected:
        print("[PASS] Subtree Ranges Correct")
    else:
        print(f"[FAIL] Expected {expected}, got {starts[0].tolist()}")

    # --- Test 2: Mutation ---
    print("\n[TEST] 2. Mutation")
    # Mutate '0' (x0) -> check if it becomes valid terminal
    # Arity 0 IDs: [0, 3]
    arity_0 = torch.tensor([0, 3], dtype=torch.long, device=device)
    arity_1 = torch.tensor([2], dtype=torch.long, device=device)
    arity_2 = torch.tensor([1], dtype=torch.long, device=device)
    
    pop_mut = torch.zeros((100, 5), dtype=torch.long, device=device) # All x0
    rand_floats = torch.zeros((100, 5), dtype=torch.float32, device=device) # Force mutation
    rand_ints = torch.randint(0, 1000, (100, 5), dtype=torch.long, device=device)
    
    rpn_cuda.mutate_population(
        pop_mut, rand_floats, rand_ints, 
        token_arities, 
        arity_0, arity_1, arity_2, 
        0.5, # Rate
        PAD
    )
    
    # Check if elements are in [0, 3]
    unique = torch.unique(pop_mut)
    print(f"Mutated elements: {unique.tolist()}")
    if set(unique.tolist()).issubset({0, 3}):
        print("[PASS] Mutation preserves arity 0")
    else:
        print("[FAIL] Mutation produced invalid tokens")

    # --- Test 3: Crossover Splicing ---
    print("\n[TEST] 3. Crossover Splicing")
    # P1: A A A A A (0 0 0 0 0)
    # P2: B B B B B (3 3 3 3 3)
    # Cut: P1[1..2] replaced by P2[2..3]
    # S1=1, E1=2 (len=2)
    # S2=2, E2=3 (len=2)
    # Result C1: A(0), B(2), B(3), A(3), A(4)...
    # Wait, splicing logic:
    # C1 = P1[0..s1-1] + P2[s2..e2] + P1[e1+1..]
    #    = P1[0] + P2[2,3] + P1[3,4]
    #    = 0 + 3,3 + 0,0 
    #    = 0, 3, 3, 0, 0
    
    N = 1
    p1 = torch.zeros((N, 5), dtype=torch.long, device=device)
    p2 = torch.full((N, 5), 3, dtype=torch.long, device=device)
    
    s1 = torch.tensor([1], dtype=torch.long, device=device)
    e1 = torch.tensor([2], dtype=torch.long, device=device)
    s2 = torch.tensor([2], dtype=torch.long, device=device)
    e2 = torch.tensor([3], dtype=torch.long, device=device)
    
    c1 = torch.zeros_like(p1)
    c2 = torch.zeros_like(p2)
    
    rpn_cuda.crossover_splicing(p1, p2, s1, e1, s2, e2, c1, c2, PAD)
    
    print(f"Child 1: {c1.tolist()}")
    expected_c1 = [[0, 3, 3, 0, 0]]
    
    if c1.tolist() == expected_c1:
        print("[PASS] Crossover Splicing Correct")
    else:
        print(f"[FAIL] CExpected {expected_c1}, got {c1.tolist()}")

if __name__ == "__main__":
    try:
        test_phase2()
        print("[ALL TESTS PASSED]")
    except Exception as e:
        print(f"[ERROR] Tests failed: {e}")
        import traceback
        traceback.print_exc(file=sys.stdout)
