
import torch
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.stdout = open('julio_phase3.log', 'w')
sys.stderr = sys.stdout

import rpn_cuda_native as rpn_cuda

def test_phase3():
    print("[TEST] Starting Phase 3 Kernel Tests...")
    device = torch.device('cuda')
    
    # --- Test 1: Tournament Selection ---
    print("\n[TEST] 1. Tournament Selection")
    # Pop size 10, Tour size 3.
    # Fitness: [10, 9, 8, ... 1] (Index 9 has best fitness 1)
    fitness = torch.arange(10, 0, -1, dtype=torch.float32, device=device) # [10, 9, ... 1]
    # We want min fitness.
    
    B = 5
    K = 3
    # Random indices: [[0, 1, 2], [3, 4, 5], ...]
    rand_idx = torch.tensor([
        [0, 1, 2], # Fits: 10, 9, 8 -> Best 8 (Idx 2)
        [3, 4, 5], # Fits: 7, 6, 5 -> Best 5 (Idx 5)
        [9, 0, 1], # Fits: 1, 10, 9 -> Best 1 (Idx 9)
        [8, 8, 8], # Fits: 2, 2, 2 -> Best 2 (Idx 8)
        [0, 5, 9]  # Fits: 10, 5, 1 -> Best 1 (Idx 9)
    ], dtype=torch.long, device=device)
    
    selected = torch.zeros(B, dtype=torch.long, device=device)
    
    rpn_cuda.tournament_selection(fitness, rand_idx, selected)
    
    print(f"Fitness: {fitness.tolist()}")
    print(f"Candidates:\n{rand_idx.tolist()}")
    print(f"Selected: {selected.tolist()}")
    
    expected = [2, 5, 9, 8, 9]
    if selected.tolist() == expected:
        print("[PASS] Tournament Selection Correct")
    else:
        print(f"[FAIL] Expected {expected}, got {selected.tolist()}")

    # --- Test 2: PSO Update ---
    print("\n[TEST] 2. PSO Update")
    # V = w*V + c1*r1*(P-X) + c2*r2*(G-X)
    # X = X + V
    
    B = 2
    P = 2
    K = 2
    
    pos = torch.zeros((B, P, K), dtype=torch.float32, device=device) # 0
    vel = torch.ones((B, P, K), dtype=torch.float32, device=device)  # 1
    
    pbest = torch.full_like(pos, 10.0) # Target Pbest = 10
    gbest = torch.full((B, K), 20.0, device=device) # Target Gbest = 20
    
    r1 = torch.full_like(pos, 0.5)
    r2 = torch.full_like(pos, 0.5)
    
    w = 0.5
    c1 = 2.0
    c2 = 2.0
    
    # Expected V:
    # V_new = 0.5*1 + 2*0.5*(10-0) + 2*0.5*(20-0)
    #       = 0.5 + 10 + 20 = 30.5
    # X_new = 0 + 30.5 = 30.5
    
    rpn_cuda.pso_update(pos, vel, pbest, gbest, r1, r2, w, c1, c2)
    
    print(f"New Pos (Sample): {pos[0,0,0].item()}")
    print(f"New Vel (Sample): {vel[0,0,0].item()}")
    
    if abs(pos[0,0,0].item() - 30.5) < 1e-5:
        print("[PASS] PSO Update Logic Correct")
    else:
        print(f"[FAIL] Expected 30.5, got {pos[0,0,0].item()}")

    # --- Test 3: PSO Update Bests ---
    print("\n[TEST] 3. PSO Bests Update")
    # Current err: [[10, 5], [20, 30]] (B=2, P=2)
    # Pbest err: [[12, 4], [25, 25]]
    # Pbest pos: Zeros
    # Current pos: Ones
    
    curr_err = torch.tensor([[10.0, 5.0], [20.0, 30.0]], device=device)
    pbest_err = torch.tensor([[12.0, 4.0], [25.0, 25.0]], device=device)
    
    pbest_pos = torch.zeros((2, 2, 2), device=device)
    curr_pos = torch.ones((2, 2, 2), device=device)
    
    gbest_err = torch.tensor([100.0, 100.0], device=device)
    gbest_pos = torch.zeros((2, 2), device=device)
    
    rpn_cuda.pso_update_bests(curr_err, pbest_err, pbest_pos, curr_pos, gbest_err, gbest_pos)
    
    # Expected Pbest Err:
    # [min(10,12), min(5,4)] -> [10, 4]
    # [min(20,25), min(30,25)] -> [20, 25]
    print(f"Pbest Err: {pbest_err.tolist()}")
    
    expected_pbest = [[10.0, 4.0], [20.0, 25.0]]
    if pbest_err.allclose(torch.tensor(expected_pbest, device=device)):
        print("[PASS] PBest Update Correct")
    else:
        print("[FAIL] PBest Update Failed")
        
    # Expected Pbest Pos:
    # changed where curr < pbest (indices [0,0] and [1,0])
    # [0,1]: unchanged (4 < 5)
    # [1,1]: unchanged (25 < 30)
    # So [0,0]=1, [0,1]=0, [1,0]=1, [1,1]=0
    
    # Expected GBest Err:
    # B=0: min(10, 4) = 4. (From Pbest, which was updated? The kernel uses Updated Pbest for Gbest?)
    # Wait, the kernel function pso_update_gbest_kernel uses pbest_err.
    # We call pso_update_bests_kernel FIRST (updates pbest_err), THEN pso_update_gbest_kernel.
    # So Gbest should see [10, 4] -> min is 4.
    # B=1: [20, 25] -> min is 20.
    
    print(f"GBest Err: {gbest_err.tolist()}")
    expected_gbest = [4.0, 20.0]
    if gbest_err.allclose(torch.tensor(expected_gbest, device=device)):
        print("[PASS] GBest Update Correct")
    else:
        print("[FAIL] GBest Update Failed")


if __name__ == "__main__":
    try:
        test_phase3()
        print("[ALL TESTS PASSED]")
    except Exception as e:
        print(f"[ERROR] Tests failed: {e}")
        import traceback
        traceback.print_exc(file=sys.stdout)
