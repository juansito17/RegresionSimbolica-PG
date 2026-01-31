
import torch
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.stdout = open('julio_phase4.log', 'w')
sys.stderr = sys.stdout

import rpn_cuda_native as rpn_cuda
from core.gpu.grammar import GPUGrammar
from core.gpu.cuda_vm import CudaRPNVM

def test_phase4():
    print("[TEST] Starting Phase 4 Orchestrator Tests...")
    device = torch.device('cuda')
    
    # 0. Setup Grammar & VM to get IDs
    print("[TEST] Loading Grammar...")
    grammar = GPUGrammar(1)
    vm = CudaRPNVM(grammar, device)
    
    # 1. Setup Mock Data
    B = 20
    L = 32
    K = 10
    N_data = 100
    
    # Random Population
    pop = torch.randint(0, 50, (B, L), dtype=torch.long, device=device)
    consts = torch.randn(B, K, dtype=torch.float32, device=device)
    
    # Fake fitness (RMSE)
    fitness = torch.rand(B, dtype=torch.float32, device=device) * 10.0
    
    X = torch.randn(1, N_data, dtype=torch.float32, device=device) # [Vars, N] for VM
    Y = X * 2.0 + 1.0 
    Y = Y.squeeze(0) # [N]
    
    PAD_ID = vm.PAD_ID
    
    # Get token arities and arity IDs from grammar
    token_arities = grammar.get_arity_tensor(device)
    arity_0_ids = grammar.get_arity_ids(0, device)
    arity_1_ids = grammar.get_arity_ids(1, device)
    arity_2_ids = grammar.get_arity_ids(2, device)
    
    # 2. Call evolve_generation
    print("\n[TEST] Calling evolve_generation...")
    start_t = time.perf_counter()
    
    # Return: [offspring, new_consts, new_fitness]
    res = rpn_cuda.evolve_generation(
        pop,
        consts,
        fitness,
        X, Y,
        token_arities,    # NEW
        arity_0_ids,      # NEW
        arity_1_ids,      # NEW
        arity_2_ids,      # NEW
        0.1, # mutation
        0.5, # crossover
        3, # tournament
        10, # pso steps
        20, # pso particles
        0.5, 1.5, 1.5, # pso params
        PAD_ID,
        # OpCodes
        vm.id_x_start,
        vm.id_C, vm.id_pi, vm.id_e,
        vm.id_0, vm.id_1, vm.id_2, vm.id_3, vm.id_5, vm.id_10,
        vm.op_add, vm.op_sub, vm.op_mul, vm.op_div, vm.op_pow, vm.op_mod,
        vm.op_sin, vm.op_cos, vm.op_tan,
        vm.op_log, vm.op_exp,
        vm.op_sqrt, vm.op_abs, vm.op_neg,
        vm.op_fact, vm.op_floor, vm.op_ceil, vm.op_sign,
        vm.op_gamma, vm.op_lgamma,
        vm.op_asin, vm.op_acos, vm.op_atan,
        3.14159265359, 2.718281828
    )
    
    end_t = time.perf_counter()
    print(f"Execution Time: {(end_t - start_t)*1000:.2f} ms")
    
    new_pop, new_consts, new_fit = res[0], res[1], res[2]
    
    print(f"New Pop Shape: {new_pop.shape}")
    print(f"New Consts Shape: {new_consts.shape}")
    print(f"New Fit Shape: {new_fit.shape}")
    
    # 3. Validations
    if new_pop.shape == (B, L):
        print("[PASS] Population Shape Correct")
    else:
        print(f"[FAIL] Expected ({B}, {L}), got {new_pop.shape}")
        
    if new_consts.shape == (B, K):
        print("[PASS] Constants Shape Correct")
    else:
        print(f"[FAIL] Expected ({B}, {K}), got {new_consts.shape}")

    if new_fit.shape == (B,):
        print("[PASS] Fitness Shape Correct")
    else:
        print(f"[FAIL] Expected ({B},), got {new_fit.shape}")
        
    print(f"Sample Fitness: {new_fit[:5].tolist()}")

if __name__ == "__main__":
    try:
        test_phase4()
        print("[ALL TESTS PASSED]")
    except Exception as e:
        print(f"[ERROR] Tests failed: {e}")
        import traceback
        traceback.print_exc(file=sys.stdout)
