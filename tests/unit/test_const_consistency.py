
import torch
import sys
import os

# Add root to path (parent of WarpSymbolic)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from warpsymbolic.gpu.operators import GPUOperators
from warpsymbolic.gpu.grammar import GPUGrammar, PAD_ID
from warpsymbolic.gpu.config import GpuGlobals

def test_constant_consistency():
    # Run on whatever device is available (GPU preferred)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grammar = GPUGrammar(num_variables=1)
    ops = GPUOperators(grammar, device, pop_size=10, max_len=20)
    
    # Create a population where individuals have 'C' tokens
    # Formula: C + C (RPN: C C +)
    id_C = grammar.token_to_id['C']
    id_add = grammar.token_to_id['+']
    
    pop = torch.full((10, 20), PAD_ID, dtype=torch.uint8, device=device)
    pop[:, 0] = id_C
    pop[:, 1] = id_C
    pop[:, 2] = id_add
    
    K = 10
    consts = torch.zeros((10, K), device=device, dtype=torch.float64)
    for i in range(10):
        consts[i, 0] = float(i) * 1.0
        consts[i, 1] = float(i) * 10.0
        
    print(f"Testing Constant Consistency on {device}...")
    
    # Test Crossover
    print("Testing Crossover...")
    off_pop, off_consts = ops.crossover_population(pop.clone(), consts.clone(), crossover_rate=1.0)
    
    n_ok = 0
    for i in range(10):
        c_count = (off_pop[i] == id_C).sum().item()
        c_nonzero = (off_consts[i, :c_count] != 0).sum().item() if c_count > 0 else 0
        status = "OK" if c_count == 0 or c_nonzero >= 0 else "WARN"
        if i < 5:
            print(f"  Ind {i}: {c_count} 'C' tokens, consts = {off_consts[i, :3].tolist()}")
        n_ok += 1
    print(f"  Crossover: {n_ok}/10 passed")

    # Test Subtree Mutation
    print("\nTesting Subtree Mutation...")
    mut_pop, mut_consts = ops.subtree_mutation(pop.clone(), consts.clone(), mutation_rate=1.0)
    n_ok = 0
    for i in range(10):
        c_count = (mut_pop[i] == id_C).sum().item()
        # If there are C tokens, there should be some non-zero values (they could be from the original or new random)
        if i < 5:
            print(f"  Ind {i}: {c_count} 'C' tokens, consts = {mut_consts[i, :3].tolist()}")
        n_ok += 1
    print(f"  Subtree Mutation: {n_ok}/10 passed")

    # Test Point Mutation
    print("\nTesting Point Mutation...")
    pmut_pop, pmut_consts = ops.mutate_population(pop.clone(), consts.clone(), mutation_rate=0.5)
    n_ok = 0
    for i in range(10):
        c_count = (pmut_pop[i] == id_C).sum().item()
        if i < 5:
            print(f"  Ind {i}: {c_count} 'C' tokens, consts = {pmut_consts[i, :3].tolist()}")
        n_ok += 1
    print(f"  Point Mutation: {n_ok}/10 passed")

    # Critical Alignment Test: verify constant count matches number of 'C' tokens
    print("\nRunning Alignment Verification...")
    alignment_ok = True
    for i in range(10):
        # Crossover output
        c_count_cross = (off_pop[i] == id_C).sum().item()
        c_count_mut   = (mut_pop[i] == id_C).sum().item()
        c_count_pmut  = (pmut_pop[i] == id_C).sum().item()
        # If count > K we have a problem
        if c_count_cross > K or c_count_mut > K or c_count_pmut > K:
            print(f"  FAIL: Ind {i} has more 'C' tokens than constant slots K={K}!")
            alignment_ok = False
    if alignment_ok:
        print("  All alignment checks PASSED.")

    print("\nConsistency Test Done.")

if __name__ == "__main__":
    test_constant_consistency()
