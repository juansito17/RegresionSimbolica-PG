"""Debug crossover issue."""
import torch
import sys
import os

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from warpsymbolic.gpu.grammar import GPUGrammar
from warpsymbolic.gpu.operators import GPUOperators, RPN_CUDA_AVAILABLE, PAD_ID
from warpsymbolic.gpu.config import GpuGlobals

# Setup
grammar = GPUGrammar(num_variables=1)
device = torch.device('cpu')
ops = GPUOperators(grammar, device, pop_size=10, max_len=15, num_variables=1)

# Generate test population
population = ops.generate_random_population(4)  # Solo 4 para debug
print('Poblacion inicial:')
for i in range(4):
    rpn = population[i].tolist()
    non_pad = [x for x in rpn if x != 0]
    print(f'  Ind {i}: len={len(non_pad)} tokens')

# Validate initial
valid = ops._validate_rpn_batch(population)
print(f'\nValidos iniciales: {valid.sum().item()}/{len(valid)}')

# Manual crossover simulation
print('\n=== Manual crossover simulation ===')
print(f'RPN_CUDA_AVAILABLE: {RPN_CUDA_AVAILABLE}')

B, L = population.shape
n_pairs = int(B * 0.5 * 1.0)  # crossover_rate = 1.0
print(f'n_pairs: {n_pairs}')

perm = torch.randperm(B, device=device)
p1_idx = perm[:n_pairs*2:2]
p2_idx = perm[1:n_pairs*2:2]
print(f'p1_idx: {p1_idx.tolist()}, p2_idx: {p2_idx.tolist()}')

parents_1 = population[p1_idx]
parents_2 = population[p2_idx]

print(f'parents_1 shape: {parents_1.shape}')
print(f'parents_2 shape: {parents_2.shape}')

starts_1_mat = ops._get_subtree_ranges(parents_1)
starts_2_mat = ops._get_subtree_ranges(parents_2)

print(f'starts_1_mat:\n{starts_1_mat}')
print(f'starts_2_mat:\n{starts_2_mat}')

valid_mask_1 = (starts_1_mat != -1)
valid_mask_2 = (starts_2_mat != -1)

probs_1 = valid_mask_1.float() + 1e-6
probs_2 = valid_mask_2.float() + 1e-6

end_1 = torch.multinomial(probs_1, 1).squeeze(1)
end_2 = torch.multinomial(probs_2, 1).squeeze(1)

print(f'end_1: {end_1.tolist()}')
print(f'end_2: {end_2.tolist()}')

start_1 = starts_1_mat.gather(1, end_1.unsqueeze(1)).squeeze(1)
start_2 = starts_2_mat.gather(1, end_2.unsqueeze(1)).squeeze(1)

print(f'start_1: {start_1.tolist()}')
print(f'start_2: {start_2.tolist()}')

# Lengths
len_1_pre = start_1
len_1_sub = end_1 - start_1 + 1
len_2_pre = start_2
len_2_sub = end_2 - start_2 + 1

print(f'\nlen_1_pre: {len_1_pre.tolist()}')
print(f'len_1_sub: {len_1_sub.tolist()}')
print(f'len_2_pre: {len_2_pre.tolist()}')
print(f'len_2_sub: {len_2_sub.tolist()}')

# Test crossover
offspring = ops.crossover_population(population.clone(), crossover_rate=1.0)
valid_off = ops._validate_rpn_batch(offspring)
print(f'\nValidos despues de crossover: {valid_off.sum().item()}/{len(valid_off)}')

# Show invalid ones
invalid_mask = ~valid_off
if invalid_mask.any():
    print('\nEjemplos invalidos:')
    for i in invalid_mask.nonzero(as_tuple=True)[0][:2]:
        rpn = offspring[i].tolist()
        non_pad = [x for x in rpn if x != 0]
        print(f'  Ind {i}: len={len(non_pad)} tokens')
