import numpy as np
from numba import cuda, float64, int32
from gpu_tree import NODE_TYPE_CONST, NODE_TYPE_VAR, NODE_TYPE_OP, OP_MAP

@cuda.jit
def gpu_generate_random_trees(trees, max_depth, pop_size, max_nodes, rng_seed):
    idx = cuda.grid(1)
    if idx >= pop_size:
        return
    import math
    # Simple LCG for random numbers
    def lcg(seed):
        return (1664525 * seed + 1013904223) % 4294967296
    seed = rng_seed + idx
    for n in range(max_nodes):
        # Decide node type
        seed = lcg(seed)
        r = seed % 100
        if n == 0 or (n < (2 ** max_depth) // 2):
            # Root or upper nodes: operator
            node_type = NODE_TYPE_OP
            op = int(seed % 5)
            left = 2 * n + 1 if 2 * n + 1 < max_nodes else -1
            right = 2 * n + 2 if 2 * n + 2 < max_nodes else -1
            trees[idx, n, 0] = node_type
            trees[idx, n, 1] = 0.0
            trees[idx, n, 2] = op
            trees[idx, n, 3] = left
            trees[idx, n, 4] = right
        else:
            # Leaf: variable o constante
            node_type = NODE_TYPE_CONST if r < 50 else NODE_TYPE_VAR
            trees[idx, n, 0] = node_type
            if node_type == NODE_TYPE_CONST:
                trees[idx, n, 1] = float((seed % 10) + 1)
            else:
                trees[idx, n, 1] = 0.0
            trees[idx, n, 2] = 0
            trees[idx, n, 3] = -1
            trees[idx, n, 4] = -1

@cuda.jit
def gpu_mutate_trees(trees, mutation_rate, rng_seed):
    idx = cuda.grid(1)
    if idx >= trees.shape[0]:
        return
    import math
    def lcg(seed):
        return (1664525 * seed + 1013904223) % 4294967296
    seed = rng_seed + idx
    for n in range(trees.shape[1]):
        seed = lcg(seed)
        r = (seed % 100) / 100.0
        if r < mutation_rate:
            # Mutar tipo de nodo
            node_type = int(trees[idx, n, 0])
            if node_type == NODE_TYPE_CONST:
                trees[idx, n, 1] += float((seed % 5) - 2)  # Pequeño cambio
            elif node_type == NODE_TYPE_OP:
                trees[idx, n, 2] = int(seed % 5)  # Cambia operador

@cuda.jit
def gpu_crossover_trees(trees, parents_idx, rng_seed):
    idx = cuda.grid(1)
    if idx >= trees.shape[0]:
        return
    # Cruza subárboles entre dos padres
    parent1 = parents_idx[idx, 0]
    parent2 = parents_idx[idx, 1]
    cross_point = 1  # Simple: cruza a partir del nodo 1
    for n in range(cross_point, trees.shape[1]):
        trees[idx, n, :] = trees[parent2, n, :]
