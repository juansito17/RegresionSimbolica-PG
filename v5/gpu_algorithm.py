import numpy as np
from gpu_tree import flatten_tree, unflatten_tree
from gpu_operators import gpu_generate_random_trees, gpu_mutate_trees, gpu_crossover_trees
from gpu_fitness import evaluate_population_gpu
from numba import cuda

# Parámetros
POP_SIZE = 1000
MAX_NODES = 64
MAX_DEPTH = 6
N_GEN = 100
N_X = 32
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 5
RNG_SEED = 42

# Inicialización de población (en GPU, árboles planos)
def init_population(pop_size, max_nodes):
    trees = np.zeros((pop_size, max_nodes, 5), dtype=np.float64)
    threads = 64
    blocks = (pop_size + threads - 1) // threads
    gpu_generate_random_trees[blocks, threads](trees, MAX_DEPTH, pop_size, max_nodes, RNG_SEED)
    return trees

def tournament_selection(fitness, k=TOURNAMENT_SIZE):
    idxs = np.random.choice(len(fitness), k, replace=False)
    best = idxs[0]
    for i in idxs[1:]:
        if fitness[i] < fitness[best]:
            best = i
    return best

def run_evolution(xs, ts):
    trees = init_population(POP_SIZE, MAX_NODES)
    xs = np.array(xs, dtype=np.float64)
    ts = np.array(ts, dtype=np.float64)
    for gen in range(N_GEN):
        fitness = evaluate_population_gpu(trees, xs, ts)
        # Selección de padres (torneo)
        parents_idx = np.zeros((POP_SIZE, 2), dtype=np.int32)
        for i in range(POP_SIZE):
            parents_idx[i, 0] = tournament_selection(fitness)
            if np.random.rand() < CROSSOVER_RATE:
                parents_idx[i, 1] = tournament_selection(fitness)
            else:
                parents_idx[i, 1] = parents_idx[i, 0]
        # Cruce en GPU
        threads = 64
        blocks = (POP_SIZE + threads - 1) // threads
        gpu_crossover_trees[blocks, threads](trees, parents_idx, RNG_SEED + gen)
        # Mutación en GPU
        gpu_mutate_trees[blocks, threads](trees, MUTATION_RATE, RNG_SEED + gen)
        # Evaluar fitness de la nueva población
        fitness = evaluate_population_gpu(trees, xs, ts)
        best_idx = np.argmin(fitness)
        print(f"Gen {gen}: best fitness {fitness[best_idx]:.6f}")
    # Decodificar el mejor árbol plano a árbol recursivo
    best_tree_flat = trees[best_idx]
    best_tree = unflatten_tree(best_tree_flat)
    return best_tree, fitness[best_idx]
