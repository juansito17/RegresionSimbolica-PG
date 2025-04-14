import numpy as np
from numba import cuda, float64

@cuda.jit
def gpu_evaluate_population(trees, xs, ts, fitness):
    idx = cuda.grid(1)
    if idx >= trees.shape[0]:
        return
    n_nodes = trees.shape[1]
    n_points = xs.shape[0]
    # Límite de seguridad para evitar desbordamiento de stack local
    if n_nodes > 256:
        fitness[idx] = 1e20  # Penalización alta
        return
    error_sum = 0.0
    for j in range(n_points):
        x = xs[j]
        # Evaluar árbol plano (no recursivo)
        stack = cuda.local.array(256, float64)  # Aumentado de 64 a 256
        sp = 0
        node_idx = 0
        res = 0.0
        for n in range(n_nodes):
            ntype = int(trees[idx, n, 0])
            if ntype == 0:
                res = trees[idx, n, 1]
            elif ntype == 1:
                res = x
            elif ntype == 2:
                op = int(trees[idx, n, 2])
                left = int(trees[idx, n, 3])
                right = int(trees[idx, n, 4])
                lval = trees[idx, left, 1] if left >= 0 else 0.0
                rval = trees[idx, right, 1] if right >= 0 else 0.0
                if op == 0:
                    res = lval + rval
                elif op == 1:
                    res = lval - rval
                elif op == 2:
                    res = lval * rval
                elif op == 3:
                    res = lval / rval if abs(rval) > 1e-9 else 1e20
                elif op == 4:
                    res = lval ** rval if not (lval == 0 and rval == 0) else 1.0
            stack[n] = res
        pred = stack[n_nodes-1]
        diff = abs(pred - ts[j])
        error_sum += diff ** 1.3
    fitness[idx] = error_sum

def evaluate_population_gpu(trees, xs, ts):
    pop_size = trees.shape[0]
    fitness = np.zeros(pop_size, dtype=np.float64)
    d_trees = cuda.to_device(trees)
    d_xs = cuda.to_device(xs)
    d_ts = cuda.to_device(ts)
    d_fitness = cuda.to_device(fitness)
    threads = 64
    blocks = (pop_size + threads - 1) // threads
    gpu_evaluate_population[blocks, threads](d_trees, d_xs, d_ts, d_fitness)
    return d_fitness.copy_to_host()
