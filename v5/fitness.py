# fitness.py
from typing import List
from expression_tree import Node, evaluate_tree, tree_size
from globals import INF, COMPLEXITY_PENALTY_FACTOR
import math
import numpy as np
from numba import cuda, float64, int32
from gpu_tree import flatten_tree

def flatten_tree(tree):
    """Convierte un árbol en una lista de nodos planos para la GPU."""
    nodes = []
    def _rec(node):
        if node is None:
            return -1
        idx = len(nodes)
        if node.type.name == 'CONSTANT':
            nodes.append([0, node.value, 0, -1, -1])
        elif node.type.name == 'VARIABLE':
            nodes.append([1, 0.0, 0, -1, -1])
        elif node.type.name == 'OPERATOR':
            left = _rec(node.left)
            right = _rec(node.right)
            op_map = {'+': 0, '-': 1, '*': 2, '/': 3, '^': 4}
            op = op_map.get(node.op, 0)
            nodes.append([2, 0.0, op, left, right])
        return idx
    _rec(tree)
    return np.array(nodes, dtype=np.float64)

@cuda.jit
def gpu_evaluate_tree(nodes, x, out):
    idx = cuda.grid(1)
    if idx >= nodes.shape[0]:
        return
    stack = cuda.local.array(256, float64)
    sp = 0
    node_idx = 0
    stack[sp] = node_idx
    sp += 1
    res = 0.0
    while sp > 0:
        sp -= 1
        node_idx = int(stack[sp])
        ntype = int(nodes[node_idx, 0])
        if ntype == 0:
            res = nodes[node_idx, 1]
        elif ntype == 1:
            res = x[idx]
        elif ntype == 2:
            op = int(nodes[node_idx, 2])
            left = int(nodes[node_idx, 3])
            right = int(nodes[node_idx, 4])
            lval = res
            rval = res
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
        out[idx] = res

def batch_fitness(trees, targets, x_values):
    # Convierte todos los árboles a arrays planos
    flat_trees = [flatten_tree(t) for t in trees]
    xs = np.array(x_values, dtype=np.float64)
    ts = np.array(targets, dtype=np.float64)
    results = []
    for nodes in flat_trees:
        d_nodes = cuda.to_device(nodes)
        d_x = cuda.to_device(xs)
        out = cuda.device_array(xs.shape, dtype=np.float64)
        threads = 64
        blocks = (xs.shape[0] + threads - 1) // threads
        gpu_evaluate_tree[blocks, threads](d_nodes, d_x, out)
        preds = out.copy_to_host()
        diffs = np.abs(preds - ts)
        all_precise = np.all(diffs < 0.001)
        diffs_pow = np.power(diffs, 1.3)
        error_sum = np.sum(diffs_pow)
        if all_precise:
            error_sum *= 0.0001
        results.append(error_sum)
    return results

def batch_fitness_gpu(trees, targets, x_values):
    # Convierte todos los árboles a arrays planos
    flat_trees = [flatten_tree(t) for t in trees]
    max_nodes = max(arr.shape[0] for arr in flat_trees)
    pop_size = len(flat_trees)
    # Pad arrays para que todos tengan el mismo tamaño
    arr = np.zeros((pop_size, max_nodes, 5), dtype=np.float64)
    for i, t in enumerate(flat_trees):
        arr[i, :t.shape[0], :] = t
    xs = np.array(x_values, dtype=np.float64)
    ts = np.array(targets, dtype=np.float64)
    fitness = np.zeros(pop_size, dtype=np.float64)
    from gpu_fitness import gpu_evaluate_population
    threads = 64
    blocks = (pop_size + threads - 1) // threads
    d_arr = cuda.to_device(arr)
    d_xs = cuda.to_device(xs)
    d_ts = cuda.to_device(ts)
    d_fitness = cuda.to_device(fitness)
    gpu_evaluate_population[blocks, threads](d_arr, d_xs, d_ts, d_fitness)
    cuda.synchronize()  # Sincroniza para detectar errores de kernel
    return d_fitness.copy_to_host().tolist()

def calculate_raw_fitness(tree: Node, targets: List[float], x_values: List[float]) -> float:
    # Vectorized evaluation for speed
    try:
        xs = np.array(x_values)
        ts = np.array(targets)
        preds = np.array([evaluate_tree(tree, float(x)) for x in xs])
        if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
            return INF
        diffs = np.abs(preds - ts)
        all_precise = np.all(diffs < 0.001)
        try:
            diffs_pow = np.power(diffs, 1.3)
        except OverflowError:
            return INF
        if np.any(diffs_pow > 1e100):
            return INF
        error_sum = np.sum(diffs_pow)
        if error_sum > 1e100:
            return INF
        if all_precise:
            error_sum *= 0.0001
        if np.isnan(error_sum) or np.isinf(error_sum):
            return INF
        return float(error_sum)
    except Exception:
        return INF

def evaluate_fitness(tree: Node, targets: List[float], x_values: List[float]) -> float:
    raw_fitness = calculate_raw_fitness(tree, targets, x_values)
    if math.isinf(raw_fitness):
        return INF
    penalty = float(tree_size(tree)) * COMPLEXITY_PENALTY_FACTOR
    return raw_fitness * (1.0 + penalty)
