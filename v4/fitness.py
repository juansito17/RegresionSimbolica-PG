# fitness.py
from typing import List
from expression_tree import Node, evaluate_tree, tree_size
from globals import INF, COMPLEXITY_PENALTY_FACTOR
import math
import numpy as np

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
