# genetic_operators.py
from typing import List, Optional
from expression_tree import Node, NodeType, clone_tree, collect_node_ptrs
from globals import INF, MAX_TREE_DEPTH_INITIAL, get_rng
import random
import math
from numba import njit, prange

class Individual:
    def __init__(self, tree: Node):
        self.tree = tree
        self.fitness = INF
        self.fitness_valid = False
    def __lt__(self, other):
        return self.fitness < other.fitness

def generate_random_tree(max_depth: int, current_depth: int = 0) -> Node:
    rng = get_rng()
    # Increase probability of terminal node as depth increases
    terminal_prob = 0.2 + 0.8 * (float(current_depth) / max_depth)
    if current_depth >= max_depth or rng.random() < terminal_prob:
        # Terminal node: 75% variable, 25% constant
        if rng.random() < 0.75:
            node = Node(NodeType.VARIABLE)
            return node
        else:
            node = Node(NodeType.CONSTANT)
            node.value = rng.randint(1, 10)
            return node
    else:
        node = Node(NodeType.OPERATOR)
        ops = ['+', '-', '*', '/', '^']
        weights = [0.3, 0.3, 0.25, 0.1, 0.05]
        node.op = random.choices(ops, weights)[0]
        if node.op == '^':
            node.left = generate_random_tree(max_depth, current_depth + 1)
            right = Node(NodeType.CONSTANT)
            right.value = rng.randint(2, 4)
            node.right = right
        else:
            node.left = generate_random_tree(max_depth, current_depth + 1)
            node.right = generate_random_tree(max_depth, current_depth + 1)
        if not node.left:
            node.left = generate_random_tree(max_depth, current_depth + 1)
        if not node.right:
            node.right = generate_random_tree(max_depth, current_depth + 1)
        return node

@njit(parallel=True)
def create_initial_population_parallel(population_size, max_depth):
    pop = []
    for i in prange(population_size):
        pop.append(generate_random_tree(max_depth))
    return pop

def create_initial_population(population_size: int) -> List[Individual]:
    rng = get_rng()
    population = []
    for _ in range(population_size):
        depth = rng.randint(3, MAX_TREE_DEPTH_INITIAL)
        population.append(Individual(generate_random_tree(depth)))
    return population

def tournament_selection(population: List[Individual], tournament_size: int) -> Individual:
    if not population:
        raise RuntimeError('Cannot perform tournament selection on empty population.')
    tournament_size = max(1, min(tournament_size, len(population)))
    rng = get_rng()
    best = rng.choice(population)
    for _ in range(1, tournament_size):
        contender = rng.choice(population)
        if not best.fitness_valid or (contender.fitness_valid and contender.fitness < best.fitness):
            if not contender.fitness_valid:
                raise RuntimeError('Tournament selection encountered individual with invalid fitness.')
            best = contender
    if not best.fitness_valid:
        raise RuntimeError('Tournament selection failed to find individual with valid fitness.')
    return best

def simplify_tree(node: Node) -> Node:
    # Simplificaciones básicas: x*1=x, x+0=x, 0*x=0, x-0=x, x/1=x, 0+x=x, etc.
    if node is None or node.type != NodeType.OPERATOR:
        return node
    node.left = simplify_tree(node.left)
    node.right = simplify_tree(node.right)
    # Multiplicación por 1 o 0
    if node.op == '*':
        if node.left and node.left.type == NodeType.CONSTANT:
            if node.left.value == 0:
                return Node(NodeType.CONSTANT)
            if node.left.value == 1:
                return node.right
        if node.right and node.right.type == NodeType.CONSTANT:
            if node.right.value == 0:
                return Node(NodeType.CONSTANT)
            if node.right.value == 1:
                return node.left
    # Suma/resta por 0
    if node.op in ['+', '-']:
        if node.right and node.right.type == NodeType.CONSTANT and node.right.value == 0:
            return node.left
        if node.left and node.left.type == NodeType.CONSTANT and node.left.value == 0 and node.op == '+':
            return node.right
    # División por 1
    if node.op == '/':
        if node.right and node.right.type == NodeType.CONSTANT and node.right.value == 1:
            return node.left
    return node

def mutate_tree(tree: Node, mutation_rate: float, max_depth: int) -> Node:
    rng = get_rng()
    if rng.random() >= mutation_rate:
        return clone_tree(tree)
    new_tree = clone_tree(tree)
    nodes: List[Node] = []
    collect_node_ptrs(new_tree, nodes)
    if not nodes:
        return new_tree
    node_to_mutate = rng.choice(nodes)
    mutation_types = ['ConstantChange', 'OperatorChange', 'SubtreeReplace', 'NodeInsertion']
    mut_type = rng.choice(mutation_types)
    if mut_type == 'ConstantChange':
        if node_to_mutate.type == NodeType.CONSTANT:
            if rng.random() < 0.5:
                node_to_mutate.value *= rng.uniform(0.8, 1.2)
            else:
                node_to_mutate.value += rng.uniform(-2.0, 2.0)
            node_to_mutate.value = max(min(node_to_mutate.value, 10000.0), -10000.0)
            if abs(node_to_mutate.value) < 1e-7:
                node_to_mutate.value = 0.0
        else:
            # fallback
            node_to_mutate = generate_random_tree(max_depth)
    elif mut_type == 'OperatorChange':
        if node_to_mutate.type == NodeType.OPERATOR:
            ops = ['+', '-', '*', '/', '^']
            possible_ops = [op for op in ops if op != node_to_mutate.op]
            if possible_ops:
                node_to_mutate.op = rng.choice(possible_ops)
                if node_to_mutate.op == '^':
                    if not node_to_mutate.right or node_to_mutate.right.type != NodeType.CONSTANT:
                        right = Node(NodeType.CONSTANT)
                        right.value = rng.randint(2, 4)
                        node_to_mutate.right = right
                    else:
                        node_to_mutate.right.value = round(max(min(node_to_mutate.right.value, 4.0), 2.0))
        else:
            node_to_mutate = generate_random_tree(max_depth)
    elif mut_type == 'SubtreeReplace':
        node_to_mutate = generate_random_tree(max_depth)
    elif mut_type == 'NodeInsertion':
        new_op_node = Node(NodeType.OPERATOR)
        insert_ops = ['+', '-']
        new_op_node.op = rng.choice(insert_ops)
        new_op_node.left = node_to_mutate
        if rng.random() < 0.5:
            right_const = Node(NodeType.CONSTANT)
            right_const.value = rng.randint(1, 3)
            new_op_node.right = right_const
        else:
            new_op_node.right = Node(NodeType.VARIABLE)
        node_to_mutate = new_op_node
    else:
        node_to_mutate = generate_random_tree(max_depth)
    new_tree = simplify_tree(new_tree)
    return new_tree

def mutate_tree_batch(trees, mutation_rate, max_depth):
    from copy import deepcopy
    from random import random
    mutated = []
    for t in trees:
        if random() < mutation_rate:
            mutated.append(mutate_tree(deepcopy(t), mutation_rate, max_depth))
        else:
            mutated.append(deepcopy(t))
    return mutated

def crossover_trees(tree1: Node, tree2: Node):
    nodes1: List[Node] = []
    nodes2: List[Node] = []
    collect_node_ptrs(tree1, nodes1)
    collect_node_ptrs(tree2, nodes2)
    if not nodes1 or not nodes2:
        return
    rng = get_rng()
    point1 = rng.choice(nodes1)
    point2 = rng.choice(nodes2)
    # Swap the subtrees (in Python, swap attributes)
    point1.type, point2.type = point2.type, point1.type
    point1.value, point2.value = point2.value, point1.value
    point1.op, point2.op = point2.op, point1.op
    point1.left, point2.left = point2.left, point1.left
    point1.right, point2.right = point2.right, point1.right
    tree1 = simplify_tree(tree1)
    tree2 = simplify_tree(tree2)
