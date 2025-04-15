# advanced_features.py
from typing import List, Tuple, Optional
from expression_tree import Node, tree_to_string, clone_tree
from fitness import evaluate_fitness
from globals import INF, MUTATION_RATE, ELITE_PERCENTAGE, TOURNAMENT_SIZE, CROSSOVER_RATE
from genetic_operators import generate_random_tree
import random

# --- EvolutionParameters ---
class EvolutionParameters:
    def __init__(self):
        self.mutation_rate = MUTATION_RATE
        self.elite_percentage = ELITE_PERCENTAGE
        self.tournament_size = TOURNAMENT_SIZE
        self.crossover_rate = CROSSOVER_RATE
        self.history = []  # Guarda historial de parámetros y estancamiento
    @staticmethod
    def create_default():
        return EvolutionParameters()
    def mutate(self, stagnation_counter=0):
        # Mutación adaptativa: más agresiva si hay mucho estancamiento
        base = 1.0 + min(stagnation_counter, 20) / 10.0
        self.mutation_rate = min(max(self.mutation_rate + random.uniform(-0.05, 0.05) * base, 0.01), 0.5)
        self.elite_percentage = min(max(self.elite_percentage + random.uniform(-0.02, 0.02) * base, 0.01), 0.5)
        self.tournament_size = max(2, self.tournament_size + int(random.choice([-1, 0, 1]) * base))
        self.crossover_rate = min(max(self.crossover_rate + random.uniform(-0.05, 0.05) * base, 0.5), 1.0)
        # Guarda el historial para análisis
        self.history.append({
            'mutation_rate': self.mutation_rate,
            'elite_percentage': self.elite_percentage,
            'tournament_size': self.tournament_size,
            'crossover_rate': self.crossover_rate,
            'stagnation_counter': stagnation_counter
        })

# --- PatternMemory (simplified) ---
class PatternMemory:
    def __init__(self):
        self.patterns = {}
    def record_success(self, tree: Node, fitness: float):
        s = tree_to_string(tree)
        if s not in self.patterns or fitness < self.patterns[s]:
            self.patterns[s] = fitness
    def suggest_pattern_based_tree(self, max_depth: int) -> Optional[Node]:
        if not self.patterns:
            return None
        # Return the tree with the best fitness pattern
        best_pattern = min(self.patterns, key=lambda k: self.patterns[k])
        # For simplicity, just generate a random tree (pattern parsing omitted)
        return generate_random_tree(max_depth)

# --- ParetoOptimizer (minimal) ---
class ParetoSolution:
    def __init__(self, tree: Node, accuracy: float, complexity: float):
        self.tree = tree
        self.accuracy = accuracy
        self.complexity = complexity
        self.dominated = False
    def dominates(self, other) -> bool:
        return (self.accuracy <= other.accuracy and self.complexity <= other.complexity and
                (self.accuracy < other.accuracy or self.complexity < other.complexity))

class ParetoOptimizer:
    def __init__(self):
        self.pareto_front: List[ParetoSolution] = []
        self.max_front_size = 50
    def update(self, population, targets, x_values):
        # Only keep non-dominated solutions
        new_front = []
        for ind in population:
            if not ind.fitness_valid:
                continue
            acc = ind.fitness
            comp = 1.0  # Placeholder for complexity
            sol = ParetoSolution(ind.tree, acc, comp)
            dominated = False
            for other in new_front:
                if other.dominates(sol):
                    dominated = True
                    break
            if not dominated:
                new_front.append(sol)
        self.pareto_front = sorted(new_front, key=lambda s: s.accuracy)[:self.max_front_size]
    def get_pareto_solutions(self):
        return [s.tree for s in self.pareto_front]

# --- DomainConstraints (minimal) ---
class DomainConstraints:
    @staticmethod
    def is_valid(tree: Node) -> bool:
        return tree is not None
    @staticmethod
    def fix_or_simplify(tree: Node) -> Node:
        return tree  # No-op for now

# --- Local improvement (minimal) ---
def try_local_improvement(tree: Node, current_fitness: float, targets: List[float], x_values: List[float], attempts: int = 10) -> Tuple[Node, float]:
    best_tree = clone_tree(tree)
    best_fitness = current_fitness
    for _ in range(attempts):
        mutated = generate_random_tree(5)
        fit = evaluate_fitness(mutated, targets, x_values)
        if fit < best_fitness:
            best_tree = mutated
            best_fitness = fit
    return best_tree, best_fitness

# --- Pattern detection (minimal) ---
def detect_target_pattern(targets: List[float]) -> Tuple[str, float]:
    # Dummy: detect if all differences are equal (arithmetic progression)
    if len(targets) < 2:
        return "none", 0.0
    diffs = [targets[i+1] - targets[i] for i in range(len(targets)-1)]
    if all(abs(d - diffs[0]) < 1e-6 for d in diffs):
        return "arithmetic", diffs[0]
    return "none", 0.0

def generate_pattern_based_tree(pattern_type: str, pattern_value: float) -> Optional[Node]:
    # Inyecta árboles polinómicos, logarítmicos y exponenciales como semillas
    from expression_tree import Node, NodeType
    import math
    if pattern_type == "arithmetic":
        node = Node(NodeType.OPERATOR)
        node.op = '*'
        node.left = Node(NodeType.VARIABLE)
        node.right = Node(NodeType.CONSTANT)
        node.right.value = pattern_value
        return node
    elif pattern_type == "polynomial":
        # x^2 + bx + c
        node = Node(NodeType.OPERATOR)
        node.op = '+'
        left = Node(NodeType.OPERATOR)
        left.op = '^'
        left.left = Node(NodeType.VARIABLE)
        left.right = Node(NodeType.CONSTANT)
        left.right.value = 2
        right = Node(NodeType.OPERATOR)
        right.op = '+'
        right.left = Node(NodeType.OPERATOR)
        right.left.op = '*'
        right.left.left = Node(NodeType.CONSTANT)
        right.left.left.value = 2
        right.left.right = Node(NodeType.VARIABLE)
        right.right = Node(NodeType.CONSTANT)
        right.right.value = 1
        node.left = left
        node.right = right
        return node
    elif pattern_type == "log":
        node = Node(NodeType.OPERATOR)
        node.op = 'l'
        node.left = Node(NodeType.VARIABLE)
        return node
    elif pattern_type == "exp":
        node = Node(NodeType.OPERATOR)
        node.op = '^'
        node.left = Node(NodeType.CONSTANT)
        node.left.value = math.e
        node.right = Node(NodeType.VARIABLE)
        return node
    return None
