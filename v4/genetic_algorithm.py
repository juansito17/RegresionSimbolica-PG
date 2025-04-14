# genetic_algorithm.py
from typing import List, Optional
from expression_tree import Node, tree_to_string, clone_tree, evaluate_tree, tree_size
from genetic_operators import Individual, create_initial_population, tournament_selection, mutate_tree, crossover_trees, generate_random_tree
from advanced_features import EvolutionParameters, PatternMemory, ParetoOptimizer, DomainConstraints, try_local_improvement, detect_target_pattern, generate_pattern_based_tree
from fitness import evaluate_fitness
from globals import *
import math
import concurrent.futures

class Island:
    def __init__(self, island_id: int, pop_size: int):
        self.id = island_id
        self.population: List[Individual] = create_initial_population(pop_size)
        self.params = EvolutionParameters.create_default()
        self.pattern_memory = PatternMemory()
        self.pareto_optimizer = ParetoOptimizer()
        self.stagnation_counter = 0
        self.best_fitness = INF
        self.fitness_history: List[float] = []

class GeneticAlgorithm:
    def __init__(self, targets: List[float], x_values: List[float], total_pop: int, gens: int, n_islands: int = NUM_ISLANDS):
        self.targets = targets
        self.x_values = x_values
        self.total_population_size = total_pop
        self.generations = gens
        self.num_islands = n_islands if n_islands > 0 else 1
        self.pop_per_island = max(10, self.total_population_size // self.num_islands)
        if self.pop_per_island < 10:
            self.pop_per_island = 10
            self.num_islands = max(1, self.total_population_size // self.pop_per_island)
        self.islands: List[Island] = [Island(i, self.pop_per_island) for i in range(self.num_islands)]
        self.overall_best_tree: Optional[Node] = None
        self.overall_best_fitness = INF
        # Pattern-based seed injection
        pattern_type, pattern_value = detect_target_pattern(self.targets)
        if pattern_type != "none":
            pattern_tree = generate_pattern_based_tree(pattern_type, pattern_value)
            if pattern_tree:
                for island in self.islands:
                    if island.population:
                        island.population[-1] = Individual(pattern_tree)
                    else:
                        island.population.append(Individual(pattern_tree))
        # Initial fitness evaluation
        for island in self.islands:
            self.evaluate_population(island)
            self.update_overall_best(island)

    def evaluate_population(self, island: Island):
        # Parallel fitness evaluation
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for ind in island.population:
                if not ind.fitness_valid and ind.tree:
                    futures.append(executor.submit(evaluate_fitness, ind.tree, self.targets, self.x_values))
                else:
                    futures.append(None)
            for idx, ind in enumerate(island.population):
                if not ind.fitness_valid and ind.tree:
                    result = futures[idx].result()
                    ind.fitness = result
                    ind.fitness_valid = True
                elif not ind.tree:
                    ind.fitness = INF
                    ind.fitness_valid = True

    def update_overall_best(self, island: Island):
        for ind in island.population:
            if ind.fitness_valid and ind.fitness < self.overall_best_fitness:
                self.overall_best_fitness = ind.fitness
                self.overall_best_tree = clone_tree(ind.tree)
                print(f"\n========================================")
                print(f"Nueva mejor fórmula encontrada (Isla {island.id})")
                print(f"Fitness: {self.overall_best_fitness:.8f}")
                print(f"Tamaño: {tree_size(self.overall_best_tree)}")
                print(f"Fórmula: {tree_to_string(self.overall_best_tree)}")
                print("Predicciones vs Objetivos:")
                for x, y in zip(self.x_values, self.targets):
                    val = evaluate_tree(self.overall_best_tree, x)
                    diff = abs(val - y)
                    print(f"  x={x}: Pred={val:.4f}, Target={y}, Diff={diff:.4f}")
                print(f"========================================\n")

    def evolve_island(self, island: Island, current_generation: int):
        island.population.sort()
        current_best_fitness = island.population[0].fitness
        island.fitness_history.append(current_best_fitness)
        if current_best_fitness < island.best_fitness - 1e-9:
            island.best_fitness = current_best_fitness
            island.stagnation_counter = 0
            best_tree, best_fit = try_local_improvement(island.population[0].tree, island.population[0].fitness, self.targets, self.x_values, 15)
            if best_fit < island.population[0].fitness:
                island.population[0].tree = best_tree
                island.population[0].fitness = best_fit
                island.best_fitness = best_fit
        else:
            island.stagnation_counter += 1
        self.update_overall_best(island)
        island.pareto_optimizer.update(island.population, self.targets, self.x_values)
        for ind in island.population:
            if ind.fitness_valid and ind.fitness < 10.0:
                island.pattern_memory.record_success(ind.tree, ind.fitness)
        next_generation = []
        elite_count = max(1, int(len(island.population) * island.params.elite_percentage))
        next_generation.extend(island.population[:elite_count])
        if island.stagnation_counter > STAGNATION_LIMIT // 2:
            for _ in range(int(len(island.population) * 0.1)):
                next_generation.append(Individual(generate_random_tree(MAX_TREE_DEPTH_INITIAL)))
        # Inyección de patrones más frecuente y mayor cantidad
        if current_generation % 5 == 0:
            for _ in range(int(len(island.population) * 0.10)):
                pattern_tree = island.pattern_memory.suggest_pattern_based_tree(MAX_TREE_DEPTH_INITIAL)
                if pattern_tree:
                    next_generation.append(Individual(pattern_tree))
                else:
                    next_generation.append(Individual(generate_random_tree(MAX_TREE_DEPTH_INITIAL)))
        if current_generation % 10 == 0:
            for _ in range(int(len(island.population) * 0.05)):
                pattern_tree = island.pattern_memory.suggest_pattern_based_tree(MAX_TREE_DEPTH_INITIAL)
                if pattern_tree:
                    next_generation.append(Individual(pattern_tree))
                else:
                    next_generation.append(Individual(generate_random_tree(MAX_TREE_DEPTH_INITIAL)))
        remaining = len(island.population) - len(next_generation)
        for _ in range(remaining):
            parent1 = tournament_selection(island.population, island.params.tournament_size)
            child = Individual(None)
            if random.random() < island.params.crossover_rate:
                parent2 = tournament_selection(island.population, island.params.tournament_size)
                p1_clone = clone_tree(parent1.tree)
                p2_clone = clone_tree(parent2.tree)
                crossover_trees(p1_clone, p2_clone)
                child.tree = p1_clone
            else:
                child.tree = clone_tree(parent1.tree)
            child.tree = mutate_tree(child.tree, island.params.mutation_rate, MAX_TREE_DEPTH_MUTATION)
            child.fitness_valid = False
            next_generation.append(child)
        island.population = next_generation
        if current_generation > 0 and current_generation % 50 == 0:
            island.params.mutate()

    def migrate(self):
        if self.num_islands <= 1:
            return
        num_migrants = min(MIGRATION_SIZE, self.pop_per_island // 5)
        if num_migrants == 0:
            return
        outgoing = []
        for island in self.islands:
            island.population.sort()
            outgoing.append([clone_tree(ind.tree) for ind in island.population[:num_migrants]])
        for dest_idx, island in enumerate(self.islands):
            source_idx = (dest_idx + self.num_islands - 1) % self.num_islands
            migrants = outgoing[source_idx]
            island.population.sort(key=lambda ind: ind.fitness if ind.fitness_valid else INF, reverse=True)
            for i in range(min(len(migrants), len(island.population))):
                island.population[i] = Individual(migrants[i])

    def run(self) -> Optional[Node]:
        for gen in range(self.generations):
            for island in self.islands:
                self.evaluate_population(island)
                self.evolve_island(island, gen)
            if (gen + 1) % MIGRATION_INTERVAL == 0 and self.num_islands > 1:
                self.migrate()
            if self.overall_best_fitness < 1e-6:
                print(f"\nSolution found at generation {gen+1}!")
                break
            # Mostrar progreso cada 10 generaciones
            if (gen + 1) % 100 == 0 or gen == 0:
                print(f"Gen {gen+1}: Best fitness so far: {self.overall_best_fitness:.6f}")
        return self.overall_best_tree
