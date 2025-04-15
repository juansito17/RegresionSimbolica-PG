# genetic_algorithm.py
from typing import List, Optional
from expression_tree import Node, tree_to_string, clone_tree, evaluate_tree, tree_size
from genetic_operators import Individual, create_initial_population, tournament_selection, mutate_tree, crossover_trees, generate_random_tree
from advanced_features import EvolutionParameters, PatternMemory, ParetoOptimizer, DomainConstraints, try_local_improvement, detect_target_pattern, generate_pattern_based_tree
from fitness import evaluate_fitness, batch_fitness_gpu
from globals import *
import math
import concurrent.futures
from fitness import evaluate_fitness_with_cache

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
        self.global_subtree_library = GlobalSubtreeLibrary(max_size=150)
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
        # Evaluación de fitness en GPU para toda la población
        trees = [ind.tree for ind in island.population]
        fitnesses = [evaluate_fitness_with_cache(tree, self.targets, self.x_values) for tree in trees]
        for ind, fit in zip(island.population, fitnesses):
            ind.fitness = fit
            ind.fitness_valid = True
        # Aplicar niching/fitness sharing tras calcular fitness
        apply_fitness_sharing(island.population)

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
        # Usar parámetros evolutivos adaptativos de la isla
        params = island.params
        mutation_rate = params.mutation_rate
        elite_percentage = params.elite_percentage
        tournament_size = params.tournament_size
        crossover_rate = params.crossover_rate
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
        # Multiobjetivo avanzado: actualizar Pareto con precisión y complejidad
        for ind in island.population:
            ind.complexity = tree_size(ind.tree) if ind.tree else 0
        island.pareto_optimizer.update_multiobjective(island.population)
        for ind in island.population:
            if ind.fitness_valid and ind.fitness < 10.0:
                island.pattern_memory.record_success(ind.tree, ind.fitness)
        next_generation = []
        elite_count = max(1, int(len(island.population) * elite_percentage))
        next_generation.extend(island.population[:elite_count])
        if island.stagnation_counter > STAGNATION_LIMIT // 2:
            for _ in range(int(len(island.population) * 0.1)):
                next_generation.append(Individual(generate_random_tree(MAX_TREE_DEPTH_INITIAL)))
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
        # Inyección de sub-árboles globales en la nueva generación
        subtree_injection_count = max(1, int(len(island.population) * 0.08))
        for _ in range(subtree_injection_count):
            subtree = self.global_subtree_library.get_random_subtree()
            if subtree:
                from expression_tree import insert_subtree_random
                base_tree = generate_random_tree(MAX_TREE_DEPTH_INITIAL)
                new_tree = insert_subtree_random(base_tree, subtree)
                next_generation.append(Individual(new_tree))
        remaining = len(island.population) - len(next_generation)
        stats = analyze_population_statistics(island.population)
        for _ in range(remaining):
            parent1 = tournament_selection(island.population, tournament_size)
            child = Individual(None)
            if random.random() < crossover_rate:
                parent2 = tournament_selection(island.population, tournament_size)
                p1_clone = clone_tree(parent1.tree)
                p2_clone = clone_tree(parent2.tree)
                crossover_trees(p1_clone, p2_clone)
                child.tree = p1_clone
            else:
                child.tree = clone_tree(parent1.tree)
            # Mutación guiada por análisis estadístico
            child.tree = guided_mutation(child.tree, stats)
            child.fitness_valid = False
            next_generation.append(child)
        island.population = next_generation
        # Mutación adaptativa de parámetros evolutivos según estancamiento
        if current_generation > 0 and (current_generation % 50 == 0 or island.stagnation_counter > STAGNATION_LIMIT // 2):
            # Si la isla está muy estancada, mutar más agresivamente
            island.params.mutate(stagnation_counter=island.stagnation_counter)

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
                memetic_local_search(island, self.targets, self.x_values, top_k=3, attempts=20)
                self.evolve_island(island, gen)
            for island in self.islands:
                assign_species(island.population)
            adaptive_migration(self.islands)
            if (gen + 1) % 10 == 0:
                transfer_successful_subtrees(self.islands, top_n=3)
            if (gen + 1) % 10 == 0:
                self.global_subtree_library.update_from_islands(self.islands, top_n=5)
            if (gen + 1) % MIGRATION_INTERVAL == 0 and self.num_islands > 1:
                self.migrate()
            if self.overall_best_fitness < 1e-6:
                print(f"\nSolution found at generation {gen+1}!")
                break
            if (gen + 1) % 10 == 0:
                for idx, island in enumerate(self.islands):
                    stats = analyze_population_statistics(island.population)
                    print(f"[Gen {gen+1}][Isla {idx}] Stats: {stats}")
            if (gen + 1) % 100 == 0 or gen == 0:
                print(f"Gen {gen+1}: Best fitness so far: {self.overall_best_fitness:.6f}")
        return self.overall_best_tree

    def get_ensemble(self, k=5):
        """Devuelve los k mejores árboles del frente de Pareto multiobjetivo para ensemble."""
        all_best = []
        for island in self.islands:
            if hasattr(island.pareto_optimizer, 'pareto_front'):
                all_best.extend([sol.tree for sol in getattr(island.pareto_optimizer, 'pareto_front')[:k]])
        # Eliminar duplicados
        unique = []
        seen = set()
        for t in all_best:
            s = tree_to_string(t)
            if s not in seen:
                unique.append(t)
                seen.add(s)
        return unique[:k]

def apply_fitness_sharing(island_population, alpha=1.0, sigma_share=0.4):
    """Ajusta el fitness de la población usando sharing/crowding para niching."""
    from expression_tree import tree_structural_distance
    n = len(island_population)
    shared_fitness = [ind.fitness for ind in island_population]
    for i in range(n):
        sh_sum = 0.0
        for j in range(n):
            if i == j:
                continue
            d = tree_structural_distance(island_population[i].tree, island_population[j].tree)
            if d < sigma_share:
                sh_sum += 1 - (d / sigma_share) ** alpha
        # Evitar división por cero
        if sh_sum > 0:
            shared_fitness[i] = island_population[i].fitness * (1 + sh_sum)
        else:
            shared_fitness[i] = island_population[i].fitness
    # Asigna el fitness compartido
    for ind, fit in zip(island_population, shared_fitness):
        ind.fitness = fit

def assign_species(population, sigma_spec=0.5):
    """Agrupa la población en especies según distancia estructural."""
    from expression_tree import tree_structural_distance
    species = []  # Lista de listas de índices
    representatives = []
    for i, ind in enumerate(population):
        assigned = False
        for s_idx, rep in enumerate(representatives):
            d = tree_structural_distance(ind.tree, rep.tree)
            if d < sigma_spec:
                species[s_idx].append(i)
                assigned = True
                break
        if not assigned:
            species.append([i])
            representatives.append(ind)
    return species

def adaptive_migration(islands, min_diversity=0.2):
    """Realiza migración adaptativa entre islas según diversidad estructural."""
    from expression_tree import tree_structural_distance
    for i, island in enumerate(islands):
        pop = island.population
        # Calcular diversidad promedio
        dists = []
        for a in range(len(pop)):
            for b in range(a+1, len(pop)):
                dists.append(tree_structural_distance(pop[a].tree, pop[b].tree))
        avg_div = sum(dists)/len(dists) if dists else 0.0
        # Si la diversidad es baja, forzar migración extra
        if avg_div < min_diversity:
            # Tomar migrantes de la isla anterior
            src_idx = (i-1) % len(islands)
            migrants = [clone_tree(ind.tree) for ind in islands[src_idx].population[:2]]
            for j in range(min(len(migrants), len(island.population))):
                island.population[-(j+1)] = Individual(migrants[j])

def memetic_local_search(island, targets, x_values, top_k=3, attempts=20):
    """Aplica búsqueda local intensiva a los top_k mejores individuos de la isla."""
    from advanced_features import try_local_improvement
    island.population.sort()
    improved = False
    for i in range(min(top_k, len(island.population))):
        ind = island.population[i]
        best_tree, best_fit = try_local_improvement(ind.tree, ind.fitness, targets, x_values, attempts)
        if best_fit < ind.fitness:
            ind.tree = best_tree
            ind.fitness = best_fit
            ind.fitness_valid = True
            improved = True
    return improved

def dominates_multi(a, b):
    """Devuelve True si a domina a b en precisión y complejidad."""
    return (a.fitness <= b.fitness and a.complexity <= b.complexity and
            (a.fitness < b.fitness or a.complexity < b.complexity))

class ParetoSolutionMulti:
    def __init__(self, tree, fitness, complexity):
        self.tree = tree
        self.fitness = fitness
        self.complexity = complexity
        self.dominated = False
    def dominates(self, other):
        return dominates_multi(self, other)

def update_multiobjective(self, population):
    """Actualiza el frente de Pareto considerando precisión y complejidad."""
    new_front = []
    for ind in population:
        if not ind.fitness_valid:
            continue
        sol = ParetoSolutionMulti(ind.tree, ind.fitness, getattr(ind, 'complexity', 0))
        dominated = False
        for other in new_front:
            if other.dominates(sol):
                dominated = True
                break
        if not dominated:
            new_front.append(sol)
    self.pareto_front = sorted(new_front, key=lambda s: (s.fitness, s.complexity))[:self.max_front_size]

# Monkey patch al ParetoOptimizer para multiobjetivo
ParetoOptimizer.update_multiobjective = update_multiobjective

def transfer_successful_subtrees(islands, top_n=3):
    """Transfiere sub-árboles exitosos entre islas para transfer learning."""
    from expression_tree import clone_tree
    # Recopilar los mejores sub-árboles de cada isla
    subtrees = []
    for island in islands:
        island.population.sort()
        for ind in island.population[:top_n]:
            if ind.tree:
                subtrees.append(clone_tree(ind.tree))
    # Insertar sub-árboles en otras islas (excepto la propia)
    for i, island in enumerate(islands):
        for subtree in subtrees:
            # Evitar insertar exactamente el mismo árbol
            if not any(tree_to_string(subtree) == tree_to_string(ind.tree) for ind in island.population[:top_n]):
                # Reemplazar un individuo aleatorio de la isla
                import random
                idx = random.randint(0, len(island.population)-1)
                island.population[idx] = Individual(clone_tree(subtree))

def ensemble_predict(trees, x):
    """Realiza la predicción promedio de un conjunto de árboles (ensemble)."""
    from expression_tree import evaluate_tree
    preds = [evaluate_tree(t, x) for t in trees if t is not None]
    if not preds:
        return float('nan')
    return sum(preds) / len(preds)

def batch_ensemble_predict(ensembles, xs):
    """Evalúa en batch un conjunto de ensembles sobre un array de xs (acelerado en GPU si es posible)."""
    import numpy as np
    from expression_tree import evaluate_tree
    preds = np.zeros((len(ensembles), len(xs)))
    for i, trees in enumerate(ensembles):
        for j, x in enumerate(xs):
            preds[i, j] = ensemble_predict(trees, x)
    return preds

def analyze_population_statistics(population):
    """Realiza un análisis estadístico de la población: fitness, diversidad, tamaño, etc."""
    import numpy as np
    from expression_tree import tree_structural_distance, tree_size
    fitnesses = [ind.fitness for ind in population if ind.fitness_valid]
    sizes = [tree_size(ind.tree) for ind in population if ind.tree]
    # Diversidad estructural promedio
    dists = []
    for i in range(len(population)):
        for j in range(i+1, len(population)):
            dists.append(tree_structural_distance(population[i].tree, population[j].tree))
    stats = {
        'fitness_mean': np.mean(fitnesses) if fitnesses else float('nan'),
        'fitness_std': np.std(fitnesses) if fitnesses else float('nan'),
        'fitness_min': np.min(fitnesses) if fitnesses else float('nan'),
        'fitness_max': np.max(fitnesses) if fitnesses else float('nan'),
        'size_mean': np.mean(sizes) if sizes else float('nan'),
        'size_std': np.std(sizes) if sizes else float('nan'),
        'diversity_mean': np.mean(dists) if dists else float('nan'),
        'diversity_std': np.std(dists) if dists else float('nan'),
        'population_size': len(population)
    }
    return stats

def guided_mutation(tree, stats):
    """Realiza una mutación guiada por análisis estadístico: si la diversidad es baja, fuerza mutaciones más grandes."""
    from genetic_operators import mutate_tree
    diversity = stats.get('diversity_mean', 0.0)
    # Si la diversidad es baja, aumenta la tasa de mutación y profundidad
    if diversity < 0.15:
        return mutate_tree(tree, mutation_rate=0.8, max_depth=7)
    # Si la diversidad es alta, mutación estándar
    return mutate_tree(tree, mutation_rate=0.4, max_depth=5)

class GlobalSubtreeLibrary:
    """Biblioteca global de sub-árboles exitosos para todas las islas."""
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.subtrees = []  # Lista de (subtree, score, string_repr)
        self.strings = set()
    def add_subtree(self, subtree, score):
        from expression_tree import tree_to_string, clone_tree
        s = tree_to_string(subtree)
        if s in self.strings:
            return
        self.subtrees.append((clone_tree(subtree), score, s))
        self.strings.add(s)
        self.subtrees.sort(key=lambda x: x[1])  # Menor score es mejor
        if len(self.subtrees) > self.max_size:
            self.subtrees = self.subtrees[:self.max_size]
            self.strings = set(x[2] for x in self.subtrees)
    def get_random_subtree(self):
        import random
        if not self.subtrees:
            return None
        return random.choice(self.subtrees)[0]
    def update_from_islands(self, islands, top_n=5):
        for island in islands:
            island.population.sort()
            for ind in island.population[:top_n]:
                if ind.tree:
                    # Extraer sub-árboles de tamaño intermedio
                    for subtree in extract_subtrees(ind.tree, min_size=2, max_size=7):
                        self.add_subtree(subtree, ind.fitness)

def extract_subtrees(tree, min_size=2, max_size=7):
    """Extrae todos los sub-árboles de un árbol cuyo tamaño esté en el rango dado."""
    from expression_tree import tree_size
    result = []
    def visit(node):
        if node is None:
            return
        size = tree_size(node)
        if min_size <= size <= max_size:
            result.append(node)
        if hasattr(node, 'left'):
            visit(node.left)
        if hasattr(node, 'right'):
            visit(node.right)
    visit(tree)
    return result