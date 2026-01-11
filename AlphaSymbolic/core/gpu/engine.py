
import torch
import numpy as np
import time
from typing import List, Tuple, Optional
from core.grammar import ExpressionTree

from .config import GpuGlobals
from .sniper import Sniper
from .pareto import ParetoOptimizer
from .pattern_memory import PatternMemory

# New Modules
from .grammar import GPUGrammar, PAD_ID
from .evaluation import GPUEvaluator
from .operators import GPUOperators
from .optimization import GPUOptimizer
from .simplification import GPUSimplifier

class TensorGeneticEngine:
    def __init__(self, device=None, pop_size=None, max_len=30, num_variables=1, max_constants=5, n_islands=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Defaults from Globals
        if pop_size is None: pop_size = GpuGlobals.POP_SIZE
        if n_islands is None: n_islands = GpuGlobals.NUM_ISLANDS
        
        self.n_islands = n_islands
        if pop_size % n_islands != 0:
            pop_size = (pop_size // n_islands) * n_islands
            
        self.pop_size = pop_size
        self.island_size = pop_size // n_islands
        self.max_len = max_len
        self.num_variables = num_variables
        self.max_constants = max_constants

        # --- Sub-components ---
        self.grammar = GPUGrammar(num_variables)
        
        self.evaluator = GPUEvaluator(self.grammar, self.device)
        self.operators = GPUOperators(self.grammar, self.device, self.pop_size, self.max_len, self.num_variables)
        self.optimizer = GPUOptimizer(self.evaluator, self.operators, self.device)
        self.simplifier = GPUSimplifier(self.grammar, self.device, self.max_constants)
        
        # --- Advanced Features ---
        self.sniper = Sniper(self.device)
        self.pareto = ParetoOptimizer(self.device, GpuGlobals.PARETO_MAX_FRONT_SIZE)
        self.pattern_memory = PatternMemory(
            self.device, 
            max_patterns=100,
            fitness_threshold=GpuGlobals.PATTERN_RECORD_FITNESS_THRESHOLD,
            min_uses=GpuGlobals.PATTERN_MEM_MIN_USES
        )

    # --- Wrappers for backward compatibility and convenience ---

    def evaluate_batch(self, population: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor, constants: torch.Tensor = None) -> torch.Tensor:
        return self.evaluator.evaluate_batch(population, x, y_target, constants)

    def optimize_constants(self, population, constants, x, y, steps=10, lr=0.1):
        return self.optimizer.optimize_constants(population, constants, x, y, steps, lr)
        
    def simplify_population(self, population, constants, top_k=None):
        return self.simplifier.simplify_population(population, constants, top_k)

    def infix_to_rpn(self, formulas: List[str]) -> torch.Tensor:
        return self.operators._infix_list_to_rpn(formulas)
        
    def rpn_to_infix(self, rpn_tensor: torch.Tensor, constants: torch.Tensor = None) -> str:
        return self.simplifier._rpn_to_infix_str(rpn_tensor, constants)

    def load_population_from_strings(self, formulas: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Logic to load specifically valid RPN + constants (extract constants from string)
        # The simplifier's logic extracts constants. 
        # But we need a loader.
        # Engine's original `infix_to_rpn_tensor` did this.
        # Operators `_infix_list_to_rpn` does NOT extract constants (it sets 1.0 or C).
        # We need `infix_to_rpn_tensor` equivalent.
        # Reuse simplifier logic or implement here using ExpressionTree + Extraction?
        # Let's put a helper here or in Operators.
        
        rpn_list = []
        const_list = []
        
        # We can reuse GPUSimplifier.simplify_expression logic basically just to parse?
        # Or reimplement extraction.
        # Since this is "load from strings", let's replicate the extraction logic briefly.
        for f in formulas:
            # We use simplifier's internal parser logic? 
            # Or better: `infix_to_rpn_tensor` from original engine.
            # I'll implement it here or adding it to Operators as `parse_infix_with_constants`.
            
            # For now, minimal implementation inside loop:
            r, c, success = self.simplifier.simplify_expression(torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)) 
            # verify call signature? simplify needs RPN input to simplify.
            # We want string -> RPN.
            # Let's use `ExpressionTree` directly here.
            
            try:
                tree = ExpressionTree.from_infix(f)
                if not tree.is_valid:
                     rpn_list.append(torch.zeros(self.max_len, dtype=torch.long, device=self.device))
                     const_list.append(torch.zeros(self.max_constants, dtype=torch.float64, device=self.device))
                     continue

                rpn_tokens = []
                def traverse(node):
                    if not node: return
                    for child in node.children: traverse(child)
                    rpn_tokens.append(node.value)
                traverse(tree.root)
                
                clean_tokens = []
                const_values = []
                
                for t in rpn_tokens:
                     if t in self.grammar.terminals and t not in ['C', '1', '2', '3', '5'] and not t.startswith('x'):
                         clean_tokens.append(t)
                     elif (t.replace('.','',1).isdigit() or (t.startswith('-') and t[1:].replace('.','',1).isdigit())):
                         if t in ['1', '2', '3', '5']:
                             clean_tokens.append(t)
                         else:
                             clean_tokens.append('C')
                             const_values.append(float(t))
                     else:
                         clean_tokens.append(t)
                
                ids = [self.grammar.token_to_id.get(t, PAD_ID) for t in clean_tokens]
                if len(ids) > self.max_len: ids = ids[:self.max_len]
                else: ids += [PAD_ID] * (self.max_len - len(ids))
                
                if len(const_values) > self.max_constants: const_values = const_values[:self.max_constants]
                else: const_values += [0.0] * (self.max_constants - len(const_values))
                
                rpn_list.append(torch.tensor(ids, dtype=torch.long, device=self.device))
                const_list.append(torch.tensor(const_values, dtype=torch.float64, device=self.device))
            except:
                pass
                
        if not rpn_list: return None, None
        return torch.stack(rpn_list), torch.stack(const_list)

    # --- Main Logic ---

    def initialize_population(self) -> torch.Tensor:
        return self.operators.generate_random_population(self.pop_size)

    def detect_patterns(self, targets: List[float]) -> List[str]:
        if len(targets) < 3: return []
        seeds = []
        diffs = np.diff(targets)
        if np.allclose(diffs, diffs[0], atol=1e-5):
            seeds.append("(C + (C * x0))") 
        if not np.any(np.abs(targets) < 1e-9):
            ratios = targets[1:] / targets[:-1]
            if np.allclose(ratios, ratios[0], atol=1e-5):
                seeds.append("(C * (C ^ x0))")
        if np.allclose(targets, targets[0], atol=1e-5):
             seeds.append("C")
        return seeds

    def migrate_islands(self, population, constants, fitness):
        # Implementation of migration
        # We can move this to Operators or keep here. 
        # Since it uses simple index manipulation, Operators is a good place.
        # But I didn't verify if I moved it to Operators in previous step. 
        # Let's check `operators.py` (I wrote it, but didn't include `migrate_islands`).
        # Oops. I should implement it here or add to operators.
        # I'll implement it here for now to save a file edit, or better, add it to Operators later.
        if self.n_islands <= 1: return population, constants
        
        pop_out = population.clone()
        const_out = constants.clone()
        island_size = self.island_size
        mig_size = min(GpuGlobals.MIGRATION_SIZE, island_size // 2)
        
        for island in range(self.n_islands):
            src_start = island * island_size
            src_end = src_start + island_size
            dst_island = (island + 1) % self.n_islands
            dst_start = dst_island * island_size
            dst_end = dst_start + island_size
            
            src_fitness = fitness[src_start:src_end]
            _, best_idx_local = torch.topk(src_fitness, mig_size, largest=False)
            best_idx_global = best_idx_local + src_start
            
            dst_fitness = fitness[dst_start:dst_end]
            _, worst_idx_local = torch.topk(dst_fitness, mig_size, largest=True)
            worst_idx_global = worst_idx_local + dst_start
            
            pop_out[worst_idx_global] = population[best_idx_global]
            const_out[worst_idx_global] = constants[best_idx_global]
            
        return pop_out, const_out

    def run(self, x_values, y_values, seeds: List[str] = None, timeout_sec: int = 10, callback=None):
        start_time = time.time()
        
        # 1. Data Setup
        if not isinstance(x_values, torch.Tensor):
            x_t = torch.tensor(x_values, dtype=torch.float64, device=self.device)
        else:
            x_t = x_values.to(self.device).to(torch.float64)
            
        if not isinstance(y_values, torch.Tensor):
            y_t = torch.tensor(y_values, dtype=torch.float64, device=self.device)
            # Log transform if needed
            if GpuGlobals.USE_LOG_TRANSFORMATION:
                 mask = y_t > 1e-9
                 y_t = torch.log(y_t[mask])
                 x_t = x_t[mask] # Assuming 1D sync
        else:
            y_t = y_values.to(self.device).to(torch.float64)
            
        if x_t.ndim == 1: x_t = x_t.unsqueeze(1)
        if y_t.ndim == 1: y_t = y_t.unsqueeze(0) # [1, N] ? Evaluation expects [B, Pop, D]?
        # Engine check: `evaluate_batch` takes inputs and handles broadcasting.
        # But `evaluate_batch` in `engine.py` expected y_target as [D] (1D) or [1, D]?
        # Line 1314: `target_matrix = y_target.unsqueeze(0).expand(B, -1)`
        # This implies y_target should be [D].
        if y_t.ndim == 2 and y_t.shape[0] == 1: y_t = y_t.squeeze(0)
            
        # 2. Init
        population = self.initialize_population()
        pop_constants = torch.randn(self.pop_size, self.max_constants, device=self.device, dtype=torch.float64)
        
        # Seeds
        if seeds:
            seed_pop, seed_consts = self.load_population_from_strings(seeds)
            if seed_pop is not None:
                n = min(seed_pop.shape[0], self.pop_size)
                population[:n] = seed_pop[:n]
                pop_constants[:n] = seed_consts[:n]
                
        # Patterns
        pats = self.detect_patterns(y_t.cpu().numpy().flatten())
        if pats:
            pat_pop, pat_consts = self.load_population_from_strings(pats)
            if pat_pop is not None:
                 offset = len(seeds) if seeds else 0
                 n = min(pat_pop.shape[0], self.pop_size - offset)
                 if n > 0:
                     population[offset:offset+n] = pat_pop[:n]
                     pop_constants[offset:offset+n] = pat_consts[:n]

        best_rmse = float('inf')
        best_rpn = None
        best_consts_vec = None
        stagnation = 0
        generations = 0
        current_mutation_rate = GpuGlobals.BASE_MUTATION_RATE
        COMPLEXITY_PENALTY = GpuGlobals.COMPLEXITY_PENALTY

        while True:
            # Timeout
            if timeout_sec and (time.time() - start_time) >= timeout_sec: break
            if generations >= GpuGlobals.GENERATIONS: break
            
            generations += 1
            
            # Eval
            fitness_rmse = self.evaluator.evaluate_batch(population, x_t, y_t, pop_constants)
            
            # Optimize Top K
            k_opt = min(self.pop_size, 200)
            _, top_idx = torch.topk(fitness_rmse, k_opt, largest=False)
            
            opt_pop = population[top_idx]
            opt_consts = pop_constants[top_idx]
            
            refined_consts, refined_mse = self.optimizer.optimize_constants(opt_pop, opt_consts, x_t, y_t, steps=10)
            pop_constants[top_idx] = refined_consts
            fitness_rmse[top_idx] = refined_mse
            
            # Selection Preparation
            lengths = (population != PAD_ID).sum(dim=1).float()
            fitness_penalized = fitness_rmse * (1.0 + COMPLEXITY_PENALTY * lengths) + lengths * 1e-6
            fitness_penalized = self.operators.tarpeian_control(population, fitness_penalized)
            
            # Best Tracking
            min_rmse, min_idx = torch.min(fitness_rmse, dim=0)
            if min_rmse.item() < best_rmse:
                best_rmse = min_rmse.item()
                best_rpn = population[min_idx].clone()
                best_consts_vec = pop_constants[min_idx].clone()
                stagnation = 0
                current_mutation_rate = GpuGlobals.BASE_MUTATION_RATE
                if callback: callback(generations, best_rmse, best_rpn, best_consts_vec, True, 0)
            else:
                stagnation += 1
                
            if callback and generations % GpuGlobals.PROGRESS_REPORT_INTERVAL == 0:
                callback(generations, best_rmse, best_rpn, best_consts_vec, False, -1)

            # Migration
            if self.n_islands > 1 and generations % GpuGlobals.MIGRATION_INTERVAL == 0:
                population, pop_constants = self.migrate_islands(population, pop_constants, fitness_rmse)

            # Cataclysm / Reset
            if stagnation >= GpuGlobals.STAGNATION_LIMIT:
                 # Logic to reset 90%
                 # self.operators doesn't have cataclysm but we can just use init logic
                 # Or implement cataclysm_population in Operators (missed that one too)
                 # Doing it inline:
                 n_elites = int(self.pop_size * 0.10)
                 n_random = self.pop_size - n_elites
                 sorted_idx = torch.argsort(fitness_rmse)
                 elites = population[sorted_idx[:n_elites]]
                 elite_c = pop_constants[sorted_idx[:n_elites]]
                 
                 new_pop = self.operators.generate_random_population(n_random)
                 new_c = torch.randn(n_random, self.max_constants, device=self.device, dtype=torch.float64)
                 
                 population = torch.cat([elites, new_pop])
                 pop_constants = torch.cat([elite_c, new_c])
                 stagnation = 0
                 continue

            # Dynamic Mutation
            if stagnation > 10:
                current_mutation_rate = min(0.4, GpuGlobals.BASE_MUTATION_RATE + (stagnation - 10) * 0.01)
            else:
                current_mutation_rate = GpuGlobals.BASE_MUTATION_RATE
                
            # Evolution Step
            next_pop_list = []
            next_c_list = []
            
            # Elitism
            # Elitism & Selection
            k_elite = max(1, int(self.pop_size * 0.05))
            
            if GpuGlobals.USE_PARETO_SELECTION:
                # 1. Compute Ranks and Crowding (Vectorized GPU)
                ranks, crowding = self.pareto.compute_ranks_and_crowding(fitness_rmse, lengths)
                
                # 2. Select Elites (Sort by Rank ASC, Crowding DESC)
                # Composite score: rank - (normalized_crowding). We want to Minimize this.
                # Crowding can be inf. Handle carefully.
                crowd_safe = torch.nan_to_num(crowding, posinf=1e9)
                max_c = crowd_safe.max() + 1e-6
                score = ranks.double() - (crowd_safe / max_c)
                
                _, elite_sorted = torch.sort(score)
                elite_idx = elite_sorted[:k_elite]
                
                # 3. Tournament Selection Metric
                selection_metric = score # Min is best
            else:
                _, elite_idx = torch.topk(fitness_penalized, k_elite, largest=False)
                selection_metric = fitness_penalized # Min is best
            
            next_pop_list.append(population[elite_idx])
            next_c_list.append(pop_constants[elite_idx])
            
            remaining = self.pop_size - k_elite
            n_cross = int(remaining * GpuGlobals.DEFAULT_CROSSOVER_RATE)
            n_mut = remaining - n_cross
            
            if n_cross > 0:
                # Tournament
                idx = torch.randint(0, self.pop_size, (n_cross, GpuGlobals.DEFAULT_TOURNAMENT_SIZE), device=self.device)
                candidates_metric = selection_metric[idx]
                best_local_idx = torch.argmin(candidates_metric, dim=1) # Returns 0..TournamentSize
                
                # Need global indices from idx
                parents_idx = idx.gather(1, best_local_idx.unsqueeze(1)).squeeze(1)
                
                parents = population[parents_idx]
                c_parents = pop_constants[parents_idx]
                
                offspring = self.operators.crossover_population(parents, 1.0) # Always cross selected
                next_pop_list.append(offspring)
                next_c_list.append(c_parents)
                
            if n_mut > 0:
                idx = torch.randint(0, self.pop_size, (n_mut, GpuGlobals.DEFAULT_TOURNAMENT_SIZE), device=self.device)
                candidates_metric = selection_metric[idx]
                best_local_idx = torch.argmin(candidates_metric, dim=1)
                parents_idx = idx.gather(1, best_local_idx.unsqueeze(1)).squeeze(1)
                
                parents = population[parents_idx]
                c_parents = pop_constants[parents_idx]
                
                offspring = self.operators.mutate_population(parents, current_mutation_rate)
                next_pop_list.append(offspring)
                next_c_list.append(c_parents)
                
            population = torch.cat(next_pop_list)[:self.pop_size]
            pop_constants = torch.cat(next_c_list)[:self.pop_size]
            
            # Deduplication
            if GpuGlobals.PREVENT_DUPLICATES and generations % 1 == 0:
                 population, pop_constants, n_dups = self.operators.deduplicate_population(population, pop_constants)
                 
            # Simplification
            if GpuGlobals.USE_SIMPLIFICATION and generations % 500 == 0:
                 population, pop_constants, _ = self.simplifier.simplify_population(population, pop_constants, top_k=50)

            if best_rmse < 1e-7:
                 return self.rpn_to_infix(best_rpn, best_consts_vec)
                 
        if best_rpn is not None:
             return self.rpn_to_infix(best_rpn, best_consts_vec)
        return None
