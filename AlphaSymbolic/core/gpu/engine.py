
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
from .gpu_simplifier import GPUSymbolicSimplifier

class TensorGeneticEngine:
    def __init__(self, device=None, pop_size=None, max_len=30, num_variables=1, max_constants=5, n_islands=None, model=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model # Neural Model (AlphaSymbolicModel)
        
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

        # --- Precision Setup ---
        self.dtype = torch.float32 if GpuGlobals.USE_FLOAT32 else torch.float64
        print(f"[Engine] Precision Mode: {self.dtype}")

        # --- Double Buffering Setup ---
        # Pre-allocate 2 sets of buffers to avoid churn
        self.pop_buffer_A = torch.full((self.pop_size, self.max_len), PAD_ID, dtype=torch.long, device=self.device)
        self.pop_buffer_B = torch.full((self.pop_size, self.max_len), PAD_ID, dtype=torch.long, device=self.device)
        self.const_buffer_A = torch.zeros((self.pop_size, self.max_constants), dtype=self.dtype, device=self.device)
        self.const_buffer_B = torch.zeros((self.pop_size, self.max_constants), dtype=self.dtype, device=self.device)

        # --- Sub-components ---
        self.grammar = GPUGrammar(num_variables)
        
        # Pass dtype to sub-components
        self.evaluator = GPUEvaluator(self.grammar, self.device, dtype=self.dtype)
        self.operators = GPUOperators(self.grammar, self.device, self.pop_size, self.max_len, self.num_variables, dtype=self.dtype)
        self.optimizer = GPUOptimizer(self.evaluator, self.operators, self.device, dtype=self.dtype)
        self.simplifier = GPUSimplifier(self.grammar, self.device, self.max_constants)
        self.gpu_simplifier = GPUSymbolicSimplifier(self.grammar, self.device, dtype=self.dtype)
        
        # --- Advanced Features ---
        self.sniper = Sniper(self.device)
        self.pareto = ParetoOptimizer(self.device, GpuGlobals.PARETO_MAX_FRONT_SIZE, dtype=self.dtype)
        self.pattern_memory = PatternMemory(
            self.device, 
            self.operators,
            max_patterns=100,
            fitness_threshold=GpuGlobals.PATTERN_RECORD_FITNESS_THRESHOLD,
            min_uses=GpuGlobals.PATTERN_MEM_MIN_USES,
            dtype=self.dtype
        )
        
        # --- Persistent Cache for Initial Population ---
        self._cached_initial_pop = None
        self._cached_initial_consts = None

    # --- Wrappers for backward compatibility and convenience ---

    def evaluate_batch(self, population: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor, constants: torch.Tensor = None) -> torch.Tensor:
        return self.evaluator.evaluate_batch(population, x, y_target, constants)

    def optimize_constants(self, population, constants, x, y, steps=10, lr=0.1):
        return self.optimizer.optimize_constants(population, constants, x, y, steps, lr)
        
    def simplify_population(self, population, constants, top_k=None):
        # GPU-native symbolic simplifier (vectorized, no CPU roundtrip)
        if top_k is None:
            # For the GPU simplifier, we can afford more than 5.
            # But let's keep a reasonable limit to avoid bloating formulas too much every gen.
            top_k = min(100, population.shape[0])
            
        return self.gpu_simplifier.simplify_batch(population, constants)

    def infix_to_rpn(self, formulas: List[str]) -> torch.Tensor:
        return self.operators._infix_list_to_rpn(formulas)
        
    def rpn_to_infix(self, rpn_tensor: torch.Tensor, constants: torch.Tensor = None) -> str:
        if not GpuGlobals.USE_SYMPY:
             # Basic conversion without SymPy cleanup
             return self.simplifier.rpn_to_infix_static(rpn_tensor, constants, self.grammar)
             
        res = self.simplifier._rpn_to_infix_str(rpn_tensor, constants)
        if res == "Invalid":
             # Fallback to basic
             return self.simplifier.rpn_to_infix_static(rpn_tensor, constants, self.grammar)
        return res

    def predict_individual(self, rpn: torch.Tensor, consts: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts y for a single individual.
        rpn: [L]
        consts: [MaxConstants]
        x: [N, Vars] or [Vars, N] or [N]
        Returns: [N] predictions
        """
        # Wrap as batch of 1
        pop = rpn.unsqueeze(0)
        c = consts.unsqueeze(0)
        
        # Standardize x to [Vars, N]
        if x.dim() == 1:
            x_for_vm = x.unsqueeze(0)
        elif x.dim() == 2:
            # If x is [N, Vars], and Vars matches num_variables, transpose.
            # But wait, how do we know which is which? 
            # We can check against self.num_variables.
            if x.shape[1] == self.num_variables:
                x_for_vm = x.T.contiguous()
            else:
                x_for_vm = x
        else:
            x_for_vm = x

        # Run VM directly
        preds_flat, sp, err = self.evaluator._run_vm(pop, x_for_vm, c)
        
        # Reshape to [N]
        return preds_flat

    def attempt_residual_boost(self, best_rpn, best_consts, x_t, y_t):
        """
        Tries to fit the residual error using The Sniper.
        Returns: (NewFormulaStr, NewRMSE) or (None, None)
        """
        try:
             # 1. Get current predictions
             y_pred = self.predict_individual(best_rpn, best_consts, x_t)
             
             # Optimize: Don't move to CPU yet
             # x_cpu = x_t.cpu().numpy()
             # y_cpu = y_t.cpu().numpy()
             # pred_cpu = y_pred.detach().cpu().numpy()
             
             best_boost_str = None
             
             # --- Additive Boost: y = f(x) + g(x) ---
             # g(x) = y - f(x)
             # Optimization: Do this on GPU
             res_add = y_t - y_pred.detach()
             
             # Check SNR: if residual is just noise (very small), skip
             # Use torch.std which is on GPU
             if torch.std(res_add) > 1e-6:
                 boost_add = self.sniper.run(x_t, res_add)
                 if boost_add:
                     # Construct new formula
                     base_str = self.rpn_to_infix(best_rpn, best_consts)
                     new_str = f"({base_str} + {boost_add})"
                     best_boost_str = new_str
                     
             # --- Multiplicative Boost: y = f(x) * g(x) ---
             # g(x) = y / f(x)
             # Avoid div by zero
             if best_boost_str is None: # Only try if additive failed (priority to additive)
                 # Optimization: GPU masking
                 mask = torch.abs(y_pred) > 1e-6
                 if mask.sum() > len(mask) * 0.9: # 90% valid
                     res_mult = torch.zeros_like(y_t)
                     valid_pred = y_pred[mask]
                     valid_y = y_t[mask]
                     
                     res_mult[mask] = valid_y / valid_pred
                     
                     if torch.std(res_mult) > 1e-6:
                         boost_mult = self.sniper.run(x_t, res_mult)
                         if boost_mult:
                              base_str = self.rpn_to_infix(best_rpn, best_consts)
                              new_str = f"({base_str} * {boost_mult})"
                              best_boost_str = new_str
                              
             if best_boost_str:
                 # Evaluate
                 # We need to compile it to RPN and eval
                 pop_boost, const_boost = self.load_population_from_strings([best_boost_str])
                 if pop_boost is not None:
                      rmse = self.evaluator.evaluate_batch(pop_boost, x_t, y_t, const_boost)
                      return best_boost_str, rmse.item(), pop_boost[0], const_boost[0]
                      
        except Exception as e:
             # print(f"Boost failed: {e}")
             pass
        return None, None, None, None

    def get_tree_size(self, rpn_tensor: torch.Tensor) -> int:
        return (rpn_tensor != PAD_ID).sum().item()

    def load_population_from_strings(self, formulas: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        rpn_list = []
        const_list = []
        
        for f in formulas:
            try:
                tree = ExpressionTree.from_infix(f)
                if not tree.is_valid:
                     rpn_list.append(torch.full((self.max_len,), PAD_ID, dtype=torch.long, device=self.device))
                     # FIX: Use self.dtype
                     const_list.append(torch.zeros(self.max_constants, dtype=self.dtype, device=self.device))
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
                     # Harmonize aliases to canonical GPU tokens
                     if t in ['g', 'lgamma']: t = 'lgamma'
                     elif t == '!': t = 'fact' # Standardize ! to fact for GPU
                     elif t in ['S', 'asin']: t = 'asin'
                     elif t in ['T', 'atan']: t = 'atan'
                     elif t in ['_', 'floor']: t = 'floor'
                     elif t == 'e': 
                         # Disambiguate: if 'e' is followed by arguments it's exp, 
                         # but here we have flat RPN. 
                         # ExpressionTree from_infix usually maps exp() to 'exp' 
                         # and constant e to 'e'.
                         pass 
                     elif t == 'exp': t = 'exp'
                     elif t in ['%', 'mod']: t = '%' 
                     elif t in ['^', 'pow']: t = 'pow' 
                     
                     if t in self.grammar.terminals and t not in ['C', '0', '1', '2', '3', '5', '10'] and not t.startswith('x'):
                         clean_tokens.append(t)
                     elif (t.replace('.','',1).isdigit() or (t.startswith('-') and t[1:].replace('.','',1).isdigit())):
                         if t in ['0', '1', '2', '3', '5', '10']:
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
                # FIX: Use self.dtype
                const_list.append(torch.tensor(const_values, dtype=self.dtype, device=self.device))
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

    def neural_flash_injection(self, x_t, y_t):
        """
        Uses the Neural Model to hallucinate new candidates based on current knowledge.
        "Neural Flash" - Insight injection.
        """
        if self.model is None: return None, None
        
        try:
             # Lazy import
             from search.beam_search import beam_solve
             
             # Convert data for Beam Search (needs CPU or specific shape)
             # beam_solve expects raw x, y
             # Our x_t, y_t might be log-transformed if USE_LOG = True
             # The model was likely trained on Raw or Log?
             # Assuming Model handles its own preprocessing or expects scaled data.
             # Let's pass what we have (y_t is target).
             
             # Sample for speed (don't run beam on 10k points)
             n_samples = min(len(x_t), 100)
             idx = torch.randperm(len(x_t))[:n_samples]
             x_sub = x_t[idx]
             y_sub = y_t[idx]
             
             results = beam_solve(
                 x_sub, y_sub, self.model, self.device, 
                 beam_width=10, 
                 num_variables=self.num_variables,
                 verbose=False
             )
             
             candidates = [r['formula'] for r in results if 'formula' in r and not r['formula'].startswith("Partial")]
             if candidates:
                 return self.load_population_from_strings(candidates)
                 
        except Exception as e:
            # print(f"[Engine] Neural Flash Failed: {e}")
            pass
            
        return None, None
        
    def alpha_mcts_refinement(self, x_t, y_t):
        """
        Runs Monte Carlo Tree Search to find 'Out of This World' structural solutions.
        This is the 'Deep Thought' mode.
        """
        if self.model is None: return None
        
        try:
             # Lazy import
             from search.mcts import MCTS
             
             # Sample for speed (MCTS is slow)
             n_samples = min(len(x_t), 50)
             idx = torch.randperm(len(x_t))[:n_samples]
             x_sub = x_t[idx].cpu().numpy()
             y_sub = y_t[idx].cpu().numpy()
             
             mcts = MCTS(
                 self.model, self.device, 
                 n_simulations=50, # Quick burst
                 max_depth=30,
                 num_variables=self.num_variables
             )
             
             result = mcts.search(x_sub, y_sub)
             if result['formula'] and result['formula'] != "None":
                 return result['formula']
                 
        except Exception as e:
             # print(f"[Engine] Alpha MCTS Failed: {e}")
             pass
        return None

    def migrate_islands(self, population, constants, fitness):
        if self.n_islands <= 1: return population, constants
        
        island_size = self.island_size
        mig_size = min(GpuGlobals.MIGRATION_SIZE, island_size // 2)
        
        # 1. Reshape fitness to [n_islands, island_size]
        fit_view = fitness.view(self.n_islands, island_size)
        
        # 2. Find best and worst indices in each island (Vectorized)
        _, best_local_idx = torch.topk(fit_view, mig_size, dim=1, largest=False)
        _, worst_local_idx = torch.topk(fit_view, mig_size, dim=1, largest=True)
        
        # 3. Convert to global indices
        island_offsets = torch.arange(0, self.pop_size, island_size, device=self.device).view(self.n_islands, 1)
        best_global_idx = (best_local_idx + island_offsets).view(-1)
        worst_global_idx = (worst_local_idx + island_offsets).view(-1)
        
        # 4. Extract Migrants
        migrants_pop = population[best_global_idx].view(self.n_islands, mig_size, self.max_len)
        migrants_const = constants[best_global_idx].view(self.n_islands, mig_size, self.max_constants)
        
        # 5. Circular Shift (Migration Logic): Island i -> Island (i+1)%N
        # We roll the first dimension to the right
        shifted_migrants_pop = torch.roll(migrants_pop, shifts=1, dims=0).view(-1, self.max_len)
        shifted_migrants_const = torch.roll(migrants_const, shifts=1, dims=0).view(-1, self.max_constants)
        
        # 6. Apply to next buffers (In-place on current clone)
        pop_out = population.clone()
        const_out = constants.clone()
        pop_out[worst_global_idx] = shifted_migrants_pop
        const_out[worst_global_idx] = shifted_migrants_const
        
        return pop_out, const_out

    def tournament_selection_island(self, population, fitness, n_islands, tournament_size=3):
        """
        Performs tournament selection within each island.
        Returns: Flattened indices of selected parents [PopSize].
        """
        B = population.shape[0]
        island_size = B // n_islands
        
        # Ensure divisible
        if B % n_islands != 0:
            raise ValueError(f"Population {B} not divisible by islands {n_islands}")
            
        # Fitness view: [Islands, IslandSize]
        fit_view = fitness.view(n_islands, island_size)
        
        # We need to select B parents (replacing full population usually? or just n parents?)
        # Usually same size.
        
        
        # We generate random indices for tournaments: [Islands, IslandSize, TourSize]
        # These are local indices 0..IslandSize-1
        rand_idx = torch.randint(0, island_size, (n_islands, island_size, tournament_size), device=self.device)
        
        # Global Indices approach is needed because gather requires matching dims.
        # Island Start Offsets: [0, 100, 200...]
        offsets = torch.arange(0, B, island_size, device=self.device).view(n_islands, 1, 1)
        
        # Global Random Indices logic:
        # local indices (0..size) + offsets
        global_rand_idx = rand_idx + offsets
        
        # Flatten to select fitness: [B * TourSize]
        flat_all_idx = global_rand_idx.view(-1)
        
        try:
             import rpn_cuda_native
             # CUDA Fast Path
             # We need indices as [B, TourSize]
             # flat_all_idx is [B * TourSize]. Reshape to [B, TourSize]
             all_idx_mat = flat_all_idx.view(B, tournament_size)
             
             winner_indices = torch.zeros(B, dtype=torch.long, device=self.device)
             
             # Fitness needs to be float32 for current kernel?
             # Check kernel signature. It takes float* fitness.
             # If engine uses float64, we might need casting or template update.
             # My kernel assumed float32. Let's cast to safety.
             
             fit_f32 = fitness.float()
             
             rpn_cuda_native.tournament_selection(fit_f32, all_idx_mat, winner_indices)
             
             return winner_indices
        except ImportError:
             pass
        
        flat_fitness = fitness[flat_all_idx].view(B, tournament_size)
        
        # ArgMin per row (Tournament Winners)
        # For fitness, smaller is better (RMSE) -> agrmin
        _, win_local_pos = torch.min(flat_fitness, dim=1) # [B] (0..TourSize-1)
        
        # Get actual global index of winners
        # Gather from flat_all_idx reshaped to [B, TourSize]
        flat_candidates = flat_all_idx.view(B, tournament_size)
        winner_indices = flat_candidates.gather(1, win_local_pos.unsqueeze(1)).squeeze(1)
        
        return winner_indices

    def crossover_population(self, population, rate):
        return self.operators.crossover_population(population, rate)

    def mutate_population(self, population, rate):
        return self.operators.mutate_population(population, rate)
    
    def evolve_generation_cuda(self, population, constants, fitness, x_t, y_t, 
                                mutation_rate=0.1, crossover_rate=0.5, 
                                tournament_size=3, pso_steps=10, pso_particles=20):
        """
        Full C++ CUDA Orchestrator: Selection + Crossover + Mutation + PSO.
        Calls the native rpn_cuda_native.evolve_generation function.
        
        Returns: (new_population, new_constants, new_fitness)
        """
        try:
            import rpn_cuda_native as rpn_cuda
            from .cuda_vm import CudaRPNVM
        except ImportError:
            print("[Engine] WARNING: CUDA orchestrator not available, falling back to Python")
            return None, None, None
        
        # Get VM for OpCode IDs
        vm = CudaRPNVM(self.grammar, self.device)
        
        # Get arity tensors
        token_arities = self.grammar.get_arity_tensor(self.device)
        arity_0_ids = self.grammar.get_arity_ids(0, self.device)
        arity_1_ids = self.grammar.get_arity_ids(1, self.device)
        arity_2_ids = self.grammar.get_arity_ids(2, self.device)
        
        # Ensure X is [Vars, N] for RPN kernel
        if x_t.ndim == 1:
            x_in = x_t.unsqueeze(0)  # [1, N]
        elif x_t.ndim == 2 and x_t.shape[1] != population.shape[0]:  # [N, Vars] -> [Vars, N]
            x_in = x_t.T.contiguous()
        else:
            x_in = x_t
        
        # Ensure y is [N]
        y_in = y_t.squeeze() if y_t.ndim > 1 else y_t
        
        result = rpn_cuda.evolve_generation(
            population,
            constants.float(),  # Ensure float32
            fitness.float(),
            x_in.float(),
            y_in.float(),
            token_arities,
            arity_0_ids,
            arity_1_ids,
            arity_2_ids,
            mutation_rate,
            crossover_rate,
            tournament_size,
            pso_steps,
            pso_particles,
            0.5, 1.5, 1.5,  # pso_w, pso_c1, pso_c2
            vm.PAD_ID,
            # OpCodes
            vm.id_x_start,
            vm.id_C, vm.id_pi, vm.id_e,
            vm.id_0, vm.id_1, vm.id_2, vm.id_3, vm.id_5, vm.id_10,
            vm.op_add, vm.op_sub, vm.op_mul, vm.op_div, vm.op_pow, vm.op_mod,
            vm.op_sin, vm.op_cos, vm.op_tan,
            vm.op_log, vm.op_exp,
            vm.op_sqrt, vm.op_abs, vm.op_neg,
            vm.op_fact, vm.op_floor, vm.op_ceil, vm.op_sign,
            vm.op_gamma, vm.op_lgamma,
            vm.op_asin, vm.op_acos, vm.op_atan,
            3.14159265359, 2.718281828
        )
        
        new_pop, new_consts, new_fit = result[0], result[1], result[2]
        
        # Convert constants back to engine dtype if needed
        if self.dtype != torch.float32:
            new_consts = new_consts.to(self.dtype)
            new_fit = new_fit.to(self.dtype)
        
        return new_pop, new_consts, new_fit


    def deterministic_crowding(self, parents, offspring, parent_fitness, off_fitness):
        """
        Deterministic Crowding: Compare Parent vs Offspring.
        Keep the one with better fitness (lower RMSE).
        """
        # Ensure shapes match
        if parents.shape != offspring.shape:
             raise ValueError("Parents/Offspring shape mismatch")
             
        # Broadcast fitness for masking
        # parent_fitness: [B]
        # mask: True if offspring better (smaller)
        mask = off_fitness < parent_fitness
        
        # Select Population
        # mask shape [B]. Pop shape [B, L]. Need broadcast.
        mask_pop = mask.unsqueeze(1).expand_as(parents)
        new_pop = torch.where(mask_pop, offspring, parents)
        
        # Select Fitness
        new_fitness = torch.where(mask, off_fitness, parent_fitness)
        
        return new_pop, new_fitness

    def epsilon_lexicase_selection(self, population, n_parents, x, y_target, constants):
        """
        Low-VRAM Epsilon-Lexicase Selection.
        Instead of pre-calculating [Pop x Cases] Error Matrix (OOM on 4GB),
        we pick Tournament Candidates first, then evaluate ONLY those candidates.
        """
        B, L = population.shape
        N_cases = y_target.flatten().shape[0]
        tour_size = 32 # Reduced tour size for memory safety
        
        # 1. Select Candidates: [n_parents, tour_size]
        rand_idx = torch.randint(0, B, (n_parents, tour_size), device=self.device)
        
        # 2. Extract Candidates Population & Constants
        flat_idx = rand_idx.view(-1)
        candidates_pop = population[flat_idx] # [n_parents*tour, L]
        candidates_c = constants[flat_idx]
        
        # 3. Evaluate ONLY Candidates (Chunked by design since n_parents*tour is small)
        # We need errors per case: [TotalCandidates, N_cases]
        # evaluate_batch returns RMSE (mean). We need evaluate_batch_full.
        # But wait, evaluate_batch_full itself was the OOM cause if run on full pop.
        # Here we run it on (n_parents * tour_size).
        # If n_parents=2000, tour=32 -> 64,000 candidates. Might still be too big.
        # We should run this loop in chunks too!
        
        total_candidates = n_parents * tour_size
        all_errors = []
        chunk_size = 50000 # Evaluate 50k candidates at a time
        
        for i in range(0, total_candidates, chunk_size):
            end = min(total_candidates, i + chunk_size)
            sub_pop = candidates_pop[i:end]
            sub_c = candidates_c[i:end]
            
            # Use evaluate_batch_full which returns [Chunk, Cases] abs errors
            sub_errs = self.evaluator.evaluate_batch_full(sub_pop, x, y_target, sub_c).float() # Use float32 to save memory
            all_errors.append(sub_errs)
        
        flat_errs = torch.cat(all_errors, dim=0) # [n_parents*tour, N_cases]
        
        # Reshape to [n_parents, tour_size, N_cases]
        candidates_err = flat_errs.view(n_parents, tour_size, N_cases)
        
        # 4. Standard Lexicase Logic (Batched)
        
        # Epsilon (approximate using batch median to avoid full pop calc)
        epsilon = torch.zeros(N_cases, device=self.device, dtype=torch.float32)
        # Optional: Calculate real epsilon from a sample of population if needed, 
        # but 0 is fine for standard Lexicase.
        
        # Shuffle Cases
        perm = torch.randperm(N_cases, device=self.device)
        candidates_err_shuffled = candidates_err[:, :, perm]
        
        # Elimination Loop
        active_mask = torch.ones((n_parents, tour_size), dtype=torch.bool, device=self.device)
        
        for i in range(N_cases):
            # Check active
            active_counts = active_mask.sum(dim=1)
            # Optimization: If all rows have 1 candidate left, stop early
            if (active_counts == 1).all():
                break
                
            curr_case_err = candidates_err_shuffled[:, :, i]
            
            # Find best among active
            masked_err = torch.where(active_mask, curr_case_err, torch.tensor(float('inf'), device=self.device))
            min_err, _ = torch.min(masked_err, dim=1) # [n_parents]
            
            # Threshold
            threshold = min_err # + epsilon (0)
            
            # Update Mask
            keep = (curr_case_err <= threshold.unsqueeze(1) + 1e-9)
            active_mask = active_mask & keep
            
        # 5. Pick Winner
        winners_local = torch.argmax(active_mask.int(), dim=1) 
        winners_global = rand_idx.gather(1, winners_local.unsqueeze(1)).squeeze(1)
        
        return winners_global        


    def run(self, x_values, y_values, seeds: List[str] = None, timeout_sec: int = 10, callback=None, use_log: bool = None):
        # 1. Data Setup
        # FIX: Use self.dtype
        if not isinstance(x_values, torch.Tensor):
            x_t = torch.tensor(x_values, dtype=self.dtype, device=self.device)
        else:
            x_t = x_values.to(self.device).to(self.dtype)
            
        if not isinstance(y_values, torch.Tensor):
            y_t = torch.tensor(y_values, dtype=self.dtype, device=self.device)
            # Log transform if needed (Use arg if provided, else Global)
            use_log_local = use_log if use_log is not None else GpuGlobals.USE_LOG_TRANSFORMATION
            
            if use_log_local:
                 mask = y_t > 1e-9
                 y_t = torch.log(y_t[mask])
                 x_t = x_t[mask] 
        else:
            y_t = y_values.to(self.device).to(self.dtype)
            
        if x_t.ndim == 1: x_t = x_t.unsqueeze(1)
        if y_t.ndim == 1: y_t = y_t.unsqueeze(0) 

        # Line 1314: `target_matrix = y_target.unsqueeze(0).expand(B, -1)`
        # This implies y_target should be [D].
        if y_t.ndim == 2 and y_t.shape[0] == 1: y_t = y_t.squeeze(0)
        
        # --- 1.5 The Sniper (Intelligence Check) ---
        # Before doing heavy lifting, check for simple patterns (Linear, Geometric)
        if GpuGlobals.USE_SNIPER:
            # Use CPU numpy for sniper
            sniper_formula = self.sniper.run(x_t.cpu().numpy(), y_t.cpu().numpy())
            if sniper_formula:
                print(f"[Engine] The Sniper found a candidate solution: {sniper_formula}")
                if seeds is None: seeds = []
                seeds.append(sniper_formula)
            
        # 2. Init with Disk Cache and Double Buffering
        import os
        # Use abs path relative to this file to avoid CWD issues
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(base_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        # Use dtype in filename to separate caches
        prec_str = "fp32" if self.dtype == torch.float32 else "fp64"
        cache_file = os.path.join(cache_dir, f"initial_pop_v2_{self.pop_size}_{self.max_len}_{self.num_variables}_{prec_str}.pt")
        
        loaded_from_cache = False
        
        # Priority 1: Memory Cache
        if self._cached_initial_pop is not None:
             self.pop_buffer_A[:] = self._cached_initial_pop.to(self.device)
             self.const_buffer_A[:] = self._cached_initial_consts.to(self.device).to(self.dtype)
             population = self.pop_buffer_A
             pop_constants = self.const_buffer_A
             loaded_from_cache = True
             
        # Priority 2: Disk Cache
        elif os.path.exists(cache_file):
             try:
                 print(f"[GPU Engine] Loading from cache: {cache_file}")

                 data = torch.load(cache_file, map_location=self.device, weights_only=False)
                 if data['pop'].shape == self.pop_buffer_A.shape:
                      self.pop_buffer_A[:] = data['pop']
                      self.const_buffer_A[:] = data['const'].to(self.dtype)
                      population = self.pop_buffer_A
                      pop_constants = self.const_buffer_A
                      loaded_from_cache = True
                      print("[GPU Engine] Cache Loaded Successfully.")
                      
                      # Cache in memory
                      self._cached_initial_pop = self.pop_buffer_A.clone()
                      self._cached_initial_consts = self.const_buffer_A.clone()
                 else:
                      print("[GPU Engine] Cache Mismatch (Shapes). Regenerating...")
             except Exception as e:
                 print(f"[GPU Engine] Cache Load Failed ({e}). Regenerating...")
        
        if not loaded_from_cache:
             print(f"[GPU Engine] Generating {self.pop_size:,} random formulas (First time, please wait)...")
             temp_pop = self.operators.generate_random_population(self.pop_size)
             temp_c = torch.randn(self.pop_size, self.max_constants, device=self.device, dtype=self.dtype)
             
             self.pop_buffer_A[:] = temp_pop
             self.const_buffer_A[:] = temp_c
             
             population = self.pop_buffer_A
             pop_constants = self.const_buffer_A
             
             # Save atómicamente para evitar colisiones en paralelo
             if not os.path.exists(cache_file):
                 try:
                     print(f"[GPU Engine] Saving to cache: {cache_file}...")
                     temp_cache = cache_file + ".tmp"
                     torch.save({'pop': temp_pop.cpu(), 'const': temp_c.cpu()}, temp_cache)
                     os.replace(temp_cache, cache_file)
                     print("[GPU Engine] Saved.")
                 except Exception as e:
                     if os.path.exists(temp_cache): os.remove(temp_cache)
                     print(f"[GPU Engine] Warning: Could not save cache ({e})")
                 
             # Cache in memory
             self._cached_initial_pop = self.pop_buffer_A.clone()
             self._cached_initial_consts = self.const_buffer_A.clone()
        
        # Seeds
        if seeds:
            seed_pop, seed_consts = self.load_population_from_strings(seeds)
            if seed_pop is not None:
                n = min(seed_pop.shape[0], self.pop_size)
                # Update Buffer directly
                population[:n] = seed_pop[:n]
                pop_constants[:n] = seed_consts[:n].to(self.dtype)
            else:
                print("[DEBUG] CRITICAL: Seed loading returned None!")
                
        # Patterns
        pats = self.detect_patterns(y_t.cpu().numpy().flatten())
        if pats:
            pat_pop, pat_consts = self.load_population_from_strings(pats)
            if pat_pop is not None:
                offset = len(seeds) if seeds else 0
                n = min(pat_pop.shape[0], self.pop_size - offset)
                if n > 0:
                    population[offset:offset+n] = pat_pop[:n]
                    pop_constants[offset:offset+n] = pat_consts[:n].to(self.dtype)

        last_reported_fitness = float('inf')
        best_rmse = float('inf')
        best_rpn = None
        best_consts_vec = None
        stagnation = 0
        generations = 0
        current_mutation_rate = GpuGlobals.BASE_MUTATION_RATE
        COMPLEXITY_PENALTY = GpuGlobals.COMPLEXITY_PENALTY

        start_time = time.time()
        while True:
            # Timeout
            if timeout_sec and (time.time() - start_time) >= timeout_sec: break
            if generations >= GpuGlobals.GENERATIONS: break
            
            generations += 1

            # Migration
            if self.n_islands > 1 and generations % 10 == 0: # Default interval 10
                 population, pop_constants = self.migrate_islands(population, pop_constants, fitness_rmse)

            # --- Neural Flash (Intelligence Injection) ---
            if GpuGlobals.USE_NEURAL_FLASH and self.model is not None and generations % 50 == 0:
                 pop_neural, const_neural = self.neural_flash_injection(x_t, y_t)
                 if pop_neural is not None:
                     # Inject into random island (or spread)

                     n_inj = pop_neural.shape[0]
                     if n_inj > 0:
                         # Replace random losers
                         limit = min(n_inj, self.pop_size // 10)
                         indices = torch.randint(0, self.pop_size, (limit,), device=self.device)
                         population[indices] = pop_neural[:limit]
                         pop_constants[indices] = const_neural[:limit]
                         # print(f"[Generation {generations}] Neural Flash injected {limit} insights.")

            # --- Alpha MCTS (Deep Thought) ---
            if GpuGlobals.USE_ALPHA_MCTS and self.model is not None and generations % 100 == 0:
                 mcts_formula = self.alpha_mcts_refinement(x_t, y_t)
                 if mcts_formula:
                      print(f"[Generation {generations}] Alpha MCTS found structure: {mcts_formula}")

                      pop_mcts, const_mcts = self.load_population_from_strings([mcts_formula])
                      if pop_mcts is not None:
                           # Inject as elite (pos 0)
                           if population is self.pop_buffer_A:
                               self.pop_buffer_A[0] = pop_mcts[0]
                               self.const_buffer_A[0] = const_mcts[0]
                           else:
                               self.pop_buffer_B[0] = pop_mcts[0]
                               self.const_buffer_B[0] = const_mcts[0]

            # --- Pattern Memory Injection ---
            # Inject successful building blocks to accelerate convergence
            if GpuGlobals.USE_PATTERN_MEMORY and generations > 10 and generations % GpuGlobals.PATTERN_INJECT_INTERVAL == 0:
                 # Inject into current population before evaluation
                 pop_inj, const_inj, n_injected = self.pattern_memory.inject_into_population(
                     population, pop_constants, self.grammar, 
                     percent=GpuGlobals.PATTERN_INJECT_PERCENT
                 )
                 if n_injected > 0:
                     population[:] = pop_inj
                     pop_constants[:] = const_inj
                     # print(f"[Generation {generations}] Injected {n_injected} patterns.")

            # Eval
            # If Lexicase, we need FULL errors
            if GpuGlobals.USE_LEXICASE_SELECTION:
                abs_errors = self.evaluator.evaluate_batch_full(population, x_t, y_t, pop_constants)
                fitness_rmse = torch.mean(abs_errors**2, dim=1).sqrt() # Approx RMSE for stats
            else:
                fitness_rmse = self.evaluator.evaluate_batch(population, x_t, y_t, pop_constants)
                abs_errors = None
                
            # --- Pattern Memory Recording ---
            # Record successful subtrees
            if GpuGlobals.USE_PATTERN_MEMORY and generations % 20 == 0:  # Reduced frequency for CPU relief
                 self.pattern_memory.record_subtrees(
                     population, fitness_rmse, self.grammar, 
                     min_size=3, max_size=12
                 )
            
            # Optimize Top K
            k_opt = min(self.pop_size, 200)
            _, top_idx = torch.topk(fitness_rmse, k_opt, largest=False)
            
            opt_pop = population[top_idx]
            opt_consts = pop_constants[top_idx]
            
            if GpuGlobals.USE_NANO_PSO:
                 refined_consts, refined_mse = self.optimizer.nano_pso(opt_pop, opt_consts, x_t, y_t, steps=50)
            else:
                 refined_consts, refined_mse = self.optimizer.optimize_constants(opt_pop, opt_consts, x_t, y_t, steps=10)
            pop_constants[top_idx] = refined_consts
            fitness_rmse[top_idx] = refined_mse
            
            # Best Tracking
            min_rmse, min_idx = torch.min(fitness_rmse, dim=0)
            
            if not torch.isnan(min_rmse) and min_rmse.item() < best_rmse:
                best_rmse = min_rmse.item()
                best_rpn = population[min_idx].clone()
                best_consts_vec = pop_constants[min_idx].clone()
                self.best_global_rmse = best_rmse
                self.best_global_rpn = best_rpn
                self.best_global_consts = best_consts_vec
                
                # Identify Island
                island_idx = (min_idx.item() // self.island_size) if self.n_islands > 1 else 0

                stagnation = 0
                current_mutation_rate = GpuGlobals.BASE_MUTATION_RATE

                # Only report to user if improvement is significant (> 0.1%)
                # Case 1: First best found (last_reported_fitness is inf)
                # Case 2: Improvement > 0.1%
                is_first = (last_reported_fitness == float('inf'))
                rel_improvement = (last_reported_fitness - best_rmse) / last_reported_fitness if not is_first else 1.0
                
                if callback and (is_first or rel_improvement > 0.001):
                    # Only report to user if fitness is within a reasonable range (not a penalty value)
                    if best_rmse < 1e100:
                        last_reported_fitness = best_rmse
                        callback(generations, best_rmse, best_rpn, best_consts_vec, True, island_idx)
            else:
                stagnation += 1
                # print(f"[DEBUG] Gen {generations}: Stagnation {stagnation} (Best {best_rmse:.6f}, Current Min {min_rmse.item():.6f})")
                
            if callback and generations % GpuGlobals.PROGRESS_REPORT_INTERVAL == 0:
                callback(generations, best_rmse, best_rpn, best_consts_vec, False, -1)

            # Cataclysm / Reset
            if stagnation >= GpuGlobals.STAGNATION_LIMIT:
                 n_elites = int(self.pop_size * 0.10)
                 n_random = self.pop_size - n_elites
                 sorted_idx = torch.argsort(fitness_rmse)
                 elites = population[sorted_idx[:n_elites]]
                 elite_c = pop_constants[sorted_idx[:n_elites]]
                 
                 new_pop = self.operators.generate_random_population(n_random)
                 # FIX: use self.dtype
                 new_c = torch.randn(n_random, self.max_constants, device=self.device, dtype=self.dtype)
                 
                 # Note: Cataclysm creates new tensor, breaking buffer link temporarily.
                 # But we assign it to 'population' variable.
                 # At end of loop, 'population' is used?
                 # No, loop uses 'population' as input.
                 # If we replace 'population', next iteration uses new tensor.
                 # But we want to stay in buffers.
                 # We should Copy TO the current buffer.
                 
                 # Current buffer is whatever 'population' points to?
                 # If 'population' is a Slice or Buffer.
                 # Safest: Write back to current buffer.
                 if population is self.pop_buffer_A:
                      self.pop_buffer_A[:n_elites] = elites
                      self.pop_buffer_A[n_elites:] = new_pop
                      self.const_buffer_A[:n_elites] = elite_c
                      self.const_buffer_A[n_elites:] = new_c
                      population = self.pop_buffer_A
                      pop_constants = self.const_buffer_A
                 elif population is self.pop_buffer_B:
                      self.pop_buffer_B[:n_elites] = elites
                      self.pop_buffer_B[n_elites:] = new_pop
                      self.const_buffer_B[:n_elites] = elite_c
                      self.const_buffer_B[n_elites:] = new_c
                      population = self.pop_buffer_B
                      pop_constants = self.const_buffer_B
                 else:
                      # If logic drifted, just cat. but this breaks buffering.
                      population = torch.cat([elites, new_pop])
                      pop_constants = torch.cat([elite_c, new_c])
                 
                 stagnation = 0
                 continue

            # Dynamic Mutation
            if stagnation > 10:
                current_mutation_rate = min(0.4, GpuGlobals.BASE_MUTATION_RATE + (stagnation - 10) * 0.01)
                
                # --- Residual Boosting (Every 20 stagnation steps) ---
                if GpuGlobals.USE_RESIDUAL_BOOSTING and stagnation % 20 == 0 and best_rpn is not None:
                     boost_str, boost_rmse, boost_rpn, boost_c = self.attempt_residual_boost(best_rpn, best_consts_vec, x_t, y_t)
                     if boost_str and boost_rmse < best_rmse:
                         print(f"[Engine] Residual Boosting SUCCESS! New RMSE: {boost_rmse:.6f}")
                         print(f"         Formula: {boost_str}")
                         # Update Best
                         best_rmse = boost_rmse
                         best_rpn = boost_rpn
                         best_consts_vec = boost_c
                         self.best_global_rmse = best_rmse
                         self.best_global_rpn = best_rpn
                         self.best_global_consts = best_consts_vec
                         stagnation = 0 # Reset!
                         
                         # Inject into population (Elitism slot 0)
                         # We need to write to CURRENT buffers
                         if population is self.pop_buffer_A:
                             self.pop_buffer_A[0] = best_rpn
                             self.const_buffer_A[0] = best_consts_vec
                         else:
                             self.pop_buffer_B[0] = best_rpn
                             self.const_buffer_B[0] = best_consts_vec
                         
            else:
                current_mutation_rate = GpuGlobals.BASE_MUTATION_RATE
                
            # Evolution Step (Vectorized Island Model + Double Buffering)
            island_offsets = torch.arange(0, self.pop_size, self.island_size, device=self.device).view(self.n_islands, 1)

            # Determine Buffers based on generation parity
            # If gen % 2 == 0: Read A, Write B (Logic: Gen started with A, so Next is B)
            # Logic Check:
            # Gen 1 (Odd): Start in A. Write to B.
            # End of loop: population = B.
            # Gen 2 (Even): Start in B. Write to A.
            
            # --- EVOLUTION STEP (Reproduction) ---
            if GpuGlobals.USE_CUDA_ORCHESTRATOR:
                # 1. Selection Metric with Complexity Penalty
                lengths = (population != PAD_ID).sum(dim=1).float()
                selection_metric = fitness_rmse * (1.0 + COMPLEXITY_PENALTY * lengths) + lengths * 1e-6
                
                # 2. Call Full C++ Orchestrator (Selection + Crossover + Mutation)
                # NOTE: We set pso_steps=0 here because running PSO on the ENTIRE population (1M) 
                # would OOM (requires ~5GB for particles).
                # The Top-K PSO optimization is already handled in Python just above.
                next_pop, next_c, next_fit = self.evolve_generation_cuda(
                    population, pop_constants, selection_metric, x_t, y_t,
                    mutation_rate=current_mutation_rate,
                    crossover_rate=GpuGlobals.DEFAULT_CROSSOVER_RATE,
                    tournament_size=GpuGlobals.DEFAULT_TOURNAMENT_SIZE,
                    pso_steps=0, # Disable global PSO to avoid OOM
                    pso_particles=20
                )
                
                if next_pop is not None:
                    # Update local refs for next generation
                    next_pop = next_pop[:self.pop_size]
                    next_c = next_c[:self.pop_size]
                else:
                    # Rare failure (ImportError), proceed to fallback if needed or return
                    print("[Engine] CUDA Orchestrator failed, please restart without USE_CUDA_ORCHESTRATOR")
                    break
            else:
                # --- LEGACY PYTHON EVOLUTION LOOP ---
                # Determine Buffers based on generation parity
                if population is self.pop_buffer_A:
                    next_pop = self.pop_buffer_B
                    next_c = self.const_buffer_B
                else:
                    next_pop = self.pop_buffer_A
                    next_c = self.const_buffer_A
                    
                p_ptr = 0
                
                # 1. Vectorized Elitism
                fit_view = fitness_rmse.view(self.n_islands, self.island_size)
                k_elite_per_island = max(1, int(self.island_size * GpuGlobals.BASE_ELITE_PERCENTAGE))
                total_elites = k_elite_per_island * self.n_islands
                
                _, elite_local_idx = torch.topk(fit_view, k_elite_per_island, dim=1, largest=False)
                elite_global_idx = (elite_local_idx + island_offsets).view(-1)
                
                next_pop[0 : total_elites] = population[elite_global_idx]
                next_c[0 : total_elites] = pop_constants[elite_global_idx]
                p_ptr += total_elites
                
                # 2. Vectorized Selection Setup
                remaining_per_island = self.island_size - k_elite_per_island
                n_cross_per_island = int(remaining_per_island * GpuGlobals.DEFAULT_CROSSOVER_RATE)
                n_mut_per_island = remaining_per_island - n_cross_per_island
                total_cross = n_cross_per_island * self.n_islands
                total_mut = n_mut_per_island * self.n_islands
                
                lengths = (population != PAD_ID).sum(dim=1).float()
                selection_metric = fitness_rmse * (1.0 + COMPLEXITY_PENALTY * lengths) + lengths * 1e-6
                
                def get_island_parents(n_needed_per_island):
                    rand_local = torch.randint(0, self.island_size, (self.n_islands, n_needed_per_island, GpuGlobals.DEFAULT_TOURNAMENT_SIZE), device=self.device)
                    offsets = island_offsets.view(self.n_islands, 1, 1)
                    rand_global = rand_local + offsets
                    flat_rand = rand_global.view(-1)
                    flat_metric = selection_metric[flat_rand]
                    view_metric = flat_metric.view(self.n_islands, n_needed_per_island, GpuGlobals.DEFAULT_TOURNAMENT_SIZE)
                    best_tour_idx = torch.argmin(view_metric, dim=2)
                    winner_global_idx = rand_global.gather(2, best_tour_idx.unsqueeze(2)).squeeze(2)
                    return winner_global_idx.view(-1)

                # --- CROSSOVER ---
                chunk_size = 50000
                if total_cross > 0:
                    parents_idx = get_island_parents(n_cross_per_island)
                    c_ptr = 0
                    while c_ptr < total_cross:
                        curr = min(chunk_size, total_cross - c_ptr)
                        sub_idx = parents_idx[c_ptr : c_ptr + curr]
                        parents = population[sub_idx]
                        offspring = self.operators.crossover_population(parents, 1.0)
                        next_pop[p_ptr : p_ptr + curr] = offspring
                        next_c[p_ptr : p_ptr + curr] = pop_constants[sub_idx]
                        p_ptr += curr
                        c_ptr += curr

                # --- MUTATION ---
                if total_mut > 0:
                    parents_idx = get_island_parents(n_mut_per_island)
                    m_ptr = 0
                    while m_ptr < total_mut:
                        curr = min(chunk_size, total_mut - m_ptr)
                        sub_idx = parents_idx[m_ptr : m_ptr + curr]
                        parents = population[sub_idx]
                        offspring = parents.clone()
                        n_p = curr // 2
                        n_s = curr - n_p
                        if n_p > 0:
                            offspring[:n_p] = self.operators.mutate_population(offspring[:n_p], current_mutation_rate)
                        if n_s > 0:
                            offspring[n_p:] = self.operators.subtree_mutation(offspring[n_p:], 1.0)
                        next_pop[p_ptr : p_ptr + curr] = offspring
                        next_c[p_ptr : p_ptr + curr] = pop_constants[sub_idx]
                        p_ptr += curr
                        m_ptr += curr

            
            # --- SIMPLIFICATION ---
            # --- SIMPLIFICATION ---
            if GpuGlobals.USE_SIMPLIFICATION and generations % GpuGlobals.SIMPLIFICATION_INTERVAL == 0:
                 # Simplify only the BEST formulas in each island to save time
                 # Pattern: Identify top formulas per island
                 k_simplify = GpuGlobals.K_SIMPLIFY
                 
                 # Reshape fitness to islands
                 island_fitness = selection_metric.view(self.n_islands, self.island_size)
                 
                 # Find top K indices within each island
                 # We want the SMALLEST metrics (RMSE + penalty)
                 _, top_local_idx = torch.topk(island_fitness, k_simplify, dim=1, largest=False)
                 
                 # Convert to global indices
                 offsets = island_offsets.view(self.n_islands, 1)
                 top_global_idx = (top_local_idx + offsets).view(-1)
                 
                 # Simplify only these individuals
                 # Next_pop and next_c are already updated for the next generation
                 # but we can simplify them now before the next loop starts.
                 sub_pop = next_pop[top_global_idx]
                 sub_const = next_c[top_global_idx]
                 
                  # Pass to optimized symbolic simplifier (vectorized)
                 try:
                     sim_pop, _, n_s = self.gpu_simplifier.simplify_batch(sub_pop, sub_const)
                 except Exception as e:
                     print(f"CRASH IN SIMPLIFIER: {e}")
                     # Fallback: do not simplify
                     sim_pop = sub_pop
                 
                 # Note: gpu_symbolic_simplifier.simplify_batch only returns population (constants are handled via PADs)
                 
                 # Write back
                 next_pop[top_global_idx] = sim_pop
                 # next_c[top_global_idx] = ... (constants preserved if not PADed)
            
            # Swap Buffers
            population = next_pop
            pop_constants = next_c
            
            # Ensure final size matches (sanity check)
            population = population[:self.pop_size]
            pop_constants = pop_constants[:self.pop_size]
                 
        if best_rpn is not None:
             formula = self.rpn_to_infix(best_rpn, best_consts_vec)
             # Inverse Transform if needed
             if GpuGlobals.USE_LOG_TRANSFORMATION:
                 # If we trained on log(y), the formula predicts log(y).
                 # To predict y, we need exp(formula).
                 formula = f"exp({formula})"
             return formula
        return None
