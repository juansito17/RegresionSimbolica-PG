
import torch
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
        # Device selection respects FORCE_CPU_MODE
        if device:
            self.device = device
        elif GpuGlobals.FORCE_CPU_MODE:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            max_patterns=GpuGlobals.PATTERN_MAX_PATTERNS,
            fitness_threshold=GpuGlobals.PATTERN_RECORD_FITNESS_THRESHOLD,
            min_uses=GpuGlobals.PATTERN_MEM_MIN_USES,
            dtype=self.dtype
        )
        
        # --- Persistent Cache for Initial Population ---
        self._cached_initial_pop = None
        self._cached_initial_consts = None
        self.mutation_bank = None
        
        # --- P0-6: Cache CudaRPNVM and arity tensors for evolve_generation_cuda ---
        self._cached_vm = None
        self._cached_token_arities = None
        self._cached_arity_0_ids = None
        self._cached_arity_1_ids = None
        self._cached_arity_2_ids = None

        # --- Pre-cached tensors for hot loop ---
        self._island_offsets = torch.arange(0, self.pop_size, self.island_size, device=self.device).view(self.n_islands, 1)
        self._selection_metric_buf = torch.empty(self.pop_size, device=self.device, dtype=self.dtype)
        
        # --- Variable diversity: cache variable token IDs for penalty computation ---
        self._var_token_ids = []
        if self.num_variables > 1:
            for vi in range(self.num_variables):
                vid = self.grammar.token_to_id.get(f'x{vi}', -1)
                if vid > 0:
                    self._var_token_ids.append(vid)
        self._var_diversity_buf = torch.empty(self.pop_size, device=self.device, dtype=self.dtype) if self._var_token_ids else None

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
        
    @staticmethod
    def _postprocess_formula(formula: str) -> str:
        """Post-process formula string for Python eval compatibility."""
        import re
        # Replace neg(expr) with (-(expr))
        # Handle nested neg() by iterating
        max_iters = 20
        for _ in range(max_iters):
            idx = formula.find('neg(')
            if idx == -1:
                break
            # Find matching closing paren
            depth = 0
            start = idx + 4  # after 'neg('
            end = start
            for i in range(start, len(formula)):
                if formula[i] == '(':
                    depth += 1
                elif formula[i] == ')':
                    if depth == 0:
                        end = i
                        break
                    depth -= 1
            inner = formula[start:end]
            formula = formula[:idx] + f'(-({inner}))' + formula[end+1:]
        # Replace ^ with ** for Python
        formula = formula.replace(' ^ ', '**').replace('^', '**')
        return formula

    def rpn_to_infix(self, rpn_tensor: torch.Tensor, constants: torch.Tensor = None) -> str:
        if not GpuGlobals.USE_SYMPY:
             # Basic conversion without SymPy cleanup
             raw = self.simplifier.rpn_to_infix_static(rpn_tensor, constants, self.grammar)
             return self._postprocess_formula(raw)
             
        res = self.simplifier._rpn_to_infix_str(rpn_tensor, constants)
        if res == "Invalid":
             # Fallback to basic
             raw = self.simplifier.rpn_to_infix_static(rpn_tensor, constants, self.grammar)
             return self._postprocess_formula(raw)
        return self._postprocess_formula(res)

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

    def _check_semantic_var_usage(self, rpn: torch.Tensor, consts: torch.Tensor, x_t: torch.Tensor) -> bool:
        """
        Check if all variables are semantically used (not dead code).
        Returns True only if changing each variable changes the formula's output.
        x_t: [N, Vars]
        """
        if self.num_variables <= 1:
            return True
        
        try:
            # Get original predictions — MUST clone since VM reuses internal buffer
            preds_orig = self.predict_individual(rpn, consts, x_t).clone()
            if preds_orig is None or torch.isnan(preds_orig).all():
                return False
            
            for vi in range(self.num_variables):
                x_mod = x_t.clone()
                # Set variable vi to a different value (shift by 1.0)
                x_mod[:, vi] = x_mod[:, vi] + 1.0
                preds_mod = self.predict_individual(rpn, consts, x_mod).clone()
                
                if preds_mod is None:
                    return False
                
                # Check if output changed meaningfully
                diff = (preds_orig - preds_mod).abs()
                # Filter out NaN/Inf
                valid = torch.isfinite(diff)
                if valid.any():
                    max_diff = diff[valid].max().item()
                    if max_diff < 1e-6:
                        return False  # Variable vi is semantically dead
                else:
                    return False  # All NaN/Inf
            
            return True  # All variables affect the output
        except Exception:
            return False

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

    def detect_patterns(self, targets) -> List[str]:
        """Detect simple patterns (linear, geometric, constant). Accepts torch.Tensor or array."""
        if isinstance(targets, torch.Tensor):
            t = targets.flatten().to(self.device)
        else:
            t = torch.tensor(targets, dtype=self.dtype, device=self.device).flatten()
        if t.shape[0] < 3: return []
        seeds = []
        diffs = t[1:] - t[:-1]
        if torch.allclose(diffs, diffs[0].expand_as(diffs), atol=1e-5):
            seeds.append("(C + (C * x0))") 
        if not torch.any(torch.abs(t) < 1e-9):
            ratios = t[1:] / t[:-1]
            if torch.allclose(ratios, ratios[0].expand_as(ratios), atol=1e-5):
                seeds.append("(C * (C ^ x0))")
        if torch.allclose(t, t[0].expand_as(t), atol=1e-5):
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
                 n_simulations=GpuGlobals.ALPHA_MCTS_N_SIMULATIONS,
                 max_depth=self.max_len,
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
        
        # 3. Convert to global indices (reuse cached tensor)
        best_global_idx = (best_local_idx + self._island_offsets).view(-1)
        worst_global_idx = (worst_local_idx + self._island_offsets).view(-1)
        
        # 4. Extract Migrants
        migrants_pop = population[best_global_idx].view(self.n_islands, mig_size, self.max_len)
        migrants_const = constants[best_global_idx].view(self.n_islands, mig_size, self.max_constants)
        
        # 5. Circular Shift (Migration Logic): Island i -> Island (i+1)%N
        # We roll the first dimension to the right
        shifted_migrants_pop = torch.roll(migrants_pop, shifts=1, dims=0).view(-1, self.max_len)
        shifted_migrants_const = torch.roll(migrants_const, shifts=1, dims=0).view(-1, self.max_constants)
        
        # 6. Apply in-place (no full clone needed - only ~200 individuals change)
        population[worst_global_idx] = shifted_migrants_pop
        constants[worst_global_idx] = shifted_migrants_const
        
        return population, constants

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
    
    def evolve_generation_cuda(self, population, constants, fitness, abs_errors, x_t, y_t, mutation_bank,
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
        
        # P0-6: Reuse cached VM and arity tensors
        if self._cached_vm is None:
            self._cached_vm = CudaRPNVM(self.grammar, self.device)
            self._cached_token_arities = self.grammar.get_arity_tensor(self.device)
            self._cached_arity_0_ids = self.grammar.get_arity_ids(0, self.device)
            self._cached_arity_1_ids = self.grammar.get_arity_ids(1, self.device)
            self._cached_arity_2_ids = self.grammar.get_arity_ids(2, self.device)
        
        vm = self._cached_vm
        token_arities = self._cached_token_arities
        arity_0_ids = self._cached_arity_0_ids
        arity_1_ids = self._cached_arity_1_ids
        arity_2_ids = self._cached_arity_2_ids
        
        # Ensure X is [Vars, N] for RPN kernel
        if x_t.ndim == 1:
            x_in = x_t.unsqueeze(0)  # [1, N]
        elif x_t.ndim == 2 and x_t.shape[1] != population.shape[0]:  # [N, Vars] -> [Vars, N]
            x_in = x_t.T.contiguous()
        else:
            x_in = x_t
        
        # Ensure y is [N]
        y_in = y_t.squeeze() if y_t.ndim > 1 else y_t
        
        if abs_errors is None:
            abs_errors = torch.empty(0, device=self.device)
        if mutation_bank is None:
            mutation_bank = torch.empty(0, device=self.device)

        result = rpn_cuda.evolve_generation(
            population,
            constants if constants.dtype == torch.float32 else constants.float(),
            fitness if fitness.dtype == torch.float32 else fitness.float(),
            abs_errors if abs_errors.dtype == torch.float32 else abs_errors.float(),
            x_in if x_in.dtype == torch.float32 else x_in.float(),
            y_in if y_in.dtype == torch.float32 else y_in.float(),
            token_arities,
            arity_0_ids,
            arity_1_ids,
            arity_2_ids,
            mutation_bank,
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
        
        # Shuffle Cases
        perm = torch.randperm(N_cases, device=self.device)
        candidates_err_shuffled = candidates_err[:, :, perm]
        
        # Elimination Loop (with early exit when all have 1 candidate)
        active_mask = torch.ones((n_parents, tour_size), dtype=torch.bool, device=self.device)
        
        for i in range(N_cases):
            # Optimization: If all rows have 1 candidate left, stop early
            active_counts = active_mask.sum(dim=1)
            if (active_counts <= 1).all():
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


    @staticmethod
    def _simplify_with_sympy(formula_str):
        """Try to simplify formula using SymPy for reduced complexity."""
        try:
            import sympy
            import re as re_mod
            from sympy import symbols, sin, cos, sqrt, exp, log, Abs, pi
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations
            # Use real=True to avoid re(), im(), Abs() from complex assumptions
            x0 = symbols('x0', real=True, positive=False)
            local_dict = {'x0': x0, 'pi': sympy.pi, 'e': sympy.E, 'abs': Abs, 'neg': lambda a: -a}
            expr = parse_expr(formula_str, local_dict=local_dict,
                             transformations=standard_transformations)
            simplified = sympy.simplify(expr)
            # Try to clean up floats close to integers
            simplified = sympy.nsimplify(simplified, tolerance=1e-3, rational=False)
            result = str(simplified)
            # Sanitize SymPy-specific syntax back to Python-evaluable
            result = result.replace('Abs(', 'abs(')
            result = re_mod.sub(r'\bre\(', '(', result)   # re(x0) → (x0)
            result = re_mod.sub(r'\bim\(', '(0*', result)  # im(x0) → (0*...)
            result = re_mod.sub(r'\bI\b', '0', result)     # imaginary unit → 0
            result = re_mod.sub(r'\bE\b', '2.718281828', result)  # Euler constant
            result = re_mod.sub(r'(?<!\w)oo(?!\w)', '1e30', result)  # infinity
            # Reject if has SymPy-specific functions we can't eval
            bad_tokens = ['zoo', 'nan', 'Symbol', 'Rational', 'Integer', 'Float',
                          'Piecewise', 'conjugate', 'Derivative', 'Integral']
            if any(tok in result for tok in bad_tokens):
                return formula_str
            # Ensure x0 is still present
            if 'x0' in formula_str and 'x0' not in result:
                return formula_str
            return result
        except Exception:
            return formula_str

    @staticmethod
    def _eval_formula_safe(formula_str, x_np):
        """Try to eval a formula string on numpy data. Returns y_pred or None."""
        import numpy as np
        try:
            safe_dict = {'x0': x_np, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp,
                         'log': np.log, 'sqrt': np.sqrt, 'abs': np.abs,
                         'pi': np.pi, 'e': np.e, 'neg': lambda a: -a}
            y = eval(formula_str, {"__builtins__": {}}, safe_dict)
            if isinstance(y, (int, float)):
                y = np.full_like(x_np, y)
            if isinstance(y, np.ndarray) and len(y) == len(x_np) and np.all(np.isfinite(y)):
                return y
        except Exception:
            pass
        return None

    def _post_simplify_formula(self, formula_str, x_t, y_t):
        """Simplify formula via Sniper re-detection and SymPy. Returns simplest valid version."""
        if not formula_str or formula_str == 'None':
            return formula_str
        
        import numpy as np
        x_np = x_t.cpu().numpy().flatten().astype(np.float64)
        y_np = y_t.cpu().numpy().flatten().astype(np.float64)
        
        best = formula_str
        best_len = len(formula_str)
        
        # First: verify original formula evaluates correctly
        y_orig_pred = self._eval_formula_safe(formula_str, x_np)
        if y_orig_pred is None:
            return formula_str  # Can't eval original, don't try to simplify
        
        rmse_orig = float(np.sqrt(np.mean((y_np - y_orig_pred) ** 2)))
        
        # Strategy 1: Re-run Sniper on the engine's predictions to find a cleaner pattern
        try:
            y_pred_t = torch.tensor(y_orig_pred, dtype=torch.float32, device=self.device)
            sniper_clean = self.sniper.run(x_t, y_pred_t)
            if sniper_clean and len(sniper_clean) < best_len:
                y_clean = self._eval_formula_safe(sniper_clean, x_np)
                if y_clean is not None:
                    rmse_clean = float(np.sqrt(np.mean((y_np - y_clean) ** 2)))
                    if rmse_clean <= rmse_orig * 1.1 + 1e-6:
                        best = sniper_clean
                        best_len = len(sniper_clean)
        except Exception:
            pass
        
        # Strategy 2: SymPy simplification (with eval validation)
        try:
            sympy_result = self._simplify_with_sympy(best)
            if sympy_result and sympy_result != best and len(sympy_result) < best_len:
                y_sympy = self._eval_formula_safe(sympy_result, x_np)
                if y_sympy is not None:
                    rmse_sympy = float(np.sqrt(np.mean((y_np - y_sympy) ** 2)))
                    if rmse_sympy <= rmse_orig * 1.1 + 1e-6:
                        best = sympy_result
                        best_len = len(sympy_result)
        except Exception:
            pass
        
        return best

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
        sniper_formula = None
        if GpuGlobals.USE_SNIPER:
            # Pass GPU tensors directly - Sniper already works with torch internally
            sniper_formula = self.sniper.run(x_t, y_t)
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
        
        # Hash operators to avoid loading incompatible cache
        import hashlib
        ops_sig = "_".join(sorted(self.grammar.operators))
        ops_hash = hashlib.md5(ops_sig.encode()).hexdigest()[:8]
        
        cache_file = os.path.join(cache_dir, f"initial_pop_v3_{self.pop_size}_{self.max_len}_{self.num_variables}_{prec_str}_{ops_hash}.pt")
        
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
             # Use uniform distribution within configured range instead of rando normal
             temp_c = torch.empty(self.pop_size, self.max_constants, device=self.device, dtype=self.dtype).uniform_(GpuGlobals.CONSTANT_MIN_VALUE, GpuGlobals.CONSTANT_MAX_VALUE)
             
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
                # Use NARROW constant range for seeds to avoid overflow in pow operations
                # Seeds with explicit numeric values already have their constants set by load_population_from_strings
                # Only randomize constants that are zero (i.e., placeholder 'C' tokens without explicit values)
                existing_consts = seed_consts[:n].to(self.dtype)
                # For each seed, randomize constants that are exactly 0.0 (placeholders)
                # Keep constants that were explicitly set from numeric values in the formula string
                placeholder_mask = (existing_consts == 0.0)
                pop_constants[:n] = existing_consts
                # Narrow init: [-2, 2] avoids overflow when used as exponents
                narrow_rand = torch.empty_like(pop_constants[:n]).uniform_(-2.0, 2.0)
                pop_constants[:n] = torch.where(placeholder_mask, narrow_rand, existing_consts)
                
                # PSO Warmup: optimize seed constants BEFORE main loop
                if GpuGlobals.USE_NANO_PSO and n > 0:
                    warmup_steps = 80  # Thorough warmup — critical for seed quality
                    batch_sz = min(n, 5000)
                    for bi in range(0, n, batch_sz):
                        be = min(bi + batch_sz, n)
                        ref_c, ref_f = self.optimizer.nano_pso(
                            population[bi:be], pop_constants[bi:be], x_t, y_t, steps=warmup_steps)
                        pop_constants[bi:be] = ref_c
                    # Report best seed after warmup
                    seed_fit = self.evaluator.evaluate_batch(population[:n], x_t, y_t, pop_constants[:n])
                    best_seed_rmse = seed_fit.min().item()
                    print(f"[Engine] PSO warmup on {n} seeds ({warmup_steps} steps): best RMSE = {best_seed_rmse:.6f}")
                    
                    # Early exit for high-confidence clean Sniper seeds
                    # Prevents overfitting to noisy data when true signal is clean
                    if sniper_formula and best_seed_rmse < 1.0:
                        import re
                        y_var = torch.var(y_t).item()
                        if y_var > 1e-12:
                            seed_r2 = 1.0 - (best_seed_rmse**2 / y_var)
                            has_long_decimal = bool(re.search(r'\d+\.\d{3,}', sniper_formula))
                            # Use lower R² threshold for very clean/short formulas (noise tolerance)
                            r2_threshold = 0.995 if len(sniper_formula) < 25 else 0.999
                            if seed_r2 > r2_threshold and not has_long_decimal and len(sniper_formula) < 60:
                                print(f"\n[Engine] Exact solution found! RMSE: {best_seed_rmse:.9e}")
                                return sniper_formula
            else:
                print("[DEBUG] CRITICAL: Seed loading returned None!")
                
        # Patterns (GPU-native, no CPU transfer)
        pats = self.detect_patterns(y_t)
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
        global_stagnation = 0  # Solo se resetea con mejora real, no con cataclismo
        generations = 0
        current_mutation_rate = GpuGlobals.BASE_MUTATION_RATE
        COMPLEXITY_PENALTY = GpuGlobals.COMPLEXITY_PENALTY
        
        # --- P0-1 Optimization: Reuse fitness from C++ orchestrator ---
        cached_next_fit = None       # Fitness returned by evolve_generation_cuda
        modified_indices = None      # Indices modified after evolution (migration/inject)

        start_time = time.time()
        while True:
            # Timeout
            elapsed = time.time() - start_time
            if timeout_sec and elapsed >= timeout_sec: break
            if generations >= GpuGlobals.GENERATIONS: break
            
            # Time-based early exit: good-enough solution before full timeout
            if best_rmse < 0.005 and elapsed > 10:
                print(f"\n[Engine] Good-enough solution (RMSE={best_rmse:.6f}, {elapsed:.1f}s). Early exit.")
                break
            
            generations += 1

            # Migration (track modified indices for fitness cache invalidation)
            all_modified = False  # Flag instead of 1M-element tensor
            modified_indices = None
            # Reduce migration during stagnation to preserve island diversity
            mig_interval = GpuGlobals.MIGRATION_INTERVAL_STAGNATION if stagnation > GpuGlobals.MIGRATION_STAGNATION_THRESHOLD else GpuGlobals.MIGRATION_INTERVAL
            if self.n_islands > 1 and generations % mig_interval == 0:
                 population, pop_constants = self.migrate_islands(population, pop_constants, 
                     fitness_rmse if cached_next_fit is None else cached_next_fit)
                 all_modified = True
            
            # Deduplication — Remove clones to maintain diversity
            if generations % GpuGlobals.DEDUPLICATION_INTERVAL == 0:
                population, pop_constants, n_dups = self.operators.deduplicate_population(population, pop_constants)
                if n_dups > 0:
                    all_modified = True

            # --- Neural Flash (Intelligence Injection) ---
            if GpuGlobals.USE_NEURAL_FLASH and self.model is not None and generations % GpuGlobals.NEURAL_FLASH_INTERVAL == 0:
                 pop_neural, const_neural = self.neural_flash_injection(x_t, y_t)
                 if pop_neural is not None:
                     n_inj = pop_neural.shape[0]
                     if n_inj > 0:
                         limit = min(n_inj, int(self.pop_size * GpuGlobals.NEURAL_FLASH_INJECT_PERCENT))
                         indices = torch.randint(0, self.pop_size, (limit,), device=self.device)
                         population[indices] = pop_neural[:limit]
                         pop_constants[indices] = const_neural[:limit]

            # --- Alpha MCTS (Deep Thought) ---
            if GpuGlobals.USE_ALPHA_MCTS and self.model is not None and generations % GpuGlobals.ALPHA_MCTS_INTERVAL == 0:
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

            # --- Pattern Memory Injection (in-place, P0-5) ---
            # Inject successful building blocks to accelerate convergence
            if GpuGlobals.USE_PATTERN_MEMORY and generations > 10 and generations % GpuGlobals.PATTERN_INJECT_INTERVAL == 0:
                 inject_positions = self.pattern_memory.inject_into_population_inplace(
                     population, pop_constants, self.grammar, 
                     percent=GpuGlobals.PATTERN_INJECT_PERCENT
                 )
                 if inject_positions is not None:
                     # Mark injected positions for re-evaluation
                     if not all_modified:
                         if modified_indices is None:
                             modified_indices = inject_positions
                         else:
                             modified_indices = torch.cat([modified_indices, inject_positions])

            # Eval — P0-1: Reuse cached fitness from C++ orchestrator when possible
            if cached_next_fit is not None and not GpuGlobals.USE_LEXICASE_SELECTION and not all_modified and modified_indices is None:
                # Fast path: reuse C++ fitness, no evaluation needed
                fitness_rmse = cached_next_fit
                abs_errors = None
            elif cached_next_fit is not None and not GpuGlobals.USE_LEXICASE_SELECTION and (all_modified or modified_indices is not None):
                # Partial or full re-eval based on modifications
                fitness_rmse = cached_next_fit
                abs_errors = None
                if all_modified:
                    fitness_rmse = self.evaluator.evaluate_batch(population, x_t, y_t, pop_constants)
                else:
                    unique_mod = modified_indices.unique()
                    partial_rmse = self.evaluator.evaluate_batch(
                        population[unique_mod], x_t, y_t, pop_constants[unique_mod])
                    fitness_rmse[unique_mod] = partial_rmse
            elif GpuGlobals.USE_LEXICASE_SELECTION:
                abs_errors = self.evaluator.evaluate_batch_full(population, x_t, y_t, pop_constants)
                fitness_rmse = torch.mean(abs_errors**2, dim=1).sqrt() # Approx RMSE for stats
            else:
                fitness_rmse = self.evaluator.evaluate_batch(population, x_t, y_t, pop_constants)
                abs_errors = None
            cached_next_fit = None  # Consumed
            
            # --- Weighted Fitness (opcional: pondera errores por dificultad) ---
            if GpuGlobals.USE_WEIGHTED_FITNESS and abs_errors is not None:
                # Ponderar cada caso por su dificultad media (errores más altos pesan más)
                case_difficulty = abs_errors.mean(dim=0)  # [D]
                weights = (case_difficulty + 1e-9) ** GpuGlobals.WEIGHTED_FITNESS_EXPONENT
                weights = weights / weights.sum() * abs_errors.shape[1]  # Normalizar
                weighted_sq = (abs_errors ** 2) * weights.unsqueeze(0)
                fitness_rmse = torch.mean(weighted_sq, dim=1).sqrt()
                
            # --- Pareto Front Update (liviano: solo top-200) ---
            if GpuGlobals.USE_PARETO_SELECTION and generations % GpuGlobals.PARETO_INTERVAL == 0:
                k_pareto = min(self.pop_size, 200)
                _, top_pareto_idx = torch.topk(fitness_rmse, k_pareto, largest=False)
                pareto_fit = fitness_rmse[top_pareto_idx]
                pareto_len = (population[top_pareto_idx] != PAD_ID).sum(dim=1).float()
                try:
                    front_local = self.pareto.get_pareto_front(pareto_fit, pareto_len)
                    if front_local:
                        # Mapear índices locales a globales
                        self._pareto_front_global = top_pareto_idx[front_local].tolist()
                        # Inyectar miembros del frente como élites en islas aleatorias
                        for fi, gi in enumerate(self._pareto_front_global[:self.n_islands]):
                            elite_slot = fi * self.island_size  # Pos 0 de cada isla
                            if population is self.pop_buffer_A:
                                self.pop_buffer_A[elite_slot] = population[gi]
                                self.const_buffer_A[elite_slot] = pop_constants[gi]
                            else:
                                self.pop_buffer_B[elite_slot] = population[gi]
                                self.const_buffer_B[elite_slot] = pop_constants[gi]
                except Exception:
                    pass  # Non-fatal
                
            # --- Pattern Memory Recording ---
            # Record successful subtrees
            if GpuGlobals.USE_PATTERN_MEMORY and generations % GpuGlobals.PATTERN_RECORD_INTERVAL == 0:
                 self.pattern_memory.record_subtrees(
                     population, fitness_rmse, self.grammar, 
                     min_size=GpuGlobals.PATTERN_MIN_SIZE, max_size=GpuGlobals.PATTERN_MAX_SIZE
                 )
            
            # Optimize Top K — focus PSO on multi-variable formulas
            if GpuGlobals.USE_NANO_PSO and generations % GpuGlobals.PSO_INTERVAL == 0:
                # During stagnation: optimize more individuals with more steps
                if stagnation > GpuGlobals.PSO_STAGNATION_THRESHOLD:
                    k_opt = min(self.pop_size, GpuGlobals.PSO_K_STAGNATION)
                    pso_steps = GpuGlobals.PSO_STEPS_STAGNATION
                else:
                    k_opt = min(self.pop_size, GpuGlobals.PSO_K_NORMAL)
                    pso_steps = GpuGlobals.PSO_STEPS_NORMAL
                
                # Use penalized metric to select top-K for PSO (favor multi-variable)
                if self._var_token_ids and GpuGlobals.VAR_DIVERSITY_PENALTY > 0:
                    _pso_metric = fitness_rmse.clone()
                    _n_miss = torch.zeros(self.pop_size, device=self.device, dtype=self.dtype)
                    _n_miss.fill_(float(len(self._var_token_ids)))
                    for vid in self._var_token_ids:
                        _n_miss.sub_((population == vid).any(dim=1).float())
                    _pso_metric.add_(_n_miss, alpha=GpuGlobals.VAR_DIVERSITY_PENALTY)
                    _, top_idx = torch.topk(_pso_metric, k_opt, largest=False)
                else:
                    _, top_idx = torch.topk(fitness_rmse, k_opt, largest=False)
                
                opt_pop = population[top_idx]
                opt_consts = pop_constants[top_idx]
                
                refined_consts, refined_mse = self.optimizer.nano_pso(opt_pop, opt_consts, x_t, y_t, steps=pso_steps)
                # Forzar constantes enteras si está configurado
                if GpuGlobals.FORCE_INTEGER_CONSTANTS:
                    refined_consts = refined_consts.round()
                pop_constants[top_idx] = refined_consts
                fitness_rmse[top_idx] = refined_mse
            
            # Best Tracking — only consider formulas using ALL variables (if multi-var mode)
            if self._var_token_ids and GpuGlobals.VAR_DIVERSITY_PENALTY > 0:
                # Compute mask: which individuals use all variables
                _uses_all = torch.ones(self.pop_size, dtype=torch.bool, device=self.device)
                for vid in self._var_token_ids:
                    _uses_all &= (population == vid).any(dim=1)
                if _uses_all.any():
                    _masked_rmse = fitness_rmse.clone()
                    _masked_rmse[~_uses_all] = float('inf')
                    min_rmse, min_idx = torch.min(_masked_rmse, dim=0)
                    min_rmse_val = min_rmse.item()
                else:
                    # No formula uses all vars yet — fall back to raw
                    min_rmse, min_idx = torch.min(fitness_rmse, dim=0)
                    min_rmse_val = min_rmse.item()
            else:
                min_rmse, min_idx = torch.min(fitness_rmse, dim=0)
                min_rmse_val = min_rmse.item()  # Single sync
            
            if min_rmse_val == min_rmse_val and min_rmse_val < (best_rmse - GpuGlobals.FITNESS_EQUALITY_TOLERANCE):  # NaN check + tolerance
                min_idx_val = min_idx.item()
                candidate_rpn = population[min_idx_val]
                candidate_consts = pop_constants[min_idx_val]
                
                # Semantic variable check — reject formulas with dead variables
                if self._var_token_ids and GpuGlobals.VAR_DIVERSITY_PENALTY > 0:
                    if not self._check_semantic_var_usage(candidate_rpn, candidate_consts, x_t):
                        # Mark this individual as "dead" so it loses selection
                        fitness_rmse[min_idx_val] = float('inf')
                        min_rmse_val = float('inf')  # Skip acceptance
                
                if min_rmse_val < (best_rmse - GpuGlobals.FITNESS_EQUALITY_TOLERANCE):
                    best_rmse = min_rmse_val
                    best_rpn = candidate_rpn.clone()
                    best_consts_vec = candidate_consts.clone()
                    self.best_global_rmse = best_rmse
                    self.best_global_rpn = best_rpn
                    self.best_global_consts = best_consts_vec
                
                # Identify Island
                island_idx = (min_idx_val // self.island_size) if self.n_islands > 1 else 0

                stagnation = 0
                global_stagnation = 0  # Mejora real resetea ambos
                
                # Check for exact solution
                if best_rmse < GpuGlobals.EXACT_SOLUTION_THRESHOLD:
                    print(f"\n[Engine] Exact solution found! RMSE: {best_rmse:.9e}")
                    if callback:
                        callback(generations, best_rmse, best_rpn, best_consts_vec, True, island_idx)
                    self.stop_flag = True
                    
                    # Convert to string to match expected return type
                    formula = self.rpn_to_infix(best_rpn, best_consts_vec)
                    # Handle Log Transform Inverse if needed
                    if GpuGlobals.USE_LOG_TRANSFORMATION:
                        formula = f"exp({formula})"
                    
                    # Post-simplify for reduced complexity
                    formula = self._post_simplify_formula(formula, x_t, y_t)
                    return formula
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
                global_stagnation += 1
                # print(f"[DEBUG] Gen {generations}: Stagnation {stagnation} (Best {best_rmse:.6f}, Current Min {min_rmse.item():.6f})")
                
            if callback and generations % GpuGlobals.PROGRESS_REPORT_INTERVAL == 0:
                callback(generations, best_rmse, best_rpn, best_consts_vec, False, -1)

            # Cataclysm / Reset — use penalized metric to preserve multi-variable formulas
            if GpuGlobals.USE_ISLAND_CATACLYSM and stagnation >= GpuGlobals.STAGNATION_LIMIT:
                 n_elites = int(self.pop_size * GpuGlobals.CATACLYSM_ELITE_PERCENT)
                 n_random = self.pop_size - n_elites
                 # Compute cataclysm metric: prefer multi-variable formulas
                 cat_metric = fitness_rmse.clone()
                 if self._var_token_ids and GpuGlobals.VAR_DIVERSITY_PENALTY > 0:
                     _n_missing = torch.zeros(self.pop_size, device=self.device, dtype=self.dtype)
                     _n_missing.fill_(float(len(self._var_token_ids)))
                     for vid in self._var_token_ids:
                         _n_missing.sub_((population == vid).any(dim=1).float())
                     cat_metric.add_(_n_missing, alpha=GpuGlobals.VAR_DIVERSITY_PENALTY)
                 _, sorted_idx = torch.topk(cat_metric, n_elites, largest=False)
                 elites = population[sorted_idx]
                 elite_c = pop_constants[sorted_idx]
                 
                 new_pop = self.operators.generate_random_population(n_random)
                 new_c = torch.empty(n_random, self.max_constants, device=self.device, dtype=self.dtype).uniform_(GpuGlobals.CONSTANT_MIN_VALUE, GpuGlobals.CONSTANT_MAX_VALUE)
                 
                 # Mutate half the elites to add structural variety near the best
                 n_mutate_elites = n_elites // 2
                 if n_mutate_elites > 0:
                     mutated = self.operators.subtree_mutation(elites[n_elites//4:n_elites//4+n_mutate_elites].clone(), 1.0)
                     elites[n_elites//4:n_elites//4+n_mutate_elites] = mutated
                 
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

            # Global Stagnation — reinicio total si ni con cataclismos mejora
            if global_stagnation >= GpuGlobals.GLOBAL_STAGNATION_LIMIT:
                print(f"[Engine] Global stagnation ({GpuGlobals.GLOBAL_STAGNATION_LIMIT} gens). Full restart around best.")
                new_pop = self.operators.generate_random_population(self.pop_size)
                new_c = torch.empty(self.pop_size, self.max_constants, device=self.device, dtype=self.dtype).uniform_(GpuGlobals.CONSTANT_MIN_VALUE, GpuGlobals.CONSTANT_MAX_VALUE)
                # Write to current buffer
                if population is self.pop_buffer_A:
                    self.pop_buffer_A[:] = new_pop
                    self.const_buffer_A[:] = new_c
                    if best_rpn is not None:
                        self.pop_buffer_A[0] = best_rpn
                        self.const_buffer_A[0] = best_consts_vec
                    population = self.pop_buffer_A
                    pop_constants = self.const_buffer_A
                else:
                    self.pop_buffer_B[:] = new_pop
                    self.const_buffer_B[:] = new_c
                    if best_rpn is not None:
                        self.pop_buffer_B[0] = best_rpn
                        self.const_buffer_B[0] = best_consts_vec
                    population = self.pop_buffer_B
                    pop_constants = self.const_buffer_B
                    
                # Re-inject seeds with fresh random constants on restart  
                if seeds:
                    seed_pop_r, seed_consts_r = self.load_population_from_strings(seeds)
                    if seed_pop_r is not None:
                        nr = min(seed_pop_r.shape[0], self.pop_size - 1)  # Leave slot 0 for best
                        population[1:1+nr] = seed_pop_r[:nr]
                        # Narrow random init for seeds
                        existing_c = seed_consts_r[:nr].to(self.dtype)
                        p_mask = (existing_c == 0.0)
                        narrow = torch.empty(nr, self.max_constants, device=self.device, dtype=self.dtype).uniform_(-2.0, 2.0)
                        pop_constants[1:1+nr] = torch.where(p_mask, narrow, existing_c)
                        # Quick PSO on re-injected seeds
                        if GpuGlobals.USE_NANO_PSO:
                            ref_c2, _ = self.optimizer.nano_pso(
                                population[1:1+nr], pop_constants[1:1+nr], x_t, y_t, steps=30)
                            pop_constants[1:1+nr] = ref_c2
                stagnation = 0
                global_stagnation = 0
                continue

            # Dynamic Mutation — Ramp up during stagnation
            if stagnation > GpuGlobals.MUTATION_STAGNATION_TRIGGER:
                current_mutation_rate = min(GpuGlobals.MUTATION_RATE_CAP, GpuGlobals.BASE_MUTATION_RATE + (stagnation - GpuGlobals.MUTATION_STAGNATION_TRIGGER) * GpuGlobals.MUTATION_RAMP_PER_GEN)
                
                # --- Residual Boosting (Every 20 stagnation steps) ---
                if GpuGlobals.USE_RESIDUAL_BOOSTING and stagnation % GpuGlobals.RESIDUAL_BOOST_INTERVAL == 0 and best_rpn is not None:
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
                
                # --- Stagnation Random Injection ---
                # Inyectar individuos aleatorios durante estancamiento para mantener diversidad
                if GpuGlobals.STAGNATION_RANDOM_INJECT_PERCENT > 0 and stagnation % 10 == 0:
                    n_inject = int(self.pop_size * GpuGlobals.STAGNATION_RANDOM_INJECT_PERCENT)
                    if n_inject > 0:
                        inject_pop = self.operators.generate_random_population(n_inject)
                        inject_c = torch.empty(n_inject, self.max_constants, device=self.device, dtype=self.dtype).uniform_(GpuGlobals.CONSTANT_MIN_VALUE, GpuGlobals.CONSTANT_MAX_VALUE)
                        # Reemplazar los peores individuos
                        _, worst_idx = torch.topk(fitness_rmse, n_inject, largest=True)
                        population[worst_idx] = inject_pop
                        pop_constants[worst_idx] = inject_c
                         
            else:
                current_mutation_rate = GpuGlobals.BASE_MUTATION_RATE
                
            # Evolution Step (Vectorized Island Model + Double Buffering)
            island_offsets = self._island_offsets

            # Determine Buffers based on generation parity
            # If gen % 2 == 0: Read A, Write B (Logic: Gen started with A, so Next is B)
            # Logic Check:
            # Gen 1 (Odd): Start in A. Write to B.
            # End of loop: population = B.
            # Gen 2 (Even): Start in B. Write to A.
            
            # --- EVOLUTION STEP (Reproduction) ---
            if GpuGlobals.USE_CUDA_ORCHESTRATOR:
                # 1. Selection Metric with Complexity Penalty (in-place to reduce allocations)
                lengths = (population != PAD_ID).sum(dim=1).float()
                # selection_metric = fitness_rmse * (1.0 + penalty * lengths) + lengths * 1e-6
                selection_metric = self._selection_metric_buf
                torch.mul(lengths, COMPLEXITY_PENALTY, out=selection_metric)
                selection_metric.add_(1.0)
                selection_metric.mul_(fitness_rmse)
                selection_metric.add_(lengths, alpha=1e-6)
                
                # 1b. Variable Diversity Penalty (ADDITIVE): penalize formulas missing variables
                if self._var_token_ids and GpuGlobals.VAR_DIVERSITY_PENALTY > 0:
                    var_pen = self._var_diversity_buf
                    var_pen.fill_(float(len(self._var_token_ids)))  # start at num_variables
                    for vid in self._var_token_ids:
                        var_pen.sub_((population == vid).any(dim=1).float())  # subtract 1 per var found
                    # var_pen is now n_missing_vars (0..num_vars)
                    # Additive: selection_metric += PENALTY * n_missing
                    selection_metric.add_(var_pen, alpha=GpuGlobals.VAR_DIVERSITY_PENALTY)
                
                # Adaptive tournament size: reduce during stagnation to allow more diversity
                effective_tournament = max(GpuGlobals.TOURNAMENT_SIZE_FLOOR, GpuGlobals.DEFAULT_TOURNAMENT_SIZE - stagnation // GpuGlobals.TOURNAMENT_ADAPTIVE_DIVISOR)
                
                # 2. Call Full C++ Orchestrator (Selection + Crossover + Mutation)
                # Refresh Mutation Bank periodically for structural innovation
                if self.mutation_bank is None or generations % GpuGlobals.MUTATION_BANK_REFRESH_INTERVAL == 0:
                    self.mutation_bank = self.operators.generate_random_population(GpuGlobals.MUTATION_BANK_SIZE)

                next_pop, next_c, next_fit = self.evolve_generation_cuda(
                    population, pop_constants, selection_metric, abs_errors, x_t, y_t, self.mutation_bank,
                    mutation_rate=current_mutation_rate,
                    crossover_rate=GpuGlobals.DEFAULT_CROSSOVER_RATE,
                    tournament_size=effective_tournament,
                    pso_steps=0, # Disable global PSO to avoid OOM
                    pso_particles=GpuGlobals.PSO_PARTICLES
                )
                
                if next_pop is not None:
                    # Update local refs for next generation
                    next_pop = next_pop[:self.pop_size]
                    next_c = next_c[:self.pop_size]
                    # P0-1: Cache the fitness from C++ for reuse in next iteration
                    cached_next_fit = next_fit[:self.pop_size] if next_fit is not None else None
                    # Global Elite Injection: preserve best formula in position 0
                    if best_rpn is not None:
                        next_pop[0] = best_rpn
                        next_c[0] = best_consts_vec
                        if cached_next_fit is not None:
                            cached_next_fit[0] = best_rmse
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
                # In-place selection metric computation (reuse buffer)
                selection_metric = self._selection_metric_buf
                torch.mul(lengths, COMPLEXITY_PENALTY, out=selection_metric)
                selection_metric.add_(1.0)
                selection_metric.mul_(fitness_rmse)
                selection_metric.add_(lengths, alpha=1e-6)
                
                # Variable Diversity Penalty (legacy path - ADDITIVE)
                if self._var_token_ids and GpuGlobals.VAR_DIVERSITY_PENALTY > 0:
                    var_pen = self._var_diversity_buf
                    var_pen.fill_(float(len(self._var_token_ids)))
                    for vid in self._var_token_ids:
                        var_pen.sub_((population == vid).any(dim=1).float())
                    selection_metric.add_(var_pen, alpha=GpuGlobals.VAR_DIVERSITY_PENALTY)
                
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
            if GpuGlobals.USE_SIMPLIFICATION and generations % GpuGlobals.SIMPLIFICATION_INTERVAL == 0:
                 # Simplify only the BEST formulas in each island to save time
                 k_simplify = GpuGlobals.K_SIMPLIFY
                 
                 # DEBUG
                 # if generations % 100 == 0:
                 #     print(f"[Debug] Gen {generations}: pop={self.pop_size}, islands={self.n_islands}, size={self.island_size}")
                 
                 # Which fitness to use? next_fit is fresh for the next population
                 # If using C++ Orchestrator, next_fit is available.
                 # If using Python loop, selection_metric is available but it's for OLD pop.
                 # For simplification of NEW offspring, ideally we evaluate them first, 
                 # but to save time we can use the parent's fitness as a proxy, 
                 # OR better, if we have next_fit from C++, use it!
                 
                 sim_fitness = next_fit if (GpuGlobals.USE_CUDA_ORCHESTRATOR and next_fit is not None) else selection_metric
                 
                 # Ensure sim_fitness size matches next_pop
                 if sim_fitness.shape[0] != next_pop.shape[0]:
                      # Fallback to selection_metric if sizes mismatch, but clamp below
                      sim_fitness = selection_metric[:next_pop.shape[0]]

                 # Reshape fitness to islands
                 try:
                     island_fitness = sim_fitness[:self.pop_size].view(self.n_islands, self.island_size)
                     
                     # Find top K indices within each island
                     _, top_local_idx = torch.topk(island_fitness, k_simplify, dim=1, largest=False)
                     
                     # Convert to global indices
                     offsets = island_offsets.view(self.n_islands, 1)
                     top_global_idx = (top_local_idx + offsets).view(-1)
                     
                     # DEFENSIVE CLAMP: ensure no index exceeds current population size
                     max_idx = next_pop.shape[0] - 1
                     if (top_global_idx > max_idx).any() or (top_global_idx < 0).any():
                         top_global_idx = top_global_idx.clamp(0, max_idx)
                     
                     # Simplify only these individuals
                     sub_pop = next_pop[top_global_idx]
                     sub_const = next_c[top_global_idx]

                     # Pass to optimized symbolic simplifier (vectorized)
                     sim_pop, _, n_s = self.gpu_simplifier.simplify_batch(sub_pop, sub_const)
                     
                     # Write back
                     next_pop[top_global_idx] = sim_pop
                 except Exception as e:
                     print(f"WARN: Simplification failed (Gen {generations}): {e}")
                     # Non-fatal, just skip simplification for this batch
                 # next_c[top_global_idx] = ... (constants preserved if not PADed)
            
            # Swap Buffers
            population = next_pop
            pop_constants = next_c
            
            # Ensure final size matches (sanity check)
            population = population[:self.pop_size]
            pop_constants = pop_constants[:self.pop_size]
                 
        if best_rpn is not None:
             # --- Final Simplification Pass ---
             try:
                 sim_pop, _, n_s = self.gpu_simplifier.simplify_batch(
                     best_rpn.unsqueeze(0), best_consts_vec.unsqueeze(0), max_passes=10)
                 if n_s > 0:
                     best_rpn = sim_pop[0]
             except Exception:
                 pass  # Non-fatal
             
             formula = self.rpn_to_infix(best_rpn, best_consts_vec)
             # Inverse Transform if needed
             if GpuGlobals.USE_LOG_TRANSFORMATION:
                 formula = f"exp({formula})"
             
             # Post-simplify for reduced complexity
             formula = self._post_simplify_formula(formula, x_t, y_t)
             return formula
        return None
