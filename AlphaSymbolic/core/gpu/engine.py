
import torch
import numpy as np
import time
from typing import List, Tuple, Optional
from core.grammar import OPERATORS, VARIABLES, CONSTANTS, ExpressionTree
from .formatting import format_const
from .sniper import Sniper
from .config import GpuGlobals
from .pareto import ParetoOptimizer
from .pattern_memory import PatternMemory

# SymPy for simplification
try:
    import sympy
    from sympy import symbols, sympify, simplify, nsimplify, Float
    from sympy.parsing.sympy_parser import parse_expr
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

# --- GPU GRAMMAR ENCODING (RPN / Postfix) ---
PAD_ID = 0

class GPUGrammar:
    def __init__(self, num_variables=1):
        self.token_to_id = {'<PAD>': PAD_ID}
        self.id_to_token = {PAD_ID: '<PAD>'}
        self.next_id = 1
        
        # Terminals (Variables + Constants)
        self.active_variables = ['x0'] # Always support x0
        if num_variables > 1:
            self.active_variables = [f'x{i}' for i in range(num_variables)]
        elif num_variables == 1:
            self.active_variables = ['x', 'x0'] 

        self.terminals = self.active_variables + ['C', '1', '2', '3', '5'] # Removed pi, e to avoid collision
        for t in self.terminals:
            self.token_to_id[t] = self.next_id
            self.id_to_token[self.next_id] = t
            self.next_id += 1
            
        # Operators
        self.operators = []
        if GpuGlobals.USE_OP_PLUS:  self.operators.append('+')
        if GpuGlobals.USE_OP_MINUS: self.operators.append('-')
        if GpuGlobals.USE_OP_MULT:  self.operators.append('*')
        if GpuGlobals.USE_OP_DIV:   self.operators.append('/')
        if GpuGlobals.USE_OP_POW:   self.operators.append('pow')
        if GpuGlobals.USE_OP_MOD:   self.operators.append('%')
        if GpuGlobals.USE_OP_SIN:   self.operators.append('sin')
        if GpuGlobals.USE_OP_COS:   self.operators.append('cos')
        if GpuGlobals.USE_OP_LOG:   self.operators.append('log')
        if GpuGlobals.USE_OP_EXP:   self.operators.append('e')
        if GpuGlobals.USE_OP_FACT:  self.operators.append('!') # tgamma
        # if GpuGlobals.USE_OP_FLOOR: self.operators.append('_') # Not mapped in default?
        if GpuGlobals.USE_OP_GAMMA: self.operators.append('g')
        if GpuGlobals.USE_OP_ASIN:  self.operators.append('S')
        if GpuGlobals.USE_OP_ACOS:  self.operators.append('C')
        if GpuGlobals.USE_OP_ATAN:  self.operators.append('T')
        
        # Always active standard ops? Or add globals for them?
        # Assuming these are always available or tracked by globals?
        # Globals.h doesn't seem to have toggles for sqrt/abs/neg explicitly in the list I saw?
        # Wait, I saw USE_OP_SIN, etc.
        # I'll add them unconditionally for now or check globals?
        # Globals.h doesn't list sqrt/abs/neg toggles. So they are likely always on or implicit.
        self.operators.append('sqrt')
        self.operators.append('abs')
        self.operators.append('neg')
        self.operators.append('_') # Floor, adding it back since I saw it in C++ kernel logic!

        for op in self.operators:
            self.token_to_id[op] = self.next_id
            self.id_to_token[self.next_id] = op
            self.next_id += 1
            
        self.vocab_size = self.next_id
        
        self.op_ids = {op: self.token_to_id[op] for op in self.operators}
        self.token_arity = {}
        for op in self.operators:
            tid = self.token_to_id[op]
            self.token_arity[op] = OPERATORS[op] 
            
    def get_subtree_span(self, rpn_ids: List[int], root_idx: int) -> Tuple[int, int]:
        """
        Finds the span (start_idx, end_idx) of the subtree rooted at root_idx in RPN.
        Scanning backwards from root_idx.
        Returns indices inclusive [start, end].
        """
        if root_idx < 0 or root_idx >= len(rpn_ids): return (-1, -1)
        
        # Get Arity of root
        root_id = rpn_ids[root_idx]
        if root_id == PAD_ID: return (root_idx, root_idx)
        
        token = self.id_to_token.get(root_id, "")
        required_args = self.token_arity.get(token, 0)
        
        current_idx = root_idx - 1
        for _ in range(required_args):
            start, _ = self.get_subtree_span(rpn_ids, current_idx)
            if start == -1: return (-1, -1) # Error
            current_idx = start - 1
            
        return (current_idx + 1, root_idx)

class TensorGeneticEngine:
    def __init__(self, device=None, pop_size=None, max_len=30, num_variables=1, max_constants=5, n_islands=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Defaults from Globals
        if pop_size is None: pop_size = GpuGlobals.POP_SIZE
        if n_islands is None: n_islands = GpuGlobals.NUM_ISLANDS
        
        self.grammar = GPUGrammar(num_variables)
        
        self.n_islands = n_islands
        if pop_size % n_islands != 0:
            pop_size = (pop_size // n_islands) * n_islands
            
        self.pop_size = pop_size
        self.island_size = pop_size // n_islands
        self.max_len = max_len
        self.num_variables = num_variables
        self.max_constants = max_constants
        
        # Pre-allocate memory for random generation
        self.terminal_ids = torch.tensor([self.grammar.token_to_id[t] for t in self.grammar.terminals], device=self.device)
        self.operator_ids = torch.tensor([self.grammar.token_to_id[op] for op in self.grammar.operators], device=self.device)
        
        # --- Pre-compute Arity Masks for Safe Mutation ---
        self.token_arity = torch.zeros(self.grammar.vocab_size + 1, dtype=torch.long, device=self.device)
        self.arity_0_ids = []
        self.arity_1_ids = []
        self.arity_2_ids = []
        
        # Terminals (0)
        # Note: self.grammar.terminals is fixed in grammar class currently.
        # Ideally Grammar should also take GpuGlobals into account.
        for t in self.grammar.terminals:
            tid = self.grammar.token_to_id[t]
            self.token_arity[tid] = 0
            self.arity_0_ids.append(tid)
            
        # Operators (1 or 2)
        for op in self.grammar.operators:
            tid = self.grammar.token_to_id[op]
            arity = OPERATORS[op]
            self.token_arity[tid] = arity
            if arity == 1: self.arity_1_ids.append(tid)
            elif arity == 2: self.arity_2_ids.append(tid)
            
        self.arity_0_ids = torch.tensor(self.arity_0_ids, device=self.device)
        self.arity_1_ids = torch.tensor(self.arity_1_ids, device=self.device)
        self.arity_2_ids = torch.tensor(self.arity_2_ids, device=self.device)
        
        # The Sniper
        self.sniper = Sniper(self.device)
        
        # Pareto Optimizer (NSGA-II)
        self.pareto = ParetoOptimizer(self.device, GpuGlobals.PARETO_MAX_FRONT_SIZE)
        
        # Pattern Memory
        self.pattern_memory = PatternMemory(
            self.device, 
            max_patterns=100,
            fitness_threshold=GpuGlobals.PATTERN_RECORD_FITNESS_THRESHOLD,
            min_uses=GpuGlobals.PATTERN_MEM_MIN_USES
        )


    def optimize_constants(self, population: torch.Tensor, constants: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor, steps=10, lr=0.1):
        """
        Refine constants using Gradient Descent (Adam).
        Returns: (best_constants, best_mse)
        """
        # Optimize a COPY of constants
        optimized_consts = constants.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([optimized_consts], lr=lr)
        
        # Track best found during steps (in case it diverges)
        best_mse = torch.full((population.shape[0],), float('inf'), device=self.device, dtype=torch.float64)
        best_consts = constants.clone().detach() 
        
        for _ in range(steps):
            optimizer.zero_grad()
            
            # Forward pass (differentiable if we implemented soft operations, 
            # but standard ops are differentiable in PyTorch!)
            # Evaluator returns RMSE, but we want MSE for gradients usually, or just minimize RMSE.
            # evaluate_batch returns RMSE [PopSize].
            # Problem: evaluate_batch uses scatter_ (in-place) which might break gradients if not careful.
            # However, for simple constant optimization, we might need a "soft" stack or ignore in-place issues if PyTorch handles them.
            # Let's try standard evaluate_batch. If scatter breaks, we might need a rewriting.
            # ACTUALLY: scatter_ IS differentiable for values, but not indices (indices are fixed by RPN).
            # So this SHOULD work.
            
            rmse = self.evaluate_batch(population, x, y_target, optimized_consts)
            
            # Loss = Sum of RMSEs (to optimize all in parallel)
            # We filter NaNs
            valid_mask = ~torch.isnan(rmse)
            if not valid_mask.any(): break
            
            # Update bests
            current_mse = rmse**2 # Approximation since we returned RMSE
            improved = (current_mse < best_mse) & valid_mask
            if improved.any():
                best_mse[improved] = current_mse[improved].detach()
                best_consts[improved] = optimized_consts[improved].detach()
            
            loss = rmse[valid_mask].sum()
            
            if not loss.requires_grad: 
                # This happens if formula has no 'C' or operations detach graph
                break
                
            loss.backward()
            optimizer.step()
            
        return best_consts, torch.sqrt(best_mse)

    def local_search(self, population: torch.Tensor, constants: torch.Tensor, 
                     x: torch.Tensor, y: torch.Tensor, 
                     top_k: int = 10, attempts: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hill climbing: try single-token mutations on top individuals, keep improvements.
        
        Args:
            population: [PopSize, L] RPN tensors
            constants: [PopSize, MaxC] constants
            x: Input data
            y: Target data  
            top_k: Number of top individuals to apply local search
            attempts: Number of mutation attempts per individual (default: LOCAL_SEARCH_ATTEMPTS)
        
        Returns:
            (improved_population, improved_constants)
        """
        if attempts is None:
            attempts = GpuGlobals.LOCAL_SEARCH_ATTEMPTS
        
        pop_out = population.clone()
        const_out = constants.clone()
        
        # Get top K individuals by fitness
        fitness = self.evaluate_batch(population, x, y, constants)
        _, top_idx = torch.topk(fitness, top_k, largest=False)
        
        for idx in top_idx:
            idx = idx.item()
            current_rpn = population[idx:idx+1]
            current_const = constants[idx:idx+1]
            current_fit = fitness[idx].item()
            
            best_rpn = current_rpn.clone()
            best_const = current_const.clone()
            best_fit = current_fit
            
            # Try random single-token mutations
            for _ in range(attempts):
                # Mutate with high rate (1 token expected change)
                mutant = self.mutate_population(current_rpn, mutation_rate=0.15)
                
                # Evaluate mutant
                mutant_fit = self.evaluate_batch(mutant, x, y, current_const)[0].item()
                
                if mutant_fit < best_fit:
                    best_rpn = mutant.clone()
                    best_fit = mutant_fit
            
            # Update if improved
            if best_fit < current_fit:
                pop_out[idx] = best_rpn[0]
                # Also optimize constants for the improved individual
                opt_const, _ = self.optimize_constants(best_rpn, best_const, x, y, steps=5)
                const_out[idx] = opt_const[0]
        
        return pop_out, const_out

    def simplify_expression(self, rpn_tensor: torch.Tensor, constants: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Simplify an RPN expression using SymPy.
        
        Args:
            rpn_tensor: [L] tensor of token IDs (single individual)
            constants: [MaxC] tensor of constant values
        
        Returns:
            (simplified_rpn, new_constants, success)
        """
        if not SYMPY_AVAILABLE or not GpuGlobals.USE_SIMPLIFICATION:
            return rpn_tensor, constants, False
        
        try:
            # 1. Convert RPN to infix string
            infix = self.rpn_to_infix(rpn_tensor, constants)
            if infix == "Invalid" or not infix:
                return rpn_tensor, constants, False
            
            # 2. Prepare SymPy symbols
            sym_vars = {f'x{i}': symbols(f'x{i}') for i in range(self.num_variables)}
            sym_vars['x'] = sym_vars.get('x0', symbols('x0'))  # Alias
            
            # 3. Parse to SymPy (handle operator conversions)
            expr_str = infix
            expr_str = expr_str.replace('^', '**')  # Power
            expr_str = expr_str.replace('lgamma', 'loggamma')
            
            # Try parsing
            try:
                expr = parse_expr(expr_str, local_dict=sym_vars)
            except:
                # Fallback to sympify
                expr = sympify(expr_str, locals=sym_vars)
            
            # 4. Simplify
            simplified = simplify(expr)
            
            # 5. Rationalize constants (e.g., 0.5 -> 1/2)
            simplified = nsimplify(simplified, tolerance=1e-6, rational=True)
            
            # 6. Convert back to infix string
            simplified_str = str(simplified)
            
            # 7. If simplification made it longer, abort
            if len(simplified_str) > len(infix) * 1.5:
                return rpn_tensor, constants, False
            
            # 8. Convert to our format (** -> ^, etc.)
            simplified_str = simplified_str.replace('**', ' ^ ')
            simplified_str = simplified_str.replace('loggamma', 'lgamma')
            
            # 9. Convert back to RPN
            new_rpn = self.infix_to_rpn([simplified_str])
            if new_rpn.shape[0] == 0 or (new_rpn[0] == PAD_ID).all():
                return rpn_tensor, constants, False
            
            # 10. Extract new constants from simplified expression
            # For now, we initialize with zeros (optimizer will refine)
            new_consts = torch.zeros(self.max_constants, device=self.device, dtype=torch.float64)
            
            # Count 'C' tokens in new RPN
            id_C = self.grammar.token_to_id.get('C', -1)
            n_consts = (new_rpn[0] == id_C).sum().item()
            
            # Try to extract numeric constants from simplified_str
            import re
            numbers = re.findall(r'[-+]?\d*\.?\d+', simplified_str)
            for i, num in enumerate(numbers[:min(n_consts, self.max_constants)]):
                try:
                    new_consts[i] = float(num)
                except:
                    pass
            
            return new_rpn[0], new_consts, True
            
        except Exception as e:
            # Simplification failed, return original
            return rpn_tensor, constants, False

    def simplify_population(self, population: torch.Tensor, constants: torch.Tensor, top_k: int = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Simplify top K individuals in the population.
        
        Args:
            population: [PopSize, L] RPN tensors
            constants: [PopSize, MaxC] constant tensors
            top_k: Number of individuals to simplify (default: 10% of population)
        
        Returns:
            (new_population, new_constants, n_simplified)
        """
        if not SYMPY_AVAILABLE or not GpuGlobals.USE_SIMPLIFICATION:
            return population, constants, 0
        
        if top_k is None:
            top_k = max(1, int(population.shape[0] * 0.1))
        
        n_simplified = 0
        pop_out = population.clone()
        const_out = constants.clone()
        
        for i in range(min(top_k, population.shape[0])):
            new_rpn, new_consts, success = self.simplify_expression(population[i], constants[i])
            if success:
                pop_out[i] = new_rpn
                const_out[i] = new_consts
                n_simplified += 1
        
        return pop_out, const_out, n_simplified

    def migrate_islands(self, population: torch.Tensor, constants: torch.Tensor, fitness: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform ring migration between islands.
        
        Top MIGRATION_SIZE individuals from each island migrate to the next island (ring topology),
        replacing the worst individuals in the destination.
        
        Args:
            population: [PopSize, L] RPN tensors
            constants: [PopSize, MaxC] constant tensors
            fitness: [PopSize] fitness scores (lower is better)
        
        Returns:
            (new_population, new_constants)
        """
        if self.n_islands <= 1:
            return population, constants
        
        pop_out = population.clone()
        const_out = constants.clone()
        
        island_size = self.island_size
        mig_size = min(GpuGlobals.MIGRATION_SIZE, island_size // 2)  # Don't migrate more than half
        
        for island in range(self.n_islands):
            # Source island
            src_start = island * island_size
            src_end = src_start + island_size
            
            # Destination island (ring: island+1 mod n_islands)
            dst_island = (island + 1) % self.n_islands
            dst_start = dst_island * island_size
            dst_end = dst_start + island_size
            
            # Get fitness for source island
            src_fitness = fitness[src_start:src_end]
            
            # Get indices of best individuals in source (lowest fitness)
            _, best_idx_local = torch.topk(src_fitness, mig_size, largest=False)
            best_idx_global = best_idx_local + src_start
            
            # Get fitness for destination island
            dst_fitness = fitness[dst_start:dst_end]
            
            # Get indices of worst individuals in destination (highest fitness)
            _, worst_idx_local = torch.topk(dst_fitness, mig_size, largest=True)
            worst_idx_global = worst_idx_local + dst_start
            
            # Migrate: replace worst in destination with best from source
            pop_out[worst_idx_global] = population[best_idx_global]
            const_out[worst_idx_global] = constants[best_idx_global]
        
        return pop_out, const_out

    def deduplicate_population(self, population: torch.Tensor, constants: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Remove duplicate individuals from the population.
        
        Duplicates are identified by hashing their RPN token sequence.
        Duplicates are replaced with **FRESH RANDOM INDIVIDUALS** (not mutated clones).
        
        Args:
            population: [PopSize, L] RPN tensors
            constants: [PopSize, MaxC] constant tensors
        
        Returns:
            (new_population, new_constants, n_replaced)
        """
        if not GpuGlobals.PREVENT_DUPLICATES:
            return population, constants, 0
        
        pop_size = population.shape[0]
        pop_cpu = population.cpu().numpy()
        
        # Hash each individual
        seen_hashes = {}
        duplicate_indices = []
        
        for i in range(pop_size):
            # Create hash from non-padding tokens
            tokens = pop_cpu[i]
            non_pad = tokens[tokens != PAD_ID]
            hash_key = tuple(non_pad.tolist())
            
            if hash_key in seen_hashes:
                duplicate_indices.append(i)
            else:
                seen_hashes[hash_key] = i
        
        n_dups = len(duplicate_indices)
        if n_dups == 0:
            return population, constants, 0
        
        # Replace duplicates with fresh random individuals
        pop_out = population.clone()
        const_out = constants.clone()
        
        # Generate N fresh random trees
        fresh_pop = self._generate_random_population(n_dups)
        fresh_consts = torch.randn(n_dups, constants.shape[1], device=self.device, dtype=torch.float64)
        
        # Assign to duplicate slots
        # duplicate_indices is a list, convert to tensor?
        # Actually indexing with list works in PyTorch if converted to tensor or list.
        # But `fresh_pop` is [n_dups, L].
        pop_out[duplicate_indices] = fresh_pop
        const_out[duplicate_indices] = fresh_consts
        
        return pop_out, const_out, n_dups

    def tarpeian_control(self, population: torch.Tensor, fitness: torch.Tensor) -> torch.Tensor:
        """
        Tarpeian bloat control: randomly penalize oversized individuals.
        
        Individuals longer than 1.5x average length have 50% chance of 
        receiving very bad fitness, pushing them out of selection.
        
        Args:
            population: [PopSize, L] RPN tensors
            fitness: [PopSize] current fitness values
            
        Returns:
            Modified fitness tensor
        """
        lengths = (population != PAD_ID).sum(dim=1).float()
        avg_len = lengths.mean()
        
        # Find oversized individuals (> 1.5x average)
        oversized = lengths > avg_len * 1.5
        
        # Randomly penalize 50% of oversized
        random_mask = torch.rand(population.shape[0], device=self.device) < 0.5
        penalize_mask = oversized & random_mask
        
        # Apply penalty
        fitness_out = fitness.clone()
        fitness_out[penalize_mask] = 1e30  # Large but within float32 range
        
        return fitness_out

    def shrink_mutation(self, individual: torch.Tensor) -> torch.Tensor:
        """
        Apply shrinking mutation - removes a subtree and replaces with terminal.
        """
        ind_cpu = individual.cpu().numpy()
        non_pad = ind_cpu[ind_cpu != PAD_ID]
        
        if len(non_pad) < 3:
            return individual
        
        # Pick a random operator position
        operator_positions = []
        for i, token_id in enumerate(non_pad):
            token = self.grammar.id_to_token.get(token_id, "")
            if token in self.grammar.operators:
                operator_positions.append(i)
        
        if not operator_positions:
            return individual
        
        target_pos = np.random.choice(operator_positions)
        span = self.grammar.get_subtree_span(non_pad.tolist(), target_pos)
        
        if span[0] == -1:
            return individual
        
        # Replace subtree with random terminal
        terminal_id = self.terminal_ids[torch.randint(len(self.terminal_ids), (1,))].item()
        new_tokens = list(non_pad[:span[0]]) + [terminal_id] + list(non_pad[target_pos+1:])
        new_tokens = new_tokens[:self.max_len]
        new_tokens = new_tokens + [PAD_ID] * (self.max_len - len(new_tokens))
        
        return torch.tensor(new_tokens, device=self.device, dtype=individual.dtype)


    def mutate_population(self, population: torch.Tensor, mutation_rate: float) -> torch.Tensor:
        """
        Performs arity-safe mutation on the population.
        """
        # Create mutation mask
        mask = torch.rand_like(population, dtype=torch.float32) < mutation_rate
        # Don't mutate padding
        mask = mask & (population != PAD_ID)
        
        # We need to know arity of current tokens to replace them with same arity
        # self.token_arity has shape [VocabSize+1]
        # Gather arity for each token in population
        current_arities = self.token_arity[population]
        
        # New Reference:
        # Arity 0 -> Sample from arity_0_ids
        # Arity 1 -> Sample from arity_1_ids
        # Arity 2 -> Sample from arity_2_ids
        
        # We can prepare 3 tensors of random replacements, one for each arity type, same shape as pop
        # Ideally only generate for needed spots, but fully generating is easier for vectorized code.
        
        # Random replacements for Arity 0
        if len(self.arity_0_ids) > 0:
            rand_idx_0 = torch.randint(0, len(self.arity_0_ids), population.shape, device=self.device)
            replacements_0 = self.arity_0_ids[rand_idx_0]
        else:
            replacements_0 = population
            
        # Random replacements for Arity 1
        if len(self.arity_1_ids) > 0:
             rand_idx_1 = torch.randint(0, len(self.arity_1_ids), population.shape, device=self.device)
             replacements_1 = self.arity_1_ids[rand_idx_1]
        else:
             replacements_1 = population

        # Random replacements for Arity 2
        if len(self.arity_2_ids) > 0:
             rand_idx_2 = torch.randint(0, len(self.arity_2_ids), population.shape, device=self.device)
             replacements_2 = self.arity_2_ids[rand_idx_2]
        else:
             replacements_2 = population
             
        # Apply Logic
        mutated_pop = population.clone()
        
        # Mask for Arity 0 mutations
        mask_0 = mask & (current_arities == 0)
        mutated_pop = torch.where(mask_0, replacements_0, mutated_pop)
        
        # Mask for Arity 1 mutations
        mask_1 = mask & (current_arities == 1)
        mutated_pop = torch.where(mask_1, replacements_1, mutated_pop)
        
        # Mask for Arity 2 mutations
        mask_2 = mask & (current_arities == 2)
        mutated_pop = torch.where(mask_2, replacements_2, mutated_pop)
        
        return mutated_pop

    def _get_subtree_ranges(self, population: torch.Tensor) -> torch.Tensor:
        """
        Calculates the start index of the subtree ending at each position.
        Returns tensor [B, L] where value is start_index, or -1 if invalid/padding.
        Optimized for RPN logic on GPU.
        """
        B, L = population.shape
        subtree_starts = torch.full((B, L), -1, device=self.device, dtype=torch.long)
        
        # 1. Map tokens to Arity Change
        # Variables/Consts: +1
        # Binary: -1 (Pop 2 Push 1 -> Net -1)
        # Unary: 0 (Pop 1 Push 1 -> Net 0)
        
        # We need a fast lookup.
        # Construct Arity Table
        # Default 1 (Operand)
        arities = torch.ones_like(population, dtype=torch.long) 
        
        # Binary (-1)
        op_add = self.grammar.token_to_id.get('+', -100); op_sub = self.grammar.token_to_id.get('-', -100)
        op_mul = self.grammar.token_to_id.get('*', -100); op_div = self.grammar.token_to_id.get('/', -100)
        op_pow = self.grammar.token_to_id.get('pow', -100); op_mod = self.grammar.token_to_id.get('%', -100)
        
        mask_bin = (population == op_add) | (population == op_sub) | (population == op_mul) | \
                   (population == op_div) | (population == op_pow) | (population == op_mod)
        arities[mask_bin] = -1
        
        # Unary (0)
        op_sin = self.grammar.token_to_id.get('sin', -100); op_cos = self.grammar.token_to_id.get('cos', -100)
        # ... (add all unary)
        # Simplified: If it's not binary and it's an operator, it's unary?
        # Better: Assume all Ops are <= some ID? No.
        # Explicit list is safer.
        unary_tokens = ['sin','cos','tan','S','C','T','e','log','sqrt','abs','neg','!','_','g']
        unary_ids = [self.grammar.token_to_id.get(t, -999) for t in unary_tokens]
        # Make tensor of unary ids
        # Ideally this table is precomputed. For now, on the fly is ok.
        
        mask_unary = torch.zeros_like(population, dtype=torch.bool)
        for uid in unary_ids:
             mask_unary = mask_unary | (population == uid)
        arities[mask_unary] = 0
        
        # Padding: -999 (Invalid)
        arities[population == PAD_ID] = -999
        
        # 2. To find start of subtree ending at 'end', we scan backwards until cumulative sum is +1.
        # Since scanning backwards is hard in vector, we can scan loops?
        # Max depth isn't huge.
        
        # Alternatively: "Stack Depth at step i".
        # Subtree at 'end' corresponds to the interval [start, end] where
        # Depth(start-1) = D
        # Depth(end) = D + 1
        # And min_depth(start...end) >= D
        
        # Let's compute cumulative sum (Stack Depth Profile)
        # cum_arity[i] is depth AFTER processing token i.
        # arities for this: PAD=0? No, PAD breaks it.
        # Let's mask PAD for cumsum.
        
        safe_arities = arities.clone()
        safe_arities[population == PAD_ID] = 0
        depths = torch.cumsum(safe_arities, dim=1)
        
        # The scan logic is still tricky O(L^2) across batch.
        # For L=30, iterating i from 0 to L is fast.
        
        for i in range(L):
            # If position i is PAD, skip
            is_pad = (population[:, i] == PAD_ID)
            
            # We want to find 'start' such that sum(arities[start...i]) == 1
            # Which means depths[i] - depths[start-1] == 1
            # implies depths[start-1] = depths[i] - 1.
            # And for all k in start...i, depths[k] >= depths[start-1] (validity).
            
            target_depth = depths[:, i] - 1
            
            # Search backwards from i
            # We can vectorize this search over B by iterating j downwards
            current_start = torch.full((B,), -1, device=self.device, dtype=torch.long)
            found = torch.zeros(B, dtype=torch.bool, device=self.device)
            
            # Optimization: Pre-calculate validity masks?
            # Brute force backwards for L=30 is fine.
            for j in range(i, -1, -1):
                # Check condition for batch
                # d[j-1] == target?
                # Actually, depth[j-1] is depth BEFORE processing j.
                # If j=0, depth[-1] = 0.
                
                prev_depth = depths[:, j-1] if j > 0 else torch.zeros(B, device=self.device)
                
                # Match condition: prev_depth == target_depth
                match = (prev_depth == target_depth)
                
                # Check if we violated lower bound in between?
                # Implicitly, if we hit match first time going backwards, it's the minimal subtree.
                # We update 'found' mask.
                
                new_found = match & (~found)
                current_start[new_found] = j
                found = found | new_found
            
            # Store valid starts for this end position 'i'
            # Only valid if not PAD and found
            valid_i = (~is_pad) & found
            subtree_starts[valid_i, i] = current_start[valid_i]
            
        return subtree_starts

    def crossover_population(self, parents: torch.Tensor, crossover_rate: float) -> torch.Tensor:
        """
        Performs subtree crossover on the population (Two-Child Crossover) fully on GPU.
        parents: [PopSize, L]
        """
        B, L = parents.shape
        n_pairs = int(B * 0.5 * crossover_rate)
        if n_pairs == 0: return parents.clone()
        
        # 1. Partner Selection (Random Shuffle)
        perm = torch.randperm(B, device=self.device)
        p1_idx = perm[:n_pairs*2:2]
        p2_idx = perm[1:n_pairs*2:2]
        
        parents_1 = parents[p1_idx] # [N, L]
        parents_2 = parents[p2_idx] # [N, L]
        
        # 2. Subtree Ranges
        # We need starts for all potential end points
        # _get_subtree_ranges returns [N, L] where val is start_idx or -1
        starts_1_mat = self._get_subtree_ranges(parents_1)
        starts_2_mat = self._get_subtree_ranges(parents_2)
        
        # Select random VALID crossover points
        # A valid point is where start != -1
        valid_mask_1 = (starts_1_mat != -1)
        valid_mask_2 = (starts_2_mat != -1)
        
        # For each pair, sample one index from valid points
        # We can use torch.multinomial on the mask (treated as weights)
        # Add epsilon to avoid error on all-invalid (shouldn't happen for valid RPN)
        probs_1 = valid_mask_1.float() + 1e-6
        probs_2 = valid_mask_2.float() + 1e-6
        
        end_1 = torch.multinomial(probs_1, 1).squeeze(1) # [N]
        end_2 = torch.multinomial(probs_2, 1).squeeze(1) # [N]
        
        start_1 = starts_1_mat.gather(1, end_1.unsqueeze(1)).squeeze(1) # [N]
        start_2 = starts_2_mat.gather(1, end_2.unsqueeze(1)).squeeze(1) # [N]
        
        # Lengths of parts
        # P1: [0...start_1-1] [start_1...end_1] [end_1+1...L]
        # P2: [0...start_2-1] [start_2...end_2] [end_2+1...L]
        
        # Segments lengths
        # Note: slices are python style [start:end] (exclusive)
        # So len = end - start
        
        # Parent 1 parts
        len_1_pre = start_1
        len_1_sub = end_1 - start_1 + 1
        len_1_post = L - 1 - end_1 # Counts remaining including padding?
        # Actually actual tokens might end earlier.
        # But we want to preserve tails including potential useful tokens?
        # Usually standard subtree crossover swaps indices.
        # If we respect fixed L, we just chop/pad.
        
        # Let's count actual useful tail? 
        # For efficiency, we just take everything after end_1 up to L. 
        # But wait, max length constraint.
        
        # Child 1: P1_pre + P2_sub + P1_post
        # Child 2: P2_pre + P1_sub + P2_post
        
        len_2_pre = start_2
        len_2_sub = end_2 - start_2 + 1
        
        # New Valid Lengths (approximation, assuming tail is full or we truncate)
        # We must truncate if new length > L.
        # Construct Gather Map [N, L]
        
        # We build indices for C1 derived from P1/P2/P1
        # Part 1: 0 to len_1_pre (indices 0..s1-1 from P1)
        # Part 2: len_1_pre to len_1_pre + len_2_sub (indices s2..e2 from P2)
        # Part 3: ... (indices e1+1..L from P1)
        
        # Vectorized Index Generation
        # We create a base grid [N, L] with values 0..L-1
        grid = torch.arange(L, device=self.device).unsqueeze(0).expand(n_pairs, L)
        
        # Mask 1: grid < len_1_pre
        mask_c1_pre = (grid < len_1_pre.unsqueeze(1))
        
        # Mask 2: grid >= len_1_pre AND grid < (len_1_pre + len_2_sub)
        cut_1 = len_1_pre + len_2_sub
        mask_c1_mid = (grid >= len_1_pre.unsqueeze(1)) & (grid < cut_1.unsqueeze(1))
        
        # Mask 3: grid >= cut_1
        mask_c1_post = (grid >= cut_1.unsqueeze(1))
        
        # Indices for C1
        # If pre: index = grid
        # If mid: index = grid - len_1_pre + start_2
        # If post: index = grid - cut_1 + end_1 + 1
        
        idx_c1 = torch.zeros((n_pairs, L), dtype=torch.long, device=self.device)
        
        term_1 = grid
        term_2 = grid - len_1_pre.unsqueeze(1) + start_2.unsqueeze(1)
        term_3 = grid - cut_1.unsqueeze(1) + end_1.unsqueeze(1) + 1
        
        # Apply Logic
        # Note: If term > L-1 (out of bounds), we clamp to PAD later?
        # Or we rely on mask.
        
        # We need to handle truncated vs padded.
        # If result is longer than L, the mask logic naturally truncates (we only compute L outputs).
        # We just need to valid indices.
        # Indices >= L should map to PAD_ID.
        
        # Check source bounds
        # term can be anything.
        # We must mask invalid sources.
        
        # Combining
        # We can use torch.where chaining
        src_idx_c1 = torch.where(mask_c1_pre, term_1, 
                       torch.where(mask_c1_mid, term_2, term_3))
                       
        # Child 2 Logic
        # C2: P2_pre + P1_sub + P2_post
        cut_2 = len_2_pre + len_1_sub
        
        mask_c2_pre = (grid < len_2_pre.unsqueeze(1))
        mask_c2_mid = (grid >= len_2_pre.unsqueeze(1)) & (grid < cut_2.unsqueeze(1))
        # Mask 3 is implicitly rest
        
        term_2_1 = grid
        term_2_2 = grid - len_2_pre.unsqueeze(1) + start_1.unsqueeze(1)
        term_2_3 = grid - cut_2.unsqueeze(1) + end_2.unsqueeze(1) + 1
        
        src_idx_c2 = torch.where(mask_c2_pre, term_2_1,
                       torch.where(mask_c2_mid, term_2_2, term_2_3))
                       
        
        # Perform Gather
        # We need to know WHICH tensor to gather from for each position.
        # C1: P1, P2, P1
        # C2: P2, P1, P2
        
        # We can construct a "Source Selector" (0=P1, 1=P2)
        # C1_sel: 0 if pre/post, 1 if mid
        sel_c1 = torch.where(mask_c1_mid, torch.tensor(1, device=self.device), torch.tensor(0, device=self.device))
        
        # C2_sel: 1 if pre/post, 0 if mid (Wait, C2 base is P2=1. So pre/post=1, mid=0)
        sel_c2 = torch.where(mask_c2_mid, torch.tensor(0, device=self.device), torch.tensor(1, device=self.device))
        
        # Helper gather
        def gather_mixed(idx_map, sel_map, t0, t1):
            # idx_map: [N, L] indices
            # sel_map: [N, L] 0 or 1
            # t0, t1: [N, L] input tensors
            
            # Safe clamping for indices to avoid error gather (mask later)
            # Max index is L-1.
            safe_idx = torch.clamp(idx_map, 0, L-1)
            
            val0 = t0.gather(1, safe_idx)
            val1 = t1.gather(1, safe_idx)
            
            res = torch.where(sel_map == 0, val0, val1)
            
            # Pad masking
            # If idx_map < 0 or >= L, result is PAD
            # Also if we are in "post" region and we exceeded original length?
            # Our index arithmetic naturally points to higher indices.
            # If pointing > L-1, it's garbage/pad.
            # Actually, we should check if idx_map >= L.
            is_pad = (idx_map < 0) | (idx_map >= L)
            res[is_pad] = PAD_ID
            return res
            
        c1 = gather_mixed(src_idx_c1, sel_c1, parents_1, parents_2)
        c2 = gather_mixed(src_idx_c2, sel_c2, parents_1, parents_2)
        
        # Update population
        parents[p1_idx] = c1
        parents[p2_idx] = c2
        
        return parents




    def infix_to_rpn_tensor(self, formula_str: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts an infix string to an RPN tensor (padded to max_len) and constants tensor.
        Handles parsing constants and extracting them.
        """
        try:
            # Use ExpressionTree for parsing structure
            tree = ExpressionTree.from_infix(formula_str)
            if not tree.is_valid:
                return torch.zeros(self.max_len, dtype=torch.long, device=self.device), torch.zeros(self.max_constants, dtype=torch.float64, device=self.device)
            
            # Use traverse to get tokens in RPN order (Post-order)
            rpn_tokens = []
            def traverse(node):
                if not node: return
                for child in node.children:
                    traverse(child)
                rpn_tokens.append(node.value)
            
            traverse(tree.root)
            
            clean_tokens = []
            const_values = []
            
            for t in rpn_tokens:
                 # Check if constant
                 # We support literals '1','2','3','5'. Others map to 'C'.
                 if t in self.grammar.terminals and t not in ['C', '1', '2', '3', '5'] and not t.startswith('x'):
                     # It's likely a variable x0..xn, keep it.
                     clean_tokens.append(t)
                 elif (t.replace('.','',1).isdigit() or (t.startswith('-') and t[1:].replace('.','',1).isdigit())):
                     # Numeric literal
                     if t in ['1', '2', '3', '5']:
                         clean_tokens.append(t)
                     else:
                         clean_tokens.append('C')
                         const_values.append(float(t))
                 else:
                     clean_tokens.append(t)
            
            # Map to IDs
            ids = [self.grammar.token_to_id.get(t, PAD_ID) for t in clean_tokens]
            
            # Pad
            if len(ids) > self.max_len:
                ids = ids[:self.max_len]
            else:
                ids += [PAD_ID] * (self.max_len - len(ids))
                
            # Constants padding
            if len(const_values) > self.max_constants:
                const_values = const_values[:self.max_constants]
            else:
                const_values += [0.0] * (self.max_constants - len(const_values))
                
            return torch.tensor(ids, dtype=torch.long, device=self.device), torch.tensor(const_values, dtype=torch.float64, device=self.device)
            
        except Exception as e:
            # print(f"Error parsing formula '{formula_str}': {e}")
            return torch.zeros(self.max_len, dtype=torch.long, device=self.device), torch.zeros(self.max_constants, dtype=torch.float64, device=self.device)

    def load_population_from_strings(self, formulas: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads a list of infix strings into the population tensors.
        Returns: (population, constants)
        """
        rpn_list = []
        const_list = []
        
        for f in formulas:
            r, c = self.infix_to_rpn_tensor(f)
            # Check validity (sum > 0 implies non-empty)
            if (r != PAD_ID).any():
                rpn_list.append(r)
                const_list.append(c)
                
        if not rpn_list:
             return None, None
             
        pop = torch.stack(rpn_list)
        consts = torch.stack(const_list)
        return pop, consts

    def infix_to_rpn(self, formulas: List[str]) -> torch.Tensor:
        """
        Legacy/Simple converter without constant extraction (constants become '1.0' or ignored if not in grammar).
        For backward compatibility or simple usage.
        """
        batch_rpn = []
        for formula_str in formulas:
            try:
                tree = ExpressionTree.from_infix(formula_str)
                if not tree.is_valid:
                    batch_rpn.append([PAD_ID]*self.max_len)
                    continue

                rpn_tokens = []
                def traverse(node):
                    if not node: return
                    for child in node.children:
                        traverse(child)
                    rpn_tokens.append(node.value)
                traverse(tree.root)

                ids = [self.grammar.token_to_id.get(t, PAD_ID) for t in rpn_tokens]
                if len(ids) > self.max_len:
                    ids = ids[:self.max_len]
                else:
                    ids = ids + [PAD_ID] * (self.max_len - len(ids))
                
                batch_rpn.append(ids)
            except Exception as e:
                batch_rpn.append([PAD_ID]*self.max_len)
                
        if not batch_rpn:
            return torch.empty((0, self.max_len), device=self.device, dtype=torch.long)
        return torch.tensor(batch_rpn, device=self.device, dtype=torch.long)


    def rpn_to_infix(self, rpn_tensor: torch.Tensor, constants: torch.Tensor = None) -> str:
        """
        Decodes RPN tensor to Infix string (CPU-style formatting).
        """
        if rpn_tensor.ndim > 1:
            rpn_tensor = rpn_tensor.view(-1)
            
        vocab = self.grammar.id_to_token
        stack = []
        const_idx = 0
        
        for token_id in rpn_tensor:

            token_id = token_id.item()
            if token_id == PAD_ID: continue
            
            token = vocab.get(token_id, "")
            
            if token in self.grammar.operators:
                arity = self.grammar.token_arity.get(token, 2)
                if arity == 1:
                    if not stack: return "Invalid"
                    a = stack.pop()
                    if token == 's': stack.append(f"sin({a})")
                    elif token == 'c': stack.append(f"cos({a})")
                    elif token == 'l': stack.append(f"log({a})")
                    elif token == 'e' or token == 'exp': stack.append(f"exp({a})")
                    elif token == 'q' or token == 'sqrt': stack.append(f"sqrt({a})")
                    elif token == 'a' or token == 'abs': stack.append(f"abs({a})")
                    elif token == 'n' or token == 'sign': stack.append(f"sign({a})")
                    elif token == 'neg': stack.append(f"neg({a})")
                    elif token == '_' or token == 'floor': stack.append(f"floor({a})")
                    elif token == '!' or token == 'gamma': stack.append(f"gamma({a})")
                    elif token == 'g' or token == 'lgamma': stack.append(f"lgamma({a})")
                    elif token == 'S' or token == 'asin': stack.append(f"asin({a})")
                    elif token == 'C' or token == 'acos': stack.append(f"acos({a})")
                    elif token == 'T' or token == 'atan': stack.append(f"atan({a})")
                    else: stack.append(f"{token}({a})")
                else: # Binary
                    if len(stack) < 2: return "Invalid"
                    b = stack.pop()
                    a = stack.pop()
                    
                    if token == '+' and b.startswith("-") and not b.startswith("(-"):
                         stack.append(f"({a} - {b[1:]})")
                    elif token == '-' and a == "0":
                         stack.append(f"(-{b})")
                    elif token == 'pow':
                         stack.append(f"({a} ^ {b})")
                    elif token == 'mod':
                         stack.append(f"({a} % {b})")
                    else:
                         stack.append(f"({a} {token} {b})")
            elif token == 'C':
                val = 1.0
                if constants is not None and const_idx < len(constants):
                    val = constants[const_idx].item()
                    const_idx += 1
                stack.append(format_const(val))
            elif token.startswith('x'):
                if token == 'x': stack.append("x0")
                else: stack.append(token)
            else:
                stack.append(str(token))
                
        if len(stack) == 1:
            return stack[0]
        return "Invalid"
    
    def get_tree_size(self, rpn_tensor: torch.Tensor) -> int:
        """
        Returns number of non-pad nodes.
        """
        return (rpn_tensor != PAD_ID).sum().item()
    


    @torch.compile(mode="reduce-overhead", dynamic=False)
    def _run_vm(self, population: torch.Tensor, x: torch.Tensor, constants: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Internal VM interpreter to evaluate RPN population on the GPU.
        Returns: (final_predictions, stack_pointer, has_error)
        """
        B, L = population.shape
        D = x.shape[0]
        MAX_STACK = 10
        eff_B = B * D
        
        pop_expanded = population.unsqueeze(1).expand(-1, D, -1).reshape(eff_B, L)
        const_expanded = None
        if constants is not None:
             const_expanded = constants.unsqueeze(1).expand(-1, D, -1).reshape(eff_B, -1)
             
        if x.ndim == 1:
            x_expanded = x.unsqueeze(0).expand(B, -1).reshape(eff_B, 1)
        else:
            x_expanded = x.unsqueeze(0).expand(B, -1, -1).reshape(eff_B, x.shape[1])
            
        stack = torch.zeros(eff_B, MAX_STACK, device=self.device, dtype=torch.float64)
        sp = torch.zeros(eff_B, device=self.device, dtype=torch.long)
        const_counters = torch.zeros(eff_B, device=self.device, dtype=torch.long)
        
        # NEW: Error tracking
        has_error = torch.zeros(eff_B, dtype=torch.bool, device=self.device)
        
        pi_val = torch.tensor(np.pi, device=self.device, dtype=torch.float64)
        e_val = torch.tensor(np.e, device=self.device, dtype=torch.float64)

        # IDs
        id_C = self.grammar.token_to_id.get('C', -100)
        id_pi = self.grammar.token_to_id.get('pi', -100)
        id_e = self.grammar.token_to_id.get('e', -100)
        
        op_add = self.grammar.token_to_id.get('+', -100); op_sub = self.grammar.token_to_id.get('-', -100)
        op_mul = self.grammar.token_to_id.get('*', -100); op_div = self.grammar.token_to_id.get('/', -100)
        op_pow = self.grammar.token_to_id.get('pow', -100); op_mod = self.grammar.token_to_id.get('%', -100)
        op_sin = self.grammar.token_to_id.get('sin', -100); op_cos = self.grammar.token_to_id.get('cos', -100)
        op_tan = self.grammar.token_to_id.get('tan', -100)
        op_asin = self.grammar.token_to_id.get('S', -100); op_acos = self.grammar.token_to_id.get('C', -100); op_atan = self.grammar.token_to_id.get('T', -100)
        op_exp = self.grammar.token_to_id.get('e', -100); op_log = self.grammar.token_to_id.get('log', -100)
        op_sqrt = self.grammar.token_to_id.get('sqrt', -100); op_abs = self.grammar.token_to_id.get('abs', -100); op_neg = self.grammar.token_to_id.get('neg', -100)
        op_fact = self.grammar.token_to_id.get('!', -100); op_floor = self.grammar.token_to_id.get('_', -100); op_gamma = self.grammar.token_to_id.get('g', -100)
        
        var_ids = [self.grammar.token_to_id.get(v, -100) for v in self.grammar.active_variables]
        id_x_legacy = self.grammar.token_to_id.get('x', -100)

        for i in range(L):
            token = pop_expanded[:, i]
            active_mask = (token != PAD_ID)
            if not active_mask.any(): continue
            
            push_vals = torch.zeros(eff_B, device=self.device, dtype=torch.float64)
            is_operand = torch.zeros(eff_B, dtype=torch.bool, device=self.device)
            
            # Variables
            mask = (token == id_x_legacy)
            if mask.any():
                push_vals[mask] = x_expanded[mask, 0]
                is_operand = is_operand | mask
                
            for v_idx, vid in enumerate(var_ids):
                mask = (token == vid)
                if mask.any():
                    v_col = v_idx if v_idx < x_expanded.shape[1] else 0
                    push_vals[mask] = x_expanded[mask, v_col]
                    is_operand = is_operand | mask
            
            mask = (token == id_pi)
            if mask.any(): push_vals[mask] = pi_val; is_operand = is_operand | mask
            mask = (token == id_e)
            if mask.any(): push_vals[mask] = e_val; is_operand = is_operand | mask
                
            mask = (token == id_C)
            if mask.any():
                if const_expanded is not None:
                     safe_idx = torch.clamp(const_counters, 0, const_expanded.shape[1]-1)
                     c_vals = const_expanded.gather(1, safe_idx.unsqueeze(1)).squeeze(1)
                     push_vals[mask] = c_vals[mask]
                     const_counters[mask] += 1
                else:
                     push_vals[mask] = 1.0 
                is_operand = is_operand | mask
            
            for val_str in ['1', '2', '3', '5']:
                vid = self.grammar.token_to_id.get(val_str, -999)
                mask = (token == vid)
                if mask.any():
                    push_vals[mask] = float(val_str)
                    is_operand = is_operand | mask
                    
            if is_operand.any():
                safe_sp = torch.clamp(sp, 0, MAX_STACK-1)
                stack = stack.scatter(1, safe_sp.unsqueeze(1), push_vals.unsqueeze(1))
                sp = sp + is_operand.long()
                
            # Binary
            is_binary = (token == op_add) | (token == op_sub) | (token == op_mul) | (token == op_div) | (token == op_pow) | (token == op_mod)
            
            enough_stack = (sp >= 2)
            valid_op = is_binary & enough_stack
            
            has_error = has_error | (is_binary & ~enough_stack)
            
            if valid_op.any():
                idx_b = torch.clamp(sp - 1, 0, MAX_STACK - 1).unsqueeze(1); val_b = stack.gather(1, idx_b).squeeze(1)
                idx_a = torch.clamp(sp - 2, 0, MAX_STACK - 1).unsqueeze(1); val_a = stack.gather(1, idx_a).squeeze(1)
                res = torch.zeros_like(val_a)
                
                m = (token == op_add) & valid_op; res[m] = val_a[m] + val_b[m]
                m = (token == op_sub) & valid_op; res[m] = val_a[m] - val_b[m]
                m = (token == op_mul) & valid_op; res[m] = val_a[m] * val_b[m]
                m = (token == op_div) & valid_op
                if m.any(): 
                    d = val_b[m]; bad = d.abs() < 1e-9; sd = torch.where(bad, torch.tensor(1.0, device=self.device, dtype=torch.float64), d)
                    out = val_a[m] / sd; out[bad] = 1e150; res[m] = out
                m = (token == op_mod) & valid_op
                if m.any():
                    d = val_b[m]; bad = d.abs() < 1e-9; sd = torch.where(bad, torch.tensor(1.0, device=self.device, dtype=torch.float64), d)
                    out = torch.fmod(val_a[m], sd); out[bad] = 1e150; res[m] = out
                m = (token == op_pow) & valid_op; 
                if m.any(): res[m] = torch.pow(val_a[m], val_b[m])
                
                wp = torch.clamp(sp - 2, 0, MAX_STACK-1)
                curr = stack.gather(1, wp.unsqueeze(1)).squeeze(1)
                fw = torch.where(valid_op, res, curr)
                stack = stack.scatter(1, wp.unsqueeze(1), fw.unsqueeze(1)); sp = sp - valid_op.long()
                
            # Unary
            is_unary = (token == op_sin) | (token == op_cos) | (token == op_tan) | \
                       (token == op_asin) | (token == op_acos) | (token == op_atan) | \
                       (token == op_exp) | (token == op_log) | \
                       (token == op_sqrt) | (token == op_abs) | (token == op_neg) | \
                       (token == op_fact) | (token == op_floor) | (token == op_gamma)
            
            enough_stack = (sp >= 1)
            valid_op = is_unary & enough_stack
            
            has_error = has_error | (is_unary & ~enough_stack)
            
            if valid_op.any():
                idx_a = torch.clamp(sp - 1, 0, MAX_STACK - 1).unsqueeze(1); val_a = stack.gather(1, idx_a).squeeze(1); res = torch.zeros_like(val_a)
                m = (token == op_sin) & valid_op; res[m] = torch.sin(val_a[m])
                m = (token == op_cos) & valid_op; res[m] = torch.cos(val_a[m])
                m = (token == op_tan) & valid_op; res[m] = torch.tan(val_a[m])
                m = (token == op_log) & valid_op
                if m.any(): 
                    inv = val_a[m]; s = inv > 1e-9; out = torch.full_like(inv, 1e150); out[s] = torch.log(inv[s]); res[m] = out
                m = (token == op_exp) & valid_op
                if m.any(): 
                    inv = val_a[m]; s = inv <= 700.0; out = torch.full_like(inv, 1e150); out[s] = torch.exp(inv[s]); res[m] = out
                m = (token == op_sqrt) & valid_op; res[m] = torch.sqrt(val_a[m].abs())
                m = (token == op_abs) & valid_op; res[m] = torch.abs(val_a[m])
                m = (token == op_neg) & valid_op; res[m] = -val_a[m]
                m = (token == op_asin) & valid_op; res[m] = torch.asin(torch.clamp(val_a[m], -1.0, 1.0))
                m = (token == op_acos) & valid_op; res[m] = torch.acos(torch.clamp(val_a[m], -1.0, 1.0))
                m = (token == op_atan) & valid_op; res[m] = torch.atan(val_a[m])
                m = (token == op_floor) & valid_op; res[m] = torch.floor(val_a[m])
                m = (token == op_fact) & valid_op
                if m.any():
                    inv = val_a[m]; u = (inv < 0) | (inv > 170.0); out = torch.full_like(inv, 1e150)
                    si = inv.clone(); si[u] = 1.0; vc = torch.special.gamma(si + 1.0); out[~u] = vc[~u]; res[m] = out
                m = (token == op_gamma) & valid_op
                if m.any():
                    inv = val_a[m]; u = (inv <= -1.0); out = torch.full_like(inv, 1e150)
                    si = inv.clone(); si[u] = 1.0; vc = torch.special.gammaln(si + 1.0); out[~u] = vc[~u]; res[m] = out

                wp = torch.clamp(sp - 1, 0, MAX_STACK-1); curr = stack.gather(1, wp.unsqueeze(1)).squeeze(1)
                fw = torch.where(valid_op, res, curr); stack = stack.scatter(1, wp.unsqueeze(1), fw.unsqueeze(1))
        return stack[:, 0], sp, has_error

    def evaluate_batch(self, population: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor, constants: torch.Tensor = None) -> torch.Tensor:
        """
        Evaluates the RPN population on the GPU.
        Returns: RMSE per individual [PopSize]
        """
        B, L = population.shape
        D = x.shape[0]
        
        final_preds, sp, has_error = self._run_vm(population, x, constants)
        
        is_valid = (sp == 1) & (~has_error)
        # Use parity with C++: if not valid or nan/inf, penalty 1e300
        final_preds = torch.where(is_valid & ~torch.isnan(final_preds) & ~torch.isinf(final_preds), 
                                  final_preds, 
                                  torch.tensor(1e300, device=self.device, dtype=torch.float64))
                                  
        preds_matrix = final_preds.view(B, D)
        target_matrix = y_target.unsqueeze(0).expand(B, -1)
        mse = torch.mean((preds_matrix - target_matrix)**2, dim=1)
        
        # Guard against MSE itself being Inf/NaN after mean
        rmse = torch.sqrt(torch.where(torch.isnan(mse) | torch.isinf(mse), 
                                      torch.tensor(1e150, device=self.device, dtype=torch.float64), 
                                      mse))
        return rmse

        
        # Precompute IDs
        id_C = self.grammar.token_to_id.get('C', -100)
        id_pi = self.grammar.token_to_id.get('pi', -100)
        id_e = self.grammar.token_to_id.get('e', -100)
        
        # Op IDs
        op_add = self.grammar.token_to_id.get('+', -100)
        op_sub = self.grammar.token_to_id.get('-', -100)
        op_mul = self.grammar.token_to_id.get('*', -100)
        op_div = self.grammar.token_to_id.get('/', -100)
        op_pow = self.grammar.token_to_id.get('pow', -100)
        op_mod = self.grammar.token_to_id.get('%', -100)
        
        op_sin = self.grammar.token_to_id.get('sin', -100)
        op_cos = self.grammar.token_to_id.get('cos', -100)
        op_tan = self.grammar.token_to_id.get('tan', -100)
        op_asin = self.grammar.token_to_id.get('S', -100)
        op_acos = self.grammar.token_to_id.get('C', -100)
        op_atan = self.grammar.token_to_id.get('T', -100)
        op_exp = self.grammar.token_to_id.get('e', -100) # 'e' is the operator token
        op_log = self.grammar.token_to_id.get('log', -100)
        op_sqrt = self.grammar.token_to_id.get('sqrt', -100)
        op_abs = self.grammar.token_to_id.get('abs', -100)
        op_neg = self.grammar.token_to_id.get('neg', -100)
        
        op_fact = self.grammar.token_to_id.get('!', -100)
        op_floor = self.grammar.token_to_id.get('_', -100)
        op_gamma = self.grammar.token_to_id.get('g', -100)
        
        # Cache Variable IDs
        # We know self.grammar.active_variables list
        var_ids = [self.grammar.token_to_id.get(v, -100) for v in self.grammar.active_variables]
        # x0 -> index 0, x1 -> index 1...
        # Also 'x' usually maps to x0
        id_x_legacy = self.grammar.token_to_id.get('x', -100)

        for i in range(L):
            token = pop_expanded[:, i]
            active_mask = (token != PAD_ID)
            if not active_mask.any(): continue
            
            # --- 1. Push ---
            push_vals = torch.zeros(eff_B, device=self.device, dtype=torch.float64)
            is_operand = torch.zeros(eff_B, dtype=torch.bool, device=self.device)
            
            # Variables
            # Check legacy 'x'
            mask = (token == id_x_legacy)
            if mask.any():
                push_vals[mask] = x_expanded[mask, 0]
                is_operand = is_operand | mask
                
            # Check x0, x1, x2...
            for v_idx, vid in enumerate(var_ids):
                mask = (token == vid)
                if mask.any():
                    # If inputs have enough columns, use them. If not, fallback to 0 or error?
                    # We assume x_expanded shape matches grammar requirements.
                    if v_idx < x_expanded.shape[1]:
                        push_vals[mask] = x_expanded[mask, v_idx]
                        is_operand = is_operand | mask
            
            mask = (token == id_pi)
            if mask.any():
                push_vals[mask] = pi_val
                is_operand = is_operand | mask
            

            mask = (token == id_e)
            if mask.any():
                push_vals[mask] = e_val
                is_operand = is_operand | mask
                
            mask = (token == id_C)
            if mask.any():
                if const_expanded is not None:
                     # Gather constants based on current counter
                     # const_expanded: [eff_B, MaxC]
                     # const_counters: [eff_B]
                     # We need to clamp counter to MaxC-1 to avoid error, though valid RPN shouldn't exceed
                     safe_idx = torch.clamp(const_counters, 0, const_expanded.shape[1]-1)
                     
                     c_vals = const_expanded.gather(1, safe_idx.unsqueeze(1)).squeeze(1)
                     push_vals[mask] = c_vals[mask]
                     
                     # Increment counter where C was used
                     const_counters[mask] += 1
                else:
                     push_vals[mask] = 1.0 
                     
                is_operand = is_operand | mask
            
            # Literals
            for val_str in ['1', '2', '3', '5']:
                vid = self.grammar.token_to_id.get(val_str, -999)
                mask = (token == vid)
                if mask.any():
                    push_vals[mask] = float(val_str)
                    is_operand = is_operand | mask
                    

            if is_operand.any():
                safe_sp = torch.clamp(sp, 0, MAX_STACK-1)
                # Out-of-place scatter for autograd safety
                stack = stack.scatter(1, safe_sp.unsqueeze(1), push_vals.unsqueeze(1))
                sp = sp + is_operand.long()
                
            # --- 2. Binary ---
            is_binary = (token == op_add) | (token == op_sub) | (token == op_mul) | (token == op_div) | (token == op_pow) | (token == op_mod)
            valid_op = is_binary & (sp >= 2)
            
            # Check stack underflow for binary
            has_error = has_error | (is_binary & (sp < 2))
            
            if valid_op.any():
                idx_b = torch.clamp(sp - 1, 0, MAX_STACK - 1).unsqueeze(1)
                val_b = stack.gather(1, idx_b).squeeze(1)
                
                idx_a = torch.clamp(sp - 2, 0, MAX_STACK - 1).unsqueeze(1)
                val_a = stack.gather(1, idx_a).squeeze(1)
                
                res = torch.zeros_like(val_a)
                
                mask = (token == op_add) & valid_op
                if mask.any(): res[mask] = val_a[mask] + val_b[mask]
                
                mask = (token == op_sub) & valid_op
                if mask.any(): res[mask] = val_a[mask] - val_b[mask]
                
                mask = (token == op_mul) & valid_op
                if mask.any(): res[mask] = val_a[mask] * val_b[mask]
                
                mask = (token == op_div) & valid_op
                if mask.any(): 
                    denom = val_b[mask]
                    # C++: if (fabs(right) < 1e-9) { result = GPU_MAX_DOUBLE; }
                    # We implement parity:
                    bad_denom = denom.abs() < 1e-9
                    
                    # We compute safe division where possible
                    safe_denom = torch.where(bad_denom, torch.tensor(1.0, device=self.device, dtype=torch.float64), denom)
                    out_div = val_a[mask] / safe_denom
                    
                    # Apply penalty for bad denom
                    # We use 1e300 as GPU_MAX_DOUBLE proxy (or just 1e15 to avoid inf issues in float64?)
                    # C++ uses DBL_MAX typically which is ~1e308. 
                    out_div[bad_denom] = 1e300
                    res[mask] = out_div
                    
                mask = (token == op_mod) & valid_op
                if mask.any():
                    denom = val_b[mask]
                    bad_denom = denom.abs() < 1e-9
                    safe_denom = torch.where(bad_denom, torch.tensor(1.0, device=self.device, dtype=torch.float64), denom)
                    # C++: fmod
                    out_mod = torch.fmod(val_a[mask], safe_denom)
                    out_mod[bad_denom] = 1e300
                    res[mask] = out_mod
                    
                mask = (token == op_pow) & valid_op
                if mask.any():
                    base = val_a[mask]
                    expon = val_b[mask]
                    # Protect: if base < 0 and exponent not int, nan in float.
                    # Also 0^0, etc.
                    # We can use torch.pow but replace NaNs if they occur
                    p = torch.pow(base, expon)
                    bad_pow = torch.isnan(p) | torch.isinf(p)
                    p[bad_pow] = 1e300
                    res[mask] = p
                
                write_pos = torch.clamp(sp - 2, 0, MAX_STACK-1)
                current_at_pos = stack.gather(1, write_pos.unsqueeze(1)).squeeze(1)
                final_write_val = torch.where(valid_op, res, current_at_pos)
                
                # Out-of-place scatter
                stack = stack.scatter(1, write_pos.unsqueeze(1), final_write_val.unsqueeze(1))
                sp = sp - valid_op.long()
                
            # --- 3. Unary ---
            is_unary = (token == op_sin) | (token == op_cos) | (token == op_tan) | \
                       (token == op_asin) | (token == op_acos) | (token == op_atan) | \
                       (token == op_exp) | (token == op_log) | \
                       (token == op_sqrt) | (token == op_abs) | (token == op_neg) | \
                       (token == op_fact) | (token == op_floor) | (token == op_gamma)
            valid_op = is_unary & (sp >= 1)
            
            # Check stack underflow for unary
            has_error = has_error | (is_unary & (sp < 1))
            
            if valid_op.any():
                idx_a = torch.clamp(sp - 1, 0, MAX_STACK - 1).unsqueeze(1)
                val_a = stack.gather(1, idx_a).squeeze(1)
                res = torch.zeros_like(val_a)
                
                mask = (token == op_sin) & valid_op
                if mask.any(): res[mask] = torch.sin(val_a[mask])
                
                mask = (token == op_cos) & valid_op
                if mask.any(): res[mask] = torch.cos(val_a[mask])
                
                mask = (token == op_log) & valid_op
                if mask.any(): 
                    # C++: (val <= 1e-9) ? GPU_MAX_DOUBLE : log(val)
                    # We use a large value for error
                    inp = val_a[mask]
                    safe_mask = inp > 1e-9
                    # Where unsafe, we put a huge value. But we must set res.
                    # We compute log everywhere but replace bad ones? Or select?
                    out = torch.full_like(inp, 1e300) # GPU_MAX_DOUBLE proxy
                    safe_inp = torch.where(safe_mask, inp, torch.tensor(1.0, device=self.device, dtype=torch.float64))
                    val_log = torch.log(safe_inp)
                    out[safe_mask] = val_log[safe_mask]
                    res[mask] = out
                
                mask = (token == op_exp) & valid_op
                if mask.any(): 
                    # C++: (val > 700.0) ? GPU_MAX_DOUBLE : exp(val)
                    inp = val_a[mask]
                    safe_mask = inp <= 700.0
                    out = torch.full_like(inp, 1e300)
                    safe_inp = torch.where(safe_mask, inp, torch.tensor(0.0, device=self.device, dtype=torch.float64))
                    val_exp = torch.exp(safe_inp)
                    out[safe_mask] = val_exp[safe_mask]
                    res[mask] = out
                
                mask = (token == op_sqrt) & valid_op
                if mask.any(): res[mask] = torch.sqrt(val_a[mask].abs())
                
                mask = (token == op_abs) & valid_op
                if mask.any(): res[mask] = torch.abs(val_a[mask])
                
                mask = (token == op_neg) & valid_op
                if mask.any(): res[mask] = -val_a[mask]
                
                mask = (token == op_tan) & valid_op
                if mask.any(): res[mask] = torch.tan(val_a[mask])
                
                mask = (token == op_asin) & valid_op
                if mask.any(): 
                    # C++ protected: asin(clamp(x, -1, 1)) (actually S op code)
                    # But if we want standard behavior or protected?
                    # The C++ code for 'asin' (op 'S' in tree string, but 'asin' in kernel?) 
                    # Kernel uses standard asin but our engine usually protects domain.
                    # Let's use protection [-1, 1]
                    res[mask] = torch.asin(torch.clamp(val_a[mask], -1.0, 1.0))
                
                mask = (token == op_acos) & valid_op
                if mask.any(): res[mask] = torch.acos(torch.clamp(val_a[mask], -1.0, 1.0))
                
                mask = (token == op_atan) & valid_op
                if mask.any(): res[mask] = torch.atan(val_a[mask])

                mask = (token == op_floor) & valid_op
                if mask.any(): res[mask] = torch.floor(val_a[mask])
                
                mask = (token == op_fact) & valid_op
                if mask.any():
                    # C++: (val < 0 || val > 170.0) ? GPU_MAX_DOUBLE : tgamma(val + 1.0);
                    inp = val_a[mask]
                    # Safe range for fact: > -1 (approx) and <= 170
                    is_safe = (inp > -1.0) & (inp <= 170.0)
                    
                    out = torch.full_like(inp, 1e300)
                    
                    # Compute only for safe
                    if is_safe.any():
                         safe_inp = inp[is_safe]
                         # tgamma(n+1)
                         val_computed = torch.special.gamma(safe_inp + 1.0)
                         out[is_safe] = val_computed
                         # Check result not inf
                         is_inf_g = torch.isinf(out[is_safe])
                         if is_inf_g.any():
                             # If gamma produced inf within safe range bounds (unlikely if <=170), clamp
                             # Actually 170! is <dbl_max, so should be fine.
                             pass
                             
                    res[mask] = out

                mask = (token == op_gamma) & valid_op
                if mask.any():
                     # C++: (val <= -1.0) ? GPU_MAX_DOUBLE : lgamma(val + 1.0); 
                     inp = val_a[mask]
                     is_safe = (inp > -1.0)
                     
                     out = torch.full_like(inp, 1e300)
                     
                     if is_safe.any():
                         safe_inp = inp[is_safe]
                         val_computed = torch.special.gammaln(safe_inp + 1.0)
                         out[is_safe] = val_computed
                         
                     res[mask] = out

                write_pos = torch.clamp(sp - 1, 0, MAX_STACK-1)
                current_at_pos = stack.gather(1, write_pos.unsqueeze(1)).squeeze(1)
                final_write_val = torch.where(valid_op, res, current_at_pos)
                
                # Out-of-place scatter
                stack = stack.scatter(1, write_pos.unsqueeze(1), final_write_val.unsqueeze(1))

        
        is_valid = (sp == 1)
        final_preds = stack[:, 0]
        final_preds = torch.where(is_valid, final_preds, torch.tensor(float('nan'), device=self.device, dtype=torch.float64))
        preds_matrix = final_preds.view(B, D)
        target_matrix = y_target.unsqueeze(0).expand(B, -1)
        mse = torch.mean((preds_matrix - target_matrix)**2, dim=1)
        rmse = torch.sqrt(torch.where(torch.isnan(mse), torch.tensor(1e300, device=self.device, dtype=torch.float64), mse))
        return rmse

    def evaluate_differentiable(self, population: torch.Tensor, constants: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """
        Evaluates population with Autograd enabled to return Loss [PopSize].
        Supports backprop to constants.
        """
        # Ensure model is in training mode if using Layers (not used here)
        # Ensure we don't detach anything inside _run_vm
        
        final_preds, sp, has_error = self._run_vm(population, x, constants)
        is_valid = (sp == 1) & (~has_error)
        
        # Reshape to [B, D]
        valid_matrix = is_valid.view(population.shape[0], x.shape[0])
        preds = final_preds.view(population.shape[0], x.shape[0])
        target = y_target.unsqueeze(0).expand_as(preds)
        
        sq_err = (preds - target)**2
        
        # Clamp squared error to prevent Inf in gradients for extremely bad individuals
        sq_err = torch.clamp(sq_err, max=1e10)
        
        # Mask invalid to 0 (ignored in gradient)
        masked_sq_err = torch.where(valid_matrix, sq_err, torch.tensor(0.0, device=self.device, dtype=torch.float64))
        
        loss = masked_sq_err.mean(dim=1)
        return loss

    def optimize_constants(self, population: torch.Tensor, constants: torch.Tensor, 
                          x: torch.Tensor, y_target: torch.Tensor, 
                          steps: int = 10, lr: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Optimizes constants in-place using Gradient Descent (Adam).
        Returns: optimized_constants, new_fitness
        """
        # We must clone start point and enable grad
        opt_const = constants.clone().detach().requires_grad_(True)
        
        # Use Adam for stability
        optimizer = torch.optim.Adam([opt_const], lr=lr)
        
        for i in range(steps):
            optimizer.zero_grad()
            
            # Forward
            try:
                loss = self.evaluate_differentiable(population, opt_const, x, y_target)
                total_loss = loss.sum()
                total_loss.backward()
                optimizer.step()
            except Exception as e:
                # Fallback if autograd fails (inplace op error or size mismatch)
                # print(f"  [DEBUG] Opt Fail Step {i}: {e}")
                return constants, self.evaluate_batch(population, x, y_target, constants)

        # Final Eval (Detached)
        with torch.no_grad():
             final_fit = self.evaluate_batch(population, x, y_target, opt_const)
             
        return opt_const.detach(), final_fit 

    def evaluate_batch_full(self, population: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor, constants: torch.Tensor = None) -> torch.Tensor:
        """
        Returns full error matrix [Pop, D]. 
        Used for Lexicase Selection.
        """
        B, L = population.shape
        D = x.shape[0]
        
        final_preds, sp = self._run_vm(population, x, constants)
        is_valid = (sp == 1)
        
        # Penalize invalid/nan/inf with 1e300
        final_preds = torch.where(is_valid & ~torch.isnan(final_preds) & ~torch.isinf(final_preds), 
                                  final_preds, 
                                  torch.tensor(1e300, device=self.device, dtype=torch.float64))
        
        preds_matrix = final_preds.view(B, D)
        target_matrix = y_target.unsqueeze(0).expand(B, -1)
        abs_err = torch.abs(preds_matrix - target_matrix)
        
        # Guard against Inf in abs_err (e.g. pred - target where one is huge)
        abs_err = torch.where(torch.isnan(abs_err) | torch.isinf(abs_err), 
                              torch.tensor(1e300, device=self.device, dtype=torch.float64), 
                              abs_err)
        return abs_err

    def compute_case_weights(self, errors: torch.Tensor) -> torch.Tensor:
        """
        Compute case weights based on difficulty (variance of errors across population).
        
        Cases with higher variance are considered harder and get higher weights.
        
        Args:
            errors: [PopSize, n_cases] error matrix
            
        Returns:
            [n_cases] weights normalized to sum to 1
        """
        # Variance across population per case
        case_variance = torch.var(errors, dim=0)
        
        # Normalize to weights (higher variance -> higher weight)
        weights = case_variance / (case_variance.sum() + 1e-9)
        
        return weights

    def weighted_rmse(self, errors: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """
        Compute weighted RMSE across cases.
        
        Args:
            errors: [PopSize, n_cases] absolute error matrix
            weights: [n_cases] optional case weights (default: uniform)
            
        Returns:
            [PopSize] weighted RMSE per individual
        """
        if weights is None or not GpuGlobals.USE_WEIGHTED_FITNESS:
            # Standard RMSE
            return torch.sqrt((errors ** 2).mean(dim=1))
        
        # Weighted mean squared error
        weighted_mse = (errors ** 2 * weights.unsqueeze(0)).sum(dim=1)
        return torch.sqrt(weighted_mse)

    def deterministic_crowding(self, parents: torch.Tensor, offspring: torch.Tensor, p_fit: torch.Tensor, o_fit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Deterministic Crowding Selection.
        Offspring competes with its direct parent.
        Returns: New Population, New Fitness
        """
        # Assuming Offspring[i] is child of Parent[i] (or derived from).
        # In our crossover, we put children back into p1_idx and p2_idx.
        # So Offspring[idx] effectively replaces Parent[idx].
        # We just need to know if it is BETTER.
        
        # Minimization (lower fitness is better)
        replace = o_fit < p_fit
        
        # Update Population
        mask = replace.unsqueeze(1).expand_as(parents)
        new_pop = torch.where(mask, offspring, parents)
        
        # Update Fitness
        new_fit = torch.where(replace, o_fit, p_fit)
        
        return new_pop, new_fit
        
    def tournament_selection_island(self, population: torch.Tensor, fitness: torch.Tensor, n_islands: int) -> torch.Tensor:
        """
        Performs tournament selection WITHIN islands.
        population: [PopSize, L]
        fitness: [PopSize]
        n_islands: Number of islands
        """
        B, L = population.shape
        island_size = B // n_islands
        
        # Reshape to [Islands, IslandSize, L]
        # But we need indices.
        
        # We want to select B parents, but selection must be local.
        # Run standard tournament but restrict indices?
        
        # Vectorized approach:
        # Perform tournament on [Islands, IslandSize] tensor.
        
        pop_view = population.view(n_islands, island_size, L)
        fit_view = fitness.view(n_islands, island_size)
        
        # Tournament indices [Islands, IslandSize]
        # Random opponents within same island
        tourn_size = 5
        idx = torch.randint(0, island_size, (n_islands, island_size, tourn_size), device=self.device)
        
        # Gather fitness of opponents
        # fit_view: [I, S]
        # idx: [I, S, T]
        # We need to gather dim 1.
        # gather needs index same dim as input except gathered dim?
        # expanded_fit: [I, 1, S] -> expand to [I, S, T]? No.
        
        # We use simple trick: add offset to indices to flatten them back to global
        offsets = torch.arange(n_islands, device=self.device) * island_size
        offsets = offsets.view(n_islands, 1, 1)
        
        global_idx = idx + offsets # [I, S, T]
        
        # Flatten to get fitness
        flat_idx = global_idx.view(-1)
        flat_fits = fitness[flat_idx].view(n_islands, island_size, tourn_size)
        
        # Min
        best_vals, best_cols = torch.min(flat_fits, dim=2) # [I, S]
        
        # Access original indices
        # We want the index of the winner
        # best_cols is [0..T-1]
        # We need actual index from `idx`
        winner_local_idx = idx.gather(2, best_cols.unsqueeze(2)).squeeze(2) # [I, S]
        
        # Global winner index
        winner_global_idx = winner_local_idx + offsets.squeeze(2) # [I, S]
        
        return winner_global_idx.view(-1)
        
    def lexicase_selection(self, population: torch.Tensor, errors: torch.Tensor, n_select: int) -> torch.Tensor:
        """
        Selects n_select parents using Tournament Lexicase Selection.
        errors: [PopSize, n_cases] (Absolute Error)
        """
        # Lexicase is slow if running on full population for every selection.
        # "Tournament Lexicase": Pick random subset, run lexicase on it to find 1 winner. Repeat.
        
        # Optimized implementation:
        # We need n_select winners.
        # For each winner:
        # 1. Pick pool (size ~50?)
        # 2. Shuffle cases
        # 3. Filter loop
        
        # Since we cannot easily loop inside tensor ops, we might need a custom kernel or CPU loop.
        # Lexicase is inherently sequential on cases.
        # CPU Loop over n_select is feasible if n_select is not huge (e.g. 1000).
        
        pop_size, n_cases = errors.shape
        pool_size = 50
        
        selected_indices = []
        
        # Errors to CPU for logic
        errors_cpu = errors.detach().cpu().numpy()
        
        for _ in range(n_select):
            # 1. Pool
            candidates = np.random.randint(0, pop_size, pool_size)
            
            # 2. Shuffle cases
            cases = np.random.permutation(n_cases)
            
            active_cands = candidates
            
            for case_idx in cases:
                # Get errors for active candidates at this case
                # errors_cpu[active_cands, case_idx]
                case_errs = errors_cpu[active_cands, case_idx]
                min_err = np.min(case_errs)
                
                # Epsilon (MAD or simple)
                epsilon = max(min_err * 0.1, 1e-9)
                
                # Filter
                survivors_mask = case_errs <= (min_err + epsilon)
                active_cands = active_cands[survivors_mask]
                
                if len(active_cands) == 1:
                    break
            
            # Pick random survivor
            winner = np.random.choice(active_cands)
            selected_indices.append(winner)
            
        return population[selected_indices]

        

    def _generate_random_population(self, size: int) -> torch.Tensor:
        """
        Helper to generate random RPN population of given size.
        """
        formulas = []
        # Generate full population (slower but ensures diversity)
        for _ in range(size):
            try:
                # Generate random valid tree
                tree = ExpressionTree.generate_random(max_depth=GpuGlobals.MAX_TREE_DEPTH_INITIAL, num_variables=self.num_variables)
                formulas.append(tree.get_infix())
            except:
                formulas.append("x0")
        
        # Convert to RPN
        return self.infix_to_rpn(formulas)

    def initialize_population(self) -> torch.Tensor:
        """
        Generates a population of VALID random formulas.
        """
        return self._generate_random_population(self.pop_size)

    def cataclysm_population(self, population: torch.Tensor, constants: torch.Tensor, fitness_rmse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hard Reset: Keep Top 10% Elites, replace the rest with new random individuals.
        Called when diversity collapses (too many duplicates).
        """
        B = population.shape[0]
        n_elites = int(B * 0.10)
        n_random = B - n_elites
        
        # Sort by fitness (RMSE ascending is better)
        sorted_indices = torch.argsort(fitness_rmse)
        elite_indices = sorted_indices[:n_elites]
        
        # Keep Elites
        elites = population[elite_indices]
        elite_consts = constants[elite_indices]
        
        # Generate fresh randoms
        new_pop = self._generate_random_population(n_random)
        new_consts = torch.zeros((n_random, constants.shape[1]), device=self.device, dtype=torch.float64)
        
        # Combine
        final_pop = torch.cat([elites, new_pop], dim=0)
        final_consts = torch.cat([elite_consts, new_consts], dim=0)
        
        return final_pop, final_consts

    def detect_patterns(self, targets: List[float]) -> List[str]:
        """
        Detects simple patterns (Arithmetic, Geometric) in 1D targets.
        Returns a list of seed formulas.
        """
        if len(targets) < 3: return []
        
        seeds = []
        
        # 1. Arithmetic: y = a + d*x  (assuming x=0, 1, 2...)
        # We need to know X to be sure, but let's assume standard index for simple detection
        # or just check diffs.
        
        diffs = np.diff(targets)
        if np.allclose(diffs, diffs[0], atol=1e-5):
            d = diffs[0]
            a = targets[0] # assuming x0=0 start? 
            # If x starts at 1, then y = a + d*(x-1) = (a-d) + d*x
            # We construct a generic candidate 'C + C*x0'
            # We can just let the optimizer find constants if we give the structure.
            seeds.append("(C + (C * x0))") 
            
        # 2. Geometric: y = a * r^x
        # Check ratios
        if not np.any(np.abs(targets) < 1e-9):
            ratios = targets[1:] / targets[:-1]
            if np.allclose(ratios, ratios[0], atol=1e-5):
                seeds.append("(C * (C ^ x0))")
                
        # 3. Constant
        if np.allclose(targets, targets[0], atol=1e-5):
             seeds.append("C")
             
        # 4. Fibonacci-ish? (Last 2 sum)
        # 5. Sinusoidal?
        
        return seeds

    def run(self, x_values: List[float], y_targets: List[float], seeds: List[str], timeout_sec=10, callback=None) -> Optional[str]:
        """
        Main evolutionary loop.
        """
        start_time = time.time()
        
        # 1. Setup Data
        if GpuGlobals.USE_LOG_TRANSFORMATION:
            print("Info: Log Transformation is ON (Target = ln(Y)).")
            y_np = np.array(y_targets)
            x_np = np.array(x_values)
            mask = y_np > 1e-9 # Parity with C++ log protection
            if not mask.all():
                print(f"Warning: Filtering out {(~mask).sum()} zero or negative data points for log transformation.")
                y_np = y_np[mask]
                x_np = x_np[mask]
            y_targets = np.log(y_np).tolist()
            x_values = x_np.tolist()

        x_t = torch.tensor(x_values, device=self.device, dtype=torch.float64)
        y_t = torch.tensor(y_targets, device=self.device, dtype=torch.float64)
        
        # Flatten only if strictly 1 variable and input is weirdly shaped?
        # If num_variables > 1, we expect x_t to be [N, Vars]
        if self.num_variables == 1:
             if x_t.ndim > 1: x_t = x_t.flatten()
        
        if y_t.ndim > 1: y_t = y_t.flatten()

        if self.num_variables == 1 and x_t.ndim == 1:
            x_t = x_t.unsqueeze(0) # [1, N]
        
        if y_t.ndim == 1: 
            y_t = y_t.unsqueeze(0) # [1, N]

        # NOTE: This ensures correct broadcasting against [B, N] stack or results.
        
        # The Sniper
        sniper_res = self.sniper.run(x_values, y_targets)
        if sniper_res: return sniper_res
        
        print("[GPU Worker] Initializing Tensor Population...")
        
        # 2. Init Population
        population = self.initialize_population()
        # population[:, 0] = torch.randint(...) # No longer needed, as we have valid RPNs

 
        
        # Seeds
        if seeds:
            seed_tensors = self.infix_to_rpn(seeds)
            k_seeds = seed_tensors.shape[0]
            if k_seeds > 0:
                population[:k_seeds] = seed_tensors
        
        # --- Pattern Detection ---
        # Detect standard sequences (Arithmetic, Geometric)
        pattern_seeds = self.detect_patterns(y_targets)
        if pattern_seeds:
            print(f"[GPU Worker] Detected patterns: {pattern_seeds}")
            pat_tensors = self.infix_to_rpn(pattern_seeds)
            k_pats = pat_tensors.shape[0]
            if k_pats > 0:
                 # Insert after user seeds
                 offset = len(seeds) if seeds else 0
                 population[offset:offset+k_pats] = pat_tensors

        pop_constants = torch.randn(self.pop_size, self.max_constants, device=self.device, dtype=torch.float64)
        
        # Stats
        best_rmse = float('inf')
        best_rpn = None
        best_consts_vec = None
        
        stagnation_counter = 0
        current_mutation_rate = GpuGlobals.BASE_MUTATION_RATE
        
        generations = 0
        COMPLEXITY_PENALTY = GpuGlobals.COMPLEXITY_PENALTY
        max_generations = GpuGlobals.GENERATIONS

        # Loop until: fitness ~0, OR max generations, OR timeout
        while generations < max_generations:
            # Check timeout (optional, set timeout_sec=None to disable)
            if timeout_sec and (time.time() - start_time) >= timeout_sec:
                print(f"[GPU] Timeout after {generations} generations")
                break
                
            generations += 1
            

            # Eval (Fast Scan)
            # print(f"DEBUG: Eval Batch Pop={population.shape}, X={x_t.shape}, Y={y_t.shape}")
            fitness_rmse = self.evaluate_batch(population, x_t, y_t, pop_constants)
            
            # --- Constant Optimization (Top K) ---
            # Optimize top 200 candidates to refine their constants
            k_opt = min(self.pop_size, 200)
            
            # Find candidates (using penalized fitness or raw rmse?)
            # Raw RMSE is better for optimization target
            _, top_idx = torch.topk(fitness_rmse, k_opt, largest=False)
            
            # Extract subset
            opt_pop = population[top_idx]
            opt_consts = pop_constants[top_idx]
            
            # Optimize (Gradient Descent)
            # Use fewer steps to keep speed up? 10 is fine.
            # print(f"DEBUG: Optimization Start Shapes: Pop={opt_pop.shape}, Const={opt_consts.shape}, X={x_t.shape}")
            refined_consts, refined_mse = self.optimize_constants(
                opt_pop, opt_consts, x_t, y_t, steps=10, lr=0.1
            )
            # print(f"DEBUG: Optimization End Shapes: Refined={refined_consts.shape}, MSE={refined_mse.shape}")
            
            # Update population constants
            pop_constants[top_idx] = refined_consts
            
            # Update fitness for optimized individuals (optional but good for accurate tracking)
            # refined_mse is actually RMSE from the function
            fitness_rmse[top_idx] = refined_mse
            
            # Re-evaluate penalties? Length doesn't change.
            # But we can just leave it for next gen or update fitness_penalized here.
            # Let's update Penalized so Elitism picks the improved versions instantly.
            # We need lengths for these
            # lengths is [PopSize], so we pick top_idx
            # fitness_penalized[top_idx] = refined_mse * (1.0 + lengths[top_idx] * COMPLEXITY_PENALTY)
            
            # Run Adam
            # refined_consts, refined_rmse = self.optimize_constants(opt_pop, opt_consts, x_t, y_t, steps=15)
            
            # Update Population
            # We must update the original tensors.
            # 1. Update Constants
            # pop_constants[top_idx] = refined_consts
            # 2. Update Fitness Scores (Evaluation was implicit in optimize)
            # But wait, fitness_rmse is [PopSize].
            # We update the scores.
            # fitness_rmse[top_idx] = refined_rmse
            
            # --- Selection ---
            lengths = (population != PAD_ID).sum(dim=1).float()
            fitness_penalized = fitness_rmse * (1.0 + COMPLEXITY_PENALTY * lengths) + lengths * 1e-6
            
            # --- Tarpeian Bloat Control ---
            fitness_penalized = self.tarpeian_control(population, fitness_penalized)
            
            # Select Best
            min_rmse, min_idx = torch.min(fitness_rmse, dim=0)
            if min_rmse.item() < best_rmse:
                best_rmse = min_rmse.item()
                best_rpn = population[min_idx].clone()
                best_consts_vec = pop_constants[min_idx].clone()
                best_island_idx = (min_idx.item() // self.island_size) 
                
                if callback:
                    callback(generations, best_rmse, best_rpn, best_consts_vec, True, best_island_idx)
                
                stagnation_counter = 0
                current_mutation_rate = GpuGlobals.BASE_MUTATION_RATE
            else:
                stagnation_counter += 1
            
            if callback and (generations % GpuGlobals.PROGRESS_REPORT_INTERVAL == 0 or generations == 1) and best_rpn is not None:
                 callback(generations, best_rmse, best_rpn, best_consts_vec, False, -1)

            # --- Island Migration ---
            if self.n_islands > 1 and generations % GpuGlobals.MIGRATION_INTERVAL == 0:
                population, pop_constants = self.migrate_islands(population, pop_constants, fitness_rmse)

            # Cataclysm
            if stagnation_counter >= GpuGlobals.STAGNATION_LIMIT:
                 saved_best_rpn = best_rpn.clone()
                 saved_best_c = best_consts_vec.clone()
                 population = self.initialize_population()
                 pop_constants = torch.randn(self.pop_size, self.max_constants, device=self.device, dtype=torch.float64)
                 population[0] = saved_best_rpn
                 pop_constants[0] = saved_best_c
                 stagnation_counter = 0
                 current_mutation_rate = GpuGlobals.BASE_MUTATION_RATE
                 continue
            
            # --- Dynamic Mutation Rate ---
            if stagnation_counter > 10:
                current_mutation_rate = min(0.4, GpuGlobals.BASE_MUTATION_RATE + (stagnation_counter - 10) * 0.01)
            else:
                current_mutation_rate = GpuGlobals.BASE_MUTATION_RATE


            # --- NEW ADVANCED EVOLUTION STEP ---
            # 1. Elitism
            # 2. Crossover (Lexicase Sel)
            # 3. Mutation (Tournament Sel)
            # 4. Uniqueness Check
            
            next_pop_list = []
            next_const_list = []
            
            # 1. Elitism (Top 5% using Pareto or Fitness)
            k_elite = max(1, int(self.pop_size * 0.05))
            
            if GpuGlobals.USE_PARETO_SELECTION:
                # Use NSGA-II to select elite individuals balancing error vs complexity
                complexity = lengths  # Tree size as complexity
                elite_idx = self.pareto.select(population, fitness_rmse, complexity, k_elite)
            else:
                # Standard fitness-based elitism
                _, elite_idx = torch.topk(fitness_penalized, k_elite, largest=False)
            
            elites = population[elite_idx]
            elites_c = pop_constants[elite_idx]
            next_pop_list.append(elites)
            next_const_list.append(elites_c)
            
            remaining_slots = self.pop_size - k_elite
            
            # 2. Crossover (Using Lexicase if costly or standard if fast?)
            # Lexicase is costly. Let's compute FULL errors only if using Lexicase.
            # Use Lexicase for Crossover Parents (Standard GP practice)
            
            # 2. Crossover Parents (GPU Tournament for Speed)
            # 2. Crossover Parents (GPU Tournament for Speed)
            n_crossover = int(remaining_slots * GpuGlobals.DEFAULT_CROSSOVER_RATE)
            n_mutation = remaining_slots - n_crossover
            
            if n_crossover > 0:
                idx_cross = torch.randint(0, self.pop_size, (n_crossover, GpuGlobals.DEFAULT_TOURNAMENT_SIZE), device=self.device)
                best_in_tourn = torch.argmin(fitness_penalized[idx_cross], dim=1)
                global_idx_cross = idx_cross.gather(1, best_in_tourn.unsqueeze(1)).squeeze(1)
                
                # We need PAIRS of parents. This logic selects N individuals.
                # crossover_population internally shuffles and pairs them.
                parents_cross = population[global_idx_cross]
                consts_cross = pop_constants[global_idx_cross]
                
                # Perform crossover (Vectorized)
                off_cross = self.crossover_population(parents_cross, crossover_rate=1.0) # Rate 1.0 because we already selected size
                next_pop_list.append(off_cross)
                next_const_list.append(consts_cross)
            
            # 3. Mutation Parents (Tournament)
            if n_mutation > 0:
                idx_mut = torch.randint(0, self.pop_size, (n_mutation, GpuGlobals.DEFAULT_TOURNAMENT_SIZE), device=self.device)
                best_in_tourn = torch.argmin(fitness_penalized[idx_mut], dim=1)
                global_idx_mut = idx_mut.gather(1, best_in_tourn.unsqueeze(1)).squeeze(1)
                
                parents_mut = population[global_idx_mut]
                consts_mut = pop_constants[global_idx_mut]
                
                off_mut = self.mutate_population(parents_mut, current_mutation_rate)
                next_pop_list.append(off_mut)
                next_const_list.append(consts_mut)
            
            
            # Concatenate
            next_pop = torch.cat(next_pop_list, dim=0)
            next_c = torch.cat(next_const_list, dim=0)
            
            population = next_pop[:self.pop_size]
            pop_constants = next_c[:self.pop_size]

            # --- Deduplication (Aggressive: Every generation to force diversity) ---
            if GpuGlobals.PREVENT_DUPLICATES and generations % 1 == 0:
                population, pop_constants, n_dups = self.deduplicate_population(population, pop_constants)
                # Silent - only log if many duplicates
                if n_dups > self.pop_size * 0.1:
                    print(f"[GPU] Removed {n_dups} duplicates (Fresh Randoms Injected)")
            
            # Debug: Report Valid Count
            if generations % 5 == 0:
                 valid_cnt = (fitness_rmse < 1e9).sum().item()
                 print(f"[GPU] Gen {generations}: Valid Individuals = {valid_cnt}/{self.pop_size}")
                    
                # TRIGGER CATACLYSM if > 90% are duplicates - REMOVED (Redundant with Fresh Random Injection)
                # if n_dups > self.pop_size * 0.9 and generations > 20: 
                #     print(f"!!! CATACLYSM TRIGGERED (Duplicates: {n_dups}/{self.pop_size}) !!!")
                #     print("!!! Resetting 90% of population with fresh DNA !!!")
                #     population, pop_constants = self.cataclysm_population(population, pop_constants, fitness_rmse)

            # --- Simplification (Reduced Frequency: Every 500 generations) ---
            if GpuGlobals.USE_SIMPLIFICATION and generations % 500 == 0:
                # Simplify top 50 individuals
                population, pop_constants, n_simp = self.simplify_population(population, pop_constants, top_k=50)
                if n_simp > 0 and callback:
                    print(f"[GPU] Simplified {n_simp} expressions")

            # --- Pattern Memory (DISABLED FOR SPEED TEST) ---
            # Record successful subtrees from current population
            # self.pattern_memory.record_subtrees(population, fitness_rmse, self.grammar)
            
            # Inject patterns periodically
            # if generations % GpuGlobals.PATTERN_INJECT_INTERVAL == 0:
            #     population, pop_constants, n_inj = self.pattern_memory.inject_into_population(
            #         population, pop_constants, self.grammar, 
            #         percent=GpuGlobals.PATTERN_INJECT_PERCENT
            #     )

            # --- Local Search (DISABLED FOR SPEED TEST) ---
            # if generations % 100 == 0:
            #     population, pop_constants = self.local_search(
            #         population, pop_constants, x_t, y_t, 
            #         top_k=10, attempts=GpuGlobals.LOCAL_SEARCH_ATTEMPTS
            #     )
            
            if best_rmse < 1e-7:
                 return self.rpn_to_infix(best_rpn, best_consts_vec)
                 
        if best_rpn is not None:
             return self.rpn_to_infix(best_rpn, best_consts_vec)
        return None

    def run(self, x_values, y_values, seeds: List[str] = None, timeout_sec: int = 10):
        """
        Full Evolutionary Loop (API compatible with GPEngine).
        
        Args:
            x_values: Input data (numpy or list)
            y_values: Target data (numpy or list)
            seeds: Optional list of infix formulas to inject
            timeout_sec: Maximum execution time in seconds
            
        Returns:
            Best formula string found (infix)
        """
        start_time = time.time()
        
        # 1. Data Conversion
        if not isinstance(x_values, torch.Tensor):
            x_t = torch.tensor(x_values, dtype=torch.float64, device=self.device)
        else:
            x_t = x_values.to(self.device).to(torch.float64)
            
        if not isinstance(y_values, torch.Tensor):
            y_t = torch.tensor(y_values, dtype=torch.float64, device=self.device)
        else:
            y_t = y_values.to(self.device).to(torch.float64)
            
        # Ensure dimensions [N, Vars]? No, usually [N] for 1D or [N, Vars]
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(1) # [N, 1]
            
        # 2. Initialization
        population = self.initialize_population()
        pop_constants = torch.randn(self.pop_size, self.max_constants, device=self.device, dtype=torch.float64)
        
        # 3. Seed Injection
        if seeds:
            seed_pop, seed_consts = self.load_population_from_strings(seeds)
            if seed_pop is not None:
                n_seeds = seed_pop.shape[0]
                # Place seeds at the beginning
                n_inject = min(n_seeds, self.pop_size)
                population[:n_inject] = seed_pop[:n_inject]
                pop_constants[:n_inject] = seed_consts[:n_inject]
                # print(f"[GPU] Injected {n_inject} seeds.")
        
        # State tracking
        best_rmse = float('inf')
        best_rpn = None
        best_consts_vec = None
        stagnation_counter = 0
        current_mutation_rate = GpuGlobals.BASE_MUTATION_RATE
        generations = 0
        
        COMPLEXITY_PENALTY = 0.001
        
        while time.time() - start_time < timeout_sec:
            generations += 1
            
            # --- EVALUATION ---
            fitness_rmse = self.evaluate_batch(population, x_t, y_t, pop_constants)
            
            # --- CONSTANT OPTIMIZATION (Hybrid) ---
            # Optimize constants for top K individuals
            k_opt = min(10, self.pop_size)
            _, top_indices = torch.topk(fitness_rmse, k_opt, largest=False)
            
            sub_pop = population[top_indices]
            sub_const = pop_constants[top_indices]
            
            # 5 steps of Adam
            opt_const, new_fit = self.optimize_constants(
                sub_pop, sub_const, x_t, y_t, steps=5, lr=0.5
            )
            
            # Update back
            pop_constants[top_indices] = opt_const
            fitness_rmse[top_indices] = new_fit
            
            # --- SELECTION & STATS ---
            min_fit, min_idx = torch.min(fitness_rmse, dim=0)
            if min_fit.item() < best_rmse:
                best_rmse = min_fit.item()
                best_rpn = population[min_idx].clone()
                best_consts_vec = pop_constants[min_idx].clone()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            if best_rmse < 1e-6:
                break
                
            # --- ISLAND MIGRATION ---
            if self.n_islands > 1 and generations % GpuGlobals.MIGRATION_INTERVAL == 0:
                population, pop_constants = self.migrate_islands(population, pop_constants, fitness_rmse)
                
            # --- OFFSPRING GENERATION ---
            # 1. Select Parents (Tournament)
            n_islands_eff = self.n_islands if self.n_islands > 0 else 1
            parent_indices = self.tournament_selection_island(population, fitness_rmse, n_islands_eff)
            parents = population[parent_indices]
            parent_fitness = fitness_rmse[parent_indices]
            
            # 2. Crossover & Mutate
            offspring = self.crossover_population(parents, GpuGlobals.DEFAULT_CROSSOVER_RATE)
            offspring = self.mutate_population(offspring, current_mutation_rate)
            
            # 3. Inherit Constants & Mutate
            offspring_const = pop_constants[parent_indices].clone()
            c_noise = torch.randn_like(offspring_const) * 0.1
            noise_mask = (torch.rand_like(offspring_const) < 0.2).to(torch.float64)
            offspring_const = offspring_const + (c_noise * noise_mask)
            
            # 4. Evaluate Offspring
            off_fitness = self.evaluate_batch(offspring, x_t, y_t, offspring_const)
            
            # 5. Deterministic Crowding (Survival)
            new_pop, new_fitness_batch = self.deterministic_crowding(parents, offspring, parent_fitness, off_fitness)
            
            # Update constants based on survival
            mask = off_fitness < parent_fitness
            new_const = torch.where(mask.unsqueeze(1), offspring_const, pop_constants[parent_indices])
            
            population = new_pop
            pop_constants = new_const
            fitness_rmse = new_fitness_batch
            
            # Dynamic rates
            if stagnation_counter > 20:
                current_mutation_rate = min(0.5, GpuGlobals.BASE_MUTATION_RATE + 0.1)
                
        # --- FINAL REFINEMENT ---
        if best_rpn is not None:
            # Deep optimization on best candidate
            best_ind_batch = best_rpn.unsqueeze(0)
            best_const_batch = best_consts_vec.unsqueeze(0)
            
            opt_const_final, final_fit_batch = self.optimize_constants(
                best_ind_batch, best_const_batch, x_t, y_t, steps=100, lr=0.1
            )
            
            best_formula = self.rpn_to_infix(best_rpn, opt_const_final[0])
            return best_formula
            
        return None
