
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
        Duplicates are replaced with mutated versions of random non-duplicates.
        
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
        unique_indices = []
        
        for i in range(pop_size):
            # Create hash from non-padding tokens
            tokens = pop_cpu[i]
            non_pad = tokens[tokens != PAD_ID]
            hash_key = tuple(non_pad.tolist())
            
            if hash_key in seen_hashes:
                duplicate_indices.append(i)
            else:
                seen_hashes[hash_key] = i
                unique_indices.append(i)
        
        n_dups = len(duplicate_indices)
        if n_dups == 0:
            return population, constants, 0
        
        # Replace duplicates with mutated versions of random unique individuals
        pop_out = population.clone()
        const_out = constants.clone()
        
        unique_indices_t = torch.tensor(unique_indices, device=self.device)
        
        for dup_idx in duplicate_indices:
            # Pick random unique individual
            rand_idx = unique_indices_t[torch.randint(len(unique_indices_t), (1,))].item()
            
            # Mutate it with higher rate for diversity
            mutant = self.mutate_population(population[rand_idx:rand_idx+1], mutation_rate=0.3)
            
            # Assign
            pop_out[dup_idx] = mutant[0]
            
            # Slightly perturb constants
            const_out[dup_idx] = constants[rand_idx] + torch.randn_like(constants[rand_idx]) * 0.1
        
        return pop_out, const_out, n_dups


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

    def crossover_population(self, parents: torch.Tensor, crossover_rate: float) -> torch.Tensor:
        """
        Performs subtree crossover on the population.
        parents: [PopSize, L] (assumed valid RPNs)
        """
        pop_size, length = parents.shape
        offspring = parents.clone()
        
        # Determine who performs crossover
        # We process in pairs. 
        # For simplicity, we can shuffle parents and take pairs.
        # Logic: Select 2 parents, produce 1 offspring (or 2). 
        # Standard GP: 2 Parents -> 2 Offspring.
        # Vectorized is hard because varying lengths. CPU loop is acceptable for logic, but slow?
        # Let's try CPU loop with some optimizations (only for those selected).
        
        # 1. Select IDs participating
        # We need pairs.
        indices = torch.randperm(pop_size, device=self.device) # Shuffle
        # Num pairs = PopSize // 2
        
        # Move to CPU for graph logic (RPN traversal is hard in tensor)
        parents_cpu = parents.detach().cpu().numpy()
        offspring_cpu = parents_cpu.copy() # Start with copies
        
        num_crossovers = int(pop_size * 0.5 * crossover_rate)
        
        for k in range(num_crossovers):
            # Indices in the shuffled array
            idx_a = indices[2*k].item()
            idx_b = indices[2*k+1].item()
            
            pA = parents_cpu[idx_a]
            pB = parents_cpu[idx_b]
            
            # Find subtrees
            # We pick a random point in A that is NOT PAD.
            len_A = np.count_nonzero(pA != PAD_ID)
            if len_A == 0: continue
            
            # Root A
            # We must pick a valid node.
            # RPN: Root is always the last valid node? No, any node is valid subtree root.
            c_point_A = np.random.randint(0, len_A)
            span_A = self.grammar.get_subtree_span(pA, c_point_A)
            
            if span_A[0] == -1: continue # Should not happen if logic valid
            
            # Parent B
            len_B = np.count_nonzero(pB != PAD_ID)
            if len_B == 0: continue
            c_point_B = np.random.randint(0, len_B)
            span_B = self.grammar.get_subtree_span(pB, c_point_B)
            
            if span_B[0] == -1: continue
            
            # Swap B into A
            # New A = A[:startA] + B[startB:endB+1] + A[endA+1:]
            
            subtree_B = pB[span_B[0] : span_B[1]+1]
            prefix_A = pA[:span_A[0]]
            suffix_A = pA[span_A[1]+1:]
            
            # Check length
            new_len = len(prefix_A) + len(subtree_B) + len(suffix_A)
            # Trim suffix (usually PADs) to real logic?
            # Actually suffix_A might contain operators that used the subtree A.
            # So we MUST keep them.
            # But suffix_A contains also the trailing PADS?
            # We need to trim trailing PADS from original A first to define 'real' suffix?
            # pA[span_A[1]+1:] includes everything till max_len.
            
            # Let's count real suffix.
            real_len_A = np.count_nonzero(pA != PAD_ID)
            # Suffix A real part is [span_A[1]+1 : real_len_A]
            
            real_suffix_A = pA[span_A[1]+1 : real_len_A]
            
            new_gene = np.concatenate([prefix_A, subtree_B, real_suffix_A])
            
            if len(new_gene) <= self.max_len:
                # Pad
                padded = np.zeros(self.max_len, dtype=pA.dtype) # 0 is PAD_ID?
                padded[:len(new_gene)] = new_gene
                padded[len(new_gene):] = PAD_ID
                offspring_cpu[idx_a] = padded
                # We update idx_a. idx_b stays as donor (or we could make 2 children).
                # Current logic updates offspring at idx_a.
            
            # Optionally: Child 2 (A into B)
            # Not strict requirement, but good for diversity.
            
        # Copy back
        return torch.tensor(offspring_cpu, device=self.device, dtype=torch.long)




    def infix_to_rpn(self, formulas: List[str]) -> torch.Tensor:
        """
        Converts a list of infix strings to a padded RPN tensor [B, L].
        """
        batch_rpn = []
        for f in formulas:
            try:
                tree = ExpressionTree.from_infix(f)
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
            except:
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
            if token_id == PAD_ID: break
            
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
    


    def _run_vm(self, population: torch.Tensor, x: torch.Tensor, constants: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Internal VM interpreter to evaluate RPN population on the GPU.
        Returns: (final_predictions, stack_pointer)
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
            valid_op = is_binary & (sp >= 2)
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
            valid_op = is_unary & (sp >= 1)
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
        return stack[:, 0], sp

    def evaluate_batch(self, population: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor, constants: torch.Tensor = None) -> torch.Tensor:
        """
        Evaluates the RPN population on the GPU.
        Returns: RMSE per individual [PopSize]
        """
        B, L = population.shape
        D = x.shape[0]
        
        final_preds, sp = self._run_vm(population, x, constants)
        
        is_valid = (sp == 1)
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
            is_binary = (token == op_add) | (token == op_sub) | (token == op_mul) | (token == op_div) | (token == op_pow)
            valid_op = is_binary & (sp >= 2)
            
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
                    # No artificial clamping for float64 unless extremely huge to avoid NaN propagation immediately
                    # C++ just does pow(l, r)
                    # But we can protect against complex numbers (negative base ^ float exp) -> NaN
                    base = val_a[mask]
                    expon = val_b[mask]
                    # If base < 0 and exponent is not integer loop, result is NaN. 
                    # We can protect base like C++ protected ops sometimes do, or just let it be NaN (yielding INF fitness)
                    res[mask] = torch.pow(base, expon)
                
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
                    out[safe_mask] = torch.log(inp[safe_mask])
                    res[mask] = out
                
                mask = (token == op_exp) & valid_op
                if mask.any(): 
                    # C++: (val > 700.0) ? GPU_MAX_DOUBLE : exp(val)
                    inp = val_a[mask]
                    safe_mask = inp <= 700.0
                    out = torch.full_like(inp, 1e300)
                    out[safe_mask] = torch.exp(inp[safe_mask])
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
                     # tgamma(n+1) = n!
                     # We use lgamma and exp to be safe? or torch.special.gamma?
                     # torch.special.gamma is 'tgamma' equivalent.
                     # Protection
                     unsafe = (inp < 0) | (inp > 170.0)
                     out = torch.full_like(inp, 1e300)
                     
                     # Only compute safe to avoid NaN/Inf in gradients or runtime
                     safe_inp = inp.clone()
                     safe_inp[unsafe] = 1.0 # dummy
                     
                     val_computed = torch.special.gamma(safe_inp + 1.0)
                     out[~unsafe] = val_computed[~unsafe]
                     res[mask] = out

                mask = (token == op_gamma) & valid_op
                if mask.any():
                     # C++: (val <= -1.0) ? GPU_MAX_DOUBLE : lgamma(val + 1.0); 
                     # Wait, snippet said lgamma(val+1). Usually 'gamma' op is just gamma function?
                     # C++ snippet: case 'g': result = (val <= -1.0) ? GPU_MAX_DOUBLE : lgamma(val + 1.0); 
                     # This seems to be Log-Gamma of (x+1)? Or is it Gamma? 
                     # 'lgamma' function usually computes log(|gamma(x)|).
                     # The snippet explicitly says lgamma. So GPU op 'g' is log-gamma.
                     inp = val_a[mask]
                     unsafe = (inp <= -1.0)
                     out = torch.full_like(inp, 1e300)
                     
                     safe_inp = inp.clone()
                     safe_inp[unsafe] = 1.0
                     
                     val_computed = torch.special.gammaln(safe_inp + 1.0) # lgamma matches gammaln in torch
                     out[~unsafe] = val_computed[~unsafe]
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

    def evaluate_differentiable(self, population: torch.Tensor, constants: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor):
        import torch.nn.functional as F
        return self.evaluate_batch(population, x, y_target), torch.zeros_like(x).expand(population.shape[0], -1) 

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

        

    def initialize_population(self) -> torch.Tensor:
        """
        Generates a population of VALID random formulas.
        """
        pool_size = min(self.pop_size, 2000)
        formulas = []
        for _ in range(pool_size):
            try:
                # Generate random valid tree
                tree = ExpressionTree.generate_random(max_depth=GpuGlobals.MAX_TREE_DEPTH_INITIAL, num_variables=self.num_variables)
                formulas.append(tree.get_infix())
            except:
                formulas.append("x0")
                
        # Convert to RPN
        pool_rpn = self.infix_to_rpn(formulas)
        
        # Tile to fill population
        if pool_size < self.pop_size:
            n_repeats = (self.pop_size // pool_size) + 1
            population = pool_rpn.repeat(n_repeats, 1)[:self.pop_size]
        else:
            population = pool_rpn
            
        return population.clone()

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

        while time.time() - start_time < timeout_sec:
            generations += 1
            

            # Eval (Fast Scan)
            fitness_rmse = self.evaluate_batch(population, x_t, y_t, pop_constants)
            
            # --- Constant Optimization (Top K) ---
            # Optimize top 200 candidates to refine their constants
            k_opt = min(self.pop_size, 200)
            _, top_idx = torch.topk(fitness_rmse, k_opt, largest=False)
            
            # Extract subset
            opt_pop = population[top_idx]
            opt_consts = pop_constants[top_idx]
            
            # Run Adam
            refined_consts, refined_rmse = self.optimize_constants(opt_pop, opt_consts, x_t, y_t, steps=15)
            
            # Update Population
            # We must update the original tensors.
            # 1. Update Constants
            pop_constants[top_idx] = refined_consts
            # 2. Update Fitness Scores (Evaluation was implicit in optimize)
            # But wait, fitness_rmse is [PopSize].
            # We update the scores.
            fitness_rmse[top_idx] = refined_rmse
            
            # --- Selection ---
            lengths = (population != PAD_ID).sum(dim=1).float()
            fitness_penalized = fitness_rmse * (1.0 + COMPLEXITY_PENALTY * lengths) + lengths * 1e-6
            
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
            
            if callback and (generations % 100 == 0 or generations == 1) and best_rpn is not None:
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
            
            # Compute Full Errors for Lexicase
            # Only do this every N generations or always? 
            # evaluate_batch_full is barely slower than evaluate_batch (just memory transfer).
            # But we already computed fitness_rmse (evaluate_batch).
            # Re-running evaluate_batch_full is wasteful. 
            # Ideally we run evaluate_batch_full ONCE and derive RMSE from it.
            # But for now, let's run it just for parent selection.
            
            full_errors = self.evaluate_batch_full(population, x_t, y_t, pop_constants)
            
            n_crossover = int(remaining_slots * 0.5) # 50% of remaining
            n_mutation = remaining_slots - n_crossover
            
            # Select parents for crossover using Lexicase
            # We need 2 * n_crossover parents? No, crossover_population takes list and shuffles.
            # So we need n_crossover parents? Wait, crossover produces same size as input?
            # self.crossover_population takes 'parents' and returns 'offspring'.
            # If we pass N parents, we get N offspring (paired).
            
            parents_cross = self.lexicase_selection(population, full_errors, n_crossover)
            # We assume parents_cross are good. We mix them.
            # Constants for crossover? We just copy them from first parent for now?
            # Or mix? Crossover only touches RPN. Constants indices stay same.
            # So we need to fetch corresponding constants.
            # Wait, logic in lexicase returns RPN tensors. We lost indices!
            # We need indices to fetch constants.
            # Hack: modify lexicase to return INDICES.
            
            # Temporarily, use tournament for crossover to save time refactoring lexicase?
            # No, I promised Lexicase.
            # I will assume lexicase returns RPN and I ignore constants (randomize new ones) 
            # OR I refactor Lexicase to return indices. 
            # Refactoring Lexicase to return Indices is cleaner.
            
            # Let's fix Lexicase later in this block or define a local helper?
            # I defined `lexicase_selection` to return population[indices].
            # I should recall it to return indices.
            
            # indices_cross = self.lexicase_selection_indices(...) 
            # But I submitted the tool call already. 
            # I will use tournament for now to ensure stability or hack it by matching tensor? No.
            # I will just re-implement simple tournament here for speed and safety, 
            # or use the full_errors to do a quick vectorized tournament.
            
            # Vectorized Tournament (RMSE based)
            # indices = torch.randint(0, self.pop_size, (n_crossover, 5), device=self.device)
            # fits = fitness_penalized[indices]
            # winner_local = torch.argmin(fits, dim=1)
            # winner_global = indices.gather(1, winner_local.unsqueeze(1)).squeeze(1)
            # parents_cross = population[winner_global]
            # consts_cross = pop_constants[winner_global]
            
            # off_cross = self.crossover_population(parents_cross, 1.0) # Always cross
            # next_pop_list.append(off_cross)
            # next_const_list.append(consts_cross)
            
            # 3. Mutation (Tournament)
            # indices_mut = torch.randint(0, self.pop_size, (n_mutation, 3), device=self.device)
            # fits_mut = fitness_penalized[indices_mut]
            # winner_mut = torch.argmin(fits_mut, dim=1)
            # global_mut = indices_mut.gather(1, winner_mut.unsqueeze(1)).squeeze(1)
            # parents_mut = population[global_mut]
            # consts_mut = pop_constants[global_mut]
            
            # # Mutate
            # # We need to flatten? mutate_population handles [N, L]
            # off_mut = self.mutate_population(parents_mut, current_mutation_rate)
            # next_pop_list.append(off_mut)
            # next_const_list.append(consts_mut)
            
            # ... Consolidate ...
            
            # --- IMPLEMENTING WITH LEXICASE (Indices Version) ---
            # I will access 'lexicase_selection' logic via copy-paste here tailored for indices,
            # since I cannot change the previous method easily in one go.
            # ACTUALLY, I can just use the tool to OVERWRITE `lexicase_selection` to return indices.
            pass

            # ... Wait, I am inside 'replace_file_content'. I must implement valid python.
            
            # Quick standard tournament for now to bridge the gap, as I shouldn't rely on 'lexicase_selection' returning indices yet.
            # I will implement 'lexicase_indices' locally.
            
            def get_lexicase_indices(n_select, errs_cpu):
                # errs_cpu: [Pop, Cases] numpy
                P, C_ = errs_cpu.shape
                selected = []
                for _ in range(n_select):
                    cands = np.random.randint(0, P, 50) # pool
                    cases = np.random.permutation(C_)
                    for c_idx in cases:
                        c_errs = errs_cpu[cands, c_idx]
                        min_e = np.min(c_errs)
                        eps = max(min_e * 0.1, 1e-9)
                        keep = c_errs <= (min_e + eps)
                        cands = cands[keep]
                        if len(cands) <= 1: break
                    selected.append(np.random.choice(cands))
                return torch.tensor(selected, device=self.device, dtype=torch.long)

            # Crossover Parents (Lexicase)
            full_errors_cpu = full_errors.detach().cpu().numpy()
            idx_cross = get_lexicase_indices(n_crossover, full_errors_cpu)
            parents_cross = population[idx_cross]
            consts_cross = pop_constants[idx_cross]
            
            off_cross = self.crossover_population(parents_cross, 0.9) # 90% crossover rate
            next_pop_list.append(off_cross)
            next_const_list.append(consts_cross)
            
            # Mutation Parents (Tournament)
            idx_mut = torch.randint(0, self.pop_size, (n_mutation, 3), device=self.device)
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

            # --- Deduplication (Every 20 generations) ---
            if GpuGlobals.PREVENT_DUPLICATES and generations % 20 == 0:
                population, pop_constants, n_dups = self.deduplicate_population(population, pop_constants)
                # Silent - only log if many duplicates
                if n_dups > self.pop_size * 0.1:
                    print(f"[GPU] Removed {n_dups} duplicates")

            # --- Simplification (Every 50 generations) ---
            if GpuGlobals.USE_SIMPLIFICATION and generations % 50 == 0:
                # Simplify top 50 individuals
                population, pop_constants, n_simp = self.simplify_population(population, pop_constants, top_k=50)
                if n_simp > 0 and callback:
                    print(f"[GPU] Simplified {n_simp} expressions")

            # --- Pattern Memory ---
            # Record successful subtrees from current population
            self.pattern_memory.record_subtrees(population, fitness_rmse, self.grammar)
            
            # Inject patterns periodically
            if generations % GpuGlobals.PATTERN_INJECT_INTERVAL == 0:
                population, pop_constants, n_inj = self.pattern_memory.inject_into_population(
                    population, pop_constants, self.grammar, 
                    percent=GpuGlobals.PATTERN_INJECT_PERCENT
                )

            # --- Local Search (Every 25 generations) ---
            if generations % 25 == 0:
                population, pop_constants = self.local_search(
                    population, pop_constants, x_t, y_t, 
                    top_k=10, attempts=GpuGlobals.LOCAL_SEARCH_ATTEMPTS
                )
            
            if best_rmse < 1e-7:
                 return self.rpn_to_infix(best_rpn, best_consts_vec)
                 
        if best_rpn is not None:
             return self.rpn_to_infix(best_rpn, best_consts_vec)
        return None
