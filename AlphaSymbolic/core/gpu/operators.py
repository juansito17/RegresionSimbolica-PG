
import torch
import random
import numpy as np
from typing import List, Tuple
from core.grammar import OPERATORS, ExpressionTree
from .grammar import PAD_ID, GPUGrammar
from .config import GpuGlobals

try:
    from sys import path as sys_path
    from os import path as os_path
    # Ensure CUDA module is reachable
    cuda_path = os_path.join(os_path.dirname(__file__), 'cuda')
    if cuda_path not in sys_path: sys_path.append(cuda_path)
    import rpn_cuda_native
    RPN_CUDA_AVAILABLE = True
except ImportError:
    RPN_CUDA_AVAILABLE = False
    # print("[GPUOperators] Warning: rpn_cuda_native not found. Using pure PyTorch.")

class GPUOperators:
    def __init__(self, grammar: GPUGrammar, device, pop_size, max_len=30, num_variables=1, dtype=torch.float64):
        self.grammar = grammar
        self.device = device
        self.dtype = dtype
        self.pop_dtype = self.grammar.dtype # uint8
        self.max_len = max_len
        self.num_variables = num_variables
        self.pop_size = pop_size
        
        # Pre-allocate memory for random generation
        self.terminal_ids = torch.tensor([self.grammar.token_to_id[t] for t in self.grammar.terminals], device=self.device, dtype=self.pop_dtype)
        self.operator_ids = torch.tensor([self.grammar.token_to_id[op] for op in self.grammar.operators], device=self.device, dtype=self.pop_dtype)
        
        # Arity masks
        self._init_arity_masks()

    def _init_arity_masks(self):
        self.token_arity = torch.zeros(self.grammar.vocab_size + 1, dtype=torch.long, device=self.device)
        self.arity_0_ids = []
        self.arity_1_ids = []
        self.arity_2_ids = []
        
        for t in self.grammar.terminals:
            tid = self.grammar.token_to_id[t]
            self.token_arity[tid] = 0
            self.arity_0_ids.append(tid)
            
        for op in self.grammar.operators:
            tid = self.grammar.token_to_id[op]
            arity = OPERATORS.get(op, 1)
            self.token_arity[tid] = arity
            if arity == 1: self.arity_1_ids.append(tid)
            elif arity == 2: self.arity_2_ids.append(tid)
            
        self.arity_0_ids = torch.tensor(self.arity_0_ids, device=self.device, dtype=self.pop_dtype)
        self.arity_1_ids = torch.tensor(self.arity_1_ids, device=self.device, dtype=self.pop_dtype)
        self.arity_2_ids = torch.tensor(self.arity_2_ids, device=self.device, dtype=self.pop_dtype)
        
        # Cache int32 arities for CUDA
        self.token_arity_int = self.token_arity.to(dtype=torch.int32)

    def _force_multi_variable(self, population: torch.Tensor) -> torch.Tensor:
        """
        Ensure a fraction of the population uses all available variables.
        Replaces random terminal positions with missing variable tokens.
        Fully vectorized — no Python per-individual loops.
        """
        if self.num_variables <= 1 or GpuGlobals.VAR_FORCE_SEED_PERCENT <= 0:
            return population
        
        size = population.shape[0]
        device = self.device
        n_force = int(size * GpuGlobals.VAR_FORCE_SEED_PERCENT)
        if n_force < 1:
            return population
        
        subset = population[:n_force]  # [n_force, max_len]
        
        for vi in range(self.num_variables):
            var_name = f'x{vi}'
            vid = self.grammar.token_to_id.get(var_name, -1)
            if vid <= 0:
                continue
            
            has_var = (subset == vid).any(dim=1)  # [n_force]
            missing_mask = ~has_var
            n_missing = missing_mask.sum().item()
            
            if n_missing == 0:
                continue
            
            missing_idx = missing_mask.nonzero(as_tuple=True)[0]  # [n_missing]
            missing_pop = subset[missing_idx]  # [n_missing, max_len]
            
            # Find terminal positions (arity-0 tokens, non-PAD)
            is_term = torch.zeros_like(missing_pop, dtype=torch.bool)
            for tid in self.arity_0_ids:
                is_term |= (missing_pop == tid.item())
            
            # Each row: pick a random terminal position to replace
            has_any = is_term.any(dim=1)  # [n_missing]
            if not has_any.any():
                continue
            
            valid_idx = has_any.nonzero(as_tuple=True)[0]
            valid_pop = missing_pop[valid_idx]
            valid_term = is_term[valid_idx]
            
            # Trick: random weights * mask → argmax picks random terminal position
            rand_w = torch.rand_like(valid_term.float()) * valid_term.float()
            pos = rand_w.argmax(dim=1)  # [n_valid]
            valid_pop[torch.arange(valid_idx.shape[0], device=device), pos] = vid
            
            missing_pop[valid_idx] = valid_pop
            subset[missing_idx] = missing_pop
        
        return population

    def generate_random_population(self, size: int) -> torch.Tensor:
        """
        Helper to generate random RPN population of given size.
        Uses GPU-native generation for speed, then forces multi-variable usage.
        """
        pop = self.generate_random_population_gpu(size)
        return self._force_multi_variable(pop)
    
    def generate_random_population_cpu(self, size: int) -> torch.Tensor:
        """
        CPU-based generation using ExpressionTree (slow, kept for reference).
        """
        formulas = []
        for _ in range(size):
            try:
                tree = ExpressionTree.generate_random(max_depth=GpuGlobals.MAX_TREE_DEPTH_INITIAL, num_variables=self.num_variables)
                formulas.append(tree.get_infix())
            except:
                formulas.append("x0")
        return self._infix_list_to_rpn(formulas)

    def generate_random_population_gpu(self, size: int) -> torch.Tensor:
        """
        GPU-native random RPN generation. Optimized to minimize kernel launches.
        
        Strategy: Try CUDA kernel first (single launch), fall back to PyTorch loop.
        """
        max_len = self.max_len
        device = self.device
        
        # ============ CUDA FAST PATH ============
        if RPN_CUDA_AVAILABLE:
            try:
                population = torch.zeros(size, max_len, dtype=self.pop_dtype, device=device)
                seed = random.getrandbits(62)
                
                # Ensure contiguous int64 tensors
                t_ids = self.terminal_ids.contiguous()
                u_ids = self.arity_1_ids.contiguous() if self.arity_1_ids.numel() > 0 else torch.zeros(0, dtype=torch.long, device=device)
                b_ids = self.arity_2_ids.contiguous() if self.arity_2_ids.numel() > 0 else torch.zeros(0, dtype=torch.long, device=device)
                
                rpn_cuda_native.generate_random_rpn(
                    population, t_ids, u_ids, b_ids, seed
                )
                
                # Quick validation
                valid = self._validate_rpn_batch(population)
                invalid = ~valid
                if invalid.any():
                    x0_id = self.grammar.token_to_id.get('x0', self.grammar.token_to_id.get('x', 1))
                    population[invalid, 0] = x0_id
                    population[invalid, 1:] = PAD_ID
                
                return population
            except Exception:
                pass  # Fall through to PyTorch implementation
        
        # ============ PyTorch FALLBACK ============
        
        
        # Output tensor
        population = torch.zeros(size, max_len, dtype=self.pop_dtype, device=device)
        
        # Stack balance tracker
        stack = torch.zeros(size, dtype=torch.long, device=device)
        
        # Token pools (cached references)
        n_terminals = self.terminal_ids.shape[0]
        n_arity1 = self.arity_1_ids.shape[0]
        n_arity2 = self.arity_2_ids.shape[0]
        
        all_tokens = torch.cat([self.terminal_ids, self.arity_1_ids, self.arity_2_ids])
        n_total = all_tokens.shape[0]
        
        token_deltas = torch.cat([
            torch.ones(n_terminals, dtype=torch.long, device=device),
            torch.zeros(n_arity1, dtype=torch.long, device=device),
            -torch.ones(n_arity2, dtype=torch.long, device=device)
        ])
        
        # PRE-GENERATE all random numbers in one bulk call (1 kernel instead of 30)
        # Using uniform random + thresholds instead of multinomial per step
        all_rand = torch.rand(size, max_len, device=device)
        
        # Pre-compute unique delta values for fast category selection
        # Categories: terminal(+1), unary(0), binary(-1)
        # We determine valid categories per step, then pick within category
        
        # Pre-generate random indices within each category (3 bulk calls total)
        rand_terminal_idx = torch.randint(0, max(1, n_terminals), (size, max_len), device=device)
        rand_arity1_idx = torch.randint(0, max(1, n_arity1), (size, max_len), device=device) if n_arity1 > 0 else None
        rand_arity2_idx = torch.randint(0, max(1, n_arity2), (size, max_len), device=device) if n_arity2 > 0 else None
        
        for j in range(max_len):
            remaining = max_len - j - 1
            
            # Determine valid token categories based on stack state
            # For category terminal (+1): new_stack = stack + 1
            # For category unary (0):     new_stack = stack
            # For category binary (-1):   new_stack = stack - 1
            
            s = stack  # [size]
            
            # Check each category validity (vectorized over all individuals)
            # Constraint: new_stack >= 1 AND new_stack <= 1 + remaining
            can_terminal = ((s + 1) >= 1) & ((s + 1) <= 1 + remaining)
            can_unary = (n_arity1 > 0) & (s >= 1) & (s <= 1 + remaining)
            can_binary = (n_arity2 > 0) & ((s - 1) >= 1) & ((s - 1) <= 1 + remaining)
            
            # Last position: must end at stack=1
            if remaining == 0:
                can_terminal = can_terminal & ((s + 1) == 1)
                can_unary = can_unary & (s == 1) if n_arity1 > 0 else can_unary
                can_binary = can_binary & ((s - 1) == 1) if n_arity2 > 0 else can_binary
            
            # Convert to float weights for normalized sampling
            # TERMINAL_VS_VARIABLE_PROB controla sesgo hacia terminales (mayor = más terminales)
            terminal_bias = GpuGlobals.TERMINAL_VS_VARIABLE_PROB
            w_t = can_terminal.float() * terminal_bias
            w_u = (can_unary.float() if isinstance(can_unary, torch.Tensor) else torch.zeros(size, device=device)) * (1.0 - terminal_bias) * 0.5
            w_b = (can_binary.float() if isinstance(can_binary, torch.Tensor) else torch.zeros(size, device=device)) * (1.0 - terminal_bias) * 0.5
            
            # Ensure at least terminal is valid (fallback)
            total_w = w_t + w_u + w_b
            no_valid = (total_w == 0)
            w_t = w_t + no_valid.float()  # fallback to terminal
            total_w = w_t + w_u + w_b
            
            # Cumulative thresholds for category selection using pre-generated rand
            # [0, p_t) -> terminal, [p_t, p_t+p_u) -> unary, [p_t+p_u, 1) -> binary
            p_t = w_t / total_w
            p_u = w_u / total_w
            
            r = all_rand[:, j]  # pre-generated random [size]
            
            is_terminal = r < p_t
            is_unary = (~is_terminal) & (r < (p_t + p_u))
            is_binary = (~is_terminal) & (~is_unary)
            
            # Select token within chosen category
            chosen_tokens = self.terminal_ids[rand_terminal_idx[:, j] % n_terminals]  # default: terminal
            
            if n_arity1 > 0:
                unary_tokens = self.arity_1_ids[rand_arity1_idx[:, j] % n_arity1]
                chosen_tokens = torch.where(is_unary, unary_tokens, chosen_tokens)
            
            if n_arity2 > 0:
                binary_tokens = self.arity_2_ids[rand_arity2_idx[:, j] % n_arity2]
                chosen_tokens = torch.where(is_binary, binary_tokens, chosen_tokens)
            
            # Compute delta for chosen tokens
            chosen_deltas = torch.ones(size, dtype=torch.long, device=device)  # terminal default
            if n_arity1 > 0:
                chosen_deltas = torch.where(is_unary, torch.zeros(size, dtype=torch.long, device=device), chosen_deltas)
            if n_arity2 > 0:
                chosen_deltas = torch.where(is_binary, -torch.ones(size, dtype=torch.long, device=device), chosen_deltas)
            
            population[:, j] = chosen_tokens
            stack = stack + chosen_deltas
        
        # Final validation
        valid = (stack == 1)
        invalid = ~valid
        
        if invalid.any():
            x0_id = self.grammar.token_to_id.get('x0', self.grammar.token_to_id.get('x', 1))
            population[invalid, 0] = x0_id
            population[invalid, 1:] = PAD_ID
        
        return population

    def _validate_rpn_batch(self, population: torch.Tensor) -> torch.Tensor:
        """
        Validate that each RPN formula has a final stack balance of exactly 1.
        Returns a boolean mask of valid formulas.
        Vectorized with cumsum - no Python loop.
        """
        B, L = population.shape
        # Get arities for all tokens at once: [B, L]
        arities = self.token_arity[population.clamp(0, self.token_arity.shape[0] - 1).long()]
        # Stack delta: terminals(arity=0) -> +1, unary(arity=1) -> 0, binary(arity=2) -> -1
        deltas = 1 - arities
        # PAD tokens should have delta 0
        is_pad = (population == PAD_ID)
        deltas[is_pad] = 0
        # Final stack = sum of all deltas
        final_stack = deltas.sum(dim=1)
        return (final_stack == 1)

    def _generate_small_subtrees(self, size: int, max_len: int) -> torch.Tensor:
        """
        Generate small RPN subtrees with limited length (for subtree mutation).
        Uses CUDA fast path if available, fallback to PyTorch loop.
        """
        device = self.device
        
        # CUDA fast path
        if RPN_CUDA_AVAILABLE:
            try:
                population = torch.zeros(size, max_len, dtype=self.pop_dtype, device=device)
                seed = random.getrandbits(62)
                t_ids = self.terminal_ids.contiguous()
                u_ids = self.arity_1_ids.contiguous() if self.arity_1_ids.numel() > 0 else torch.zeros(0, dtype=torch.long, device=device)
                b_ids = self.arity_2_ids.contiguous() if self.arity_2_ids.numel() > 0 else torch.zeros(0, dtype=torch.long, device=device)
                rpn_cuda_native.generate_random_rpn(population, t_ids, u_ids, b_ids, seed)
                valid = self._validate_rpn_batch_custom(population, max_len)
                invalid = ~valid
                if invalid.any():
                    x0_id = self.grammar.token_to_id.get('x0', self.grammar.token_to_id.get('x', 1))
                    population[invalid, 0] = x0_id
                    population[invalid, 1:] = PAD_ID
                return population
            except Exception:
                pass
        
        # PyTorch fallback — same as generate_random_population_gpu but with shorter max_len
        population = torch.zeros(size, max_len, dtype=self.pop_dtype, device=device)
        stack = torch.zeros(size, dtype=torch.long, device=device)
        n_terminals = self.terminal_ids.shape[0]
        n_arity1 = self.arity_1_ids.shape[0]
        n_arity2 = self.arity_2_ids.shape[0]
        all_rand = torch.rand(size, max_len, device=device)
        rand_terminal_idx = torch.randint(0, max(1, n_terminals), (size, max_len), device=device)
        rand_arity1_idx = torch.randint(0, max(1, n_arity1), (size, max_len), device=device) if n_arity1 > 0 else None
        rand_arity2_idx = torch.randint(0, max(1, n_arity2), (size, max_len), device=device) if n_arity2 > 0 else None
        terminal_bias = GpuGlobals.TERMINAL_VS_VARIABLE_PROB
        
        for j in range(max_len):
            remaining = max_len - j - 1
            s = stack
            can_terminal = ((s + 1) >= 1) & ((s + 1) <= 1 + remaining)
            can_unary = (n_arity1 > 0) & (s >= 1) & (s <= 1 + remaining)
            can_binary = (n_arity2 > 0) & ((s - 1) >= 1) & ((s - 1) <= 1 + remaining)
            if remaining == 0:
                can_terminal = can_terminal & ((s + 1) == 1)
                can_unary = can_unary & (s == 1) if n_arity1 > 0 else can_unary
                can_binary = can_binary & ((s - 1) == 1) if n_arity2 > 0 else can_binary
            w_t = can_terminal.float() * terminal_bias
            w_u = (can_unary.float() if isinstance(can_unary, torch.Tensor) else torch.zeros(size, device=device)) * (1.0 - terminal_bias) * 0.5
            w_b = (can_binary.float() if isinstance(can_binary, torch.Tensor) else torch.zeros(size, device=device)) * (1.0 - terminal_bias) * 0.5
            total_w = w_t + w_u + w_b
            no_valid = (total_w == 0)
            w_t = w_t + no_valid.float()
            total_w = w_t + w_u + w_b
            p_t = w_t / total_w
            p_u = w_u / total_w
            r = all_rand[:, j]
            is_terminal = r < p_t
            is_unary = (~is_terminal) & (r < (p_t + p_u))
            is_binary = (~is_terminal) & (~is_unary)
            chosen_tokens = self.terminal_ids[rand_terminal_idx[:, j] % n_terminals]
            if n_arity1 > 0:
                unary_tokens = self.arity_1_ids[rand_arity1_idx[:, j] % n_arity1]
                chosen_tokens = torch.where(is_unary, unary_tokens, chosen_tokens)
            if n_arity2 > 0:
                binary_tokens = self.arity_2_ids[rand_arity2_idx[:, j] % n_arity2]
                chosen_tokens = torch.where(is_binary, binary_tokens, chosen_tokens)
            chosen_deltas = torch.ones(size, dtype=torch.long, device=device)
            if n_arity1 > 0:
                chosen_deltas = torch.where(is_unary, torch.zeros(size, dtype=torch.long, device=device), chosen_deltas)
            if n_arity2 > 0:
                chosen_deltas = torch.where(is_binary, -torch.ones(size, dtype=torch.long, device=device), chosen_deltas)
            population[:, j] = chosen_tokens
            stack = stack + chosen_deltas
        
        valid = (stack == 1)
        invalid = ~valid
        if invalid.any():
            x0_id = self.grammar.token_to_id.get('x0', self.grammar.token_to_id.get('x', 1))
            population[invalid, 0] = x0_id
            population[invalid, 1:] = PAD_ID
        return population

    def _validate_rpn_batch_custom(self, population: torch.Tensor, max_len: int) -> torch.Tensor:
        """Validate RPN batch for custom-length populations."""
        B, L = population.shape
        arities = self.token_arity[population.clamp(0, self.token_arity.shape[0] - 1)]
        deltas = 1 - arities
        is_pad = (population == PAD_ID)
        deltas[is_pad] = 0
        final_stack = deltas.sum(dim=1)
        return (final_stack == 1)

    def _infix_list_to_rpn(self, formulas: list) -> torch.Tensor:
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

                ids = []
                for t in rpn_tokens:
                    if t in self.grammar.token_to_id:
                        ids.append(self.grammar.token_to_id[t])
                    else:
                        # Check number
                        try:
                            float(t)
                            # It's a number. Map to C (or specific literal if supported)
                            # But wait, grammar has '1', '2' etc. but token_to_id needs exact string match.
                            # '1.0' != '1'.
                            # Simple logic: Always 'C' for floats, unless integer match?
                            # Optimally: 'C'. The engine randomizes constants anyway.
                            ids.append(self.grammar.token_to_id.get('C', PAD_ID))
                        except ValueError:
                             ids.append(PAD_ID)
                if len(ids) > self.max_len:
                    ids = ids[:self.max_len]
                else:
                    ids = ids + [PAD_ID] * (self.max_len - len(ids))
                
                batch_rpn.append(ids)
            except Exception as e:
                batch_rpn.append([PAD_ID]*self.max_len)
                
        if not batch_rpn:
            return torch.empty((0, self.max_len), device=self.device, dtype=self.pop_dtype)
        return torch.tensor(batch_rpn, device=self.device, dtype=self.pop_dtype)

    def mutate_population(self, population: torch.Tensor, mutation_rate: float) -> torch.Tensor:
        """
        Performs arity-safe mutation on the population.
        """
        if RPN_CUDA_AVAILABLE and hasattr(rpn_cuda_native, 'mutate_population'):
             # CUDA Fast Path
             B, L = population.shape
             rand_floats = torch.rand(population.shape, device=self.device, dtype=torch.float32)
             rand_ints = torch.randint(0, 2**30, population.shape, device=self.device, dtype=torch.long)
             
             # In-place modification? No, clone to be safe (or in-place if allowed)
             # PyTorch usually clones for mutation.
             mutated_pop = population.clone()
             
             rpn_cuda_native.mutate_population(
                 mutated_pop, rand_floats, rand_ints,
                 self.token_arity_int,
                 self.arity_0_ids, self.arity_1_ids, self.arity_2_ids,
                 mutation_rate, PAD_ID
             )
             return mutated_pop

        # Fallback to PyTorch
        mask = torch.rand_like(population, dtype=self.dtype) < mutation_rate
        mask = mask & (population != PAD_ID)
        
        current_arities = self.token_arity[population]
        
        if len(self.arity_0_ids) > 0:
            rand_idx_0 = torch.randint(0, len(self.arity_0_ids), population.shape, device=self.device)
            replacements_0 = self.arity_0_ids[rand_idx_0]
        else:
            replacements_0 = population
            
        if len(self.arity_1_ids) > 0:
             rand_idx_1 = torch.randint(0, len(self.arity_1_ids), population.shape, device=self.device)
             replacements_1 = self.arity_1_ids[rand_idx_1]
        else:
             replacements_1 = population

        if len(self.arity_2_ids) > 0:
             rand_idx_2 = torch.randint(0, len(self.arity_2_ids), population.shape, device=self.device)
             replacements_2 = self.arity_2_ids[rand_idx_2]
        else:
             replacements_2 = population
             
        mutated_pop = population.clone()
        
        mask_0 = mask & (current_arities == 0)
        mutated_pop = torch.where(mask_0, replacements_0, mutated_pop)
        
        mask_1 = mask & (current_arities == 1)
        mutated_pop = torch.where(mask_1, replacements_1, mutated_pop)
        
        mask_2 = mask & (current_arities == 2)
        mutated_pop = torch.where(mask_2, replacements_2, mutated_pop)
        
        return mutated_pop

    def _get_subtree_ranges(self, population: torch.Tensor) -> torch.Tensor:
        """
        Calculates the start index of the subtree ending at each position.
        Returns tensor [B, L] where value is start_index, or -1 if invalid/padding.
        """
        if RPN_CUDA_AVAILABLE and hasattr(rpn_cuda_native, 'find_subtree_ranges'):
            B, L = population.shape
            starts = torch.full((B, L), -1, dtype=torch.long, device=self.device)
            # Ensure token_arity_int exists (legacy safety)
            if not hasattr(self, 'token_arity_int'): self.token_arity_int = self.token_arity.to(dtype=torch.int32)
            
            rpn_cuda_native.find_subtree_ranges(population, self.token_arity_int, starts, PAD_ID)
            return starts

        # PyTorch Fallback
        B, L = population.shape
        device = self.device
        
        # 1. Compute Stack Delta for each token
        # terminals: +1, arity1: 0, arity2: -1
        token_net_change = 1 - self.token_arity
        arities = token_net_change[population]
        arities[population == PAD_ID] = 0
        
        # 2. Compute Stack Depths at each position
        # depth[i] is the stack size AFTER processing token i
        depths = torch.cumsum(arities, dim=1)
        
        # 3. Find subtrees
        max_possible_depth = L + 2
        last_idx_for_depth = torch.full((B, max_possible_depth), -1, device=device, dtype=torch.long)
        
        # Initial depth is 0 at "index -1"
        last_idx_for_depth[:, 1] = 0 # depth 0 seen at start (virtual index)
        
        subtree_starts = torch.full((B, L), -1, device=device, dtype=torch.long)
        
        for i in range(L):
            d = depths[:, i]
            target_d_idx = (d).clamp(0, max_possible_depth - 1)
            starts = last_idx_for_depth.gather(1, target_d_idx.unsqueeze(1)).squeeze(1)
            
            is_valid = (starts != -1) & (population[:, i] != PAD_ID)
            subtree_starts[is_valid, i] = starts[is_valid]
            
            update_idx = (d + 1).clamp(0, max_possible_depth - 1)
            last_idx_for_depth.scatter_(1, update_idx.unsqueeze(1), torch.full((B, 1), i + 1, device=device, dtype=torch.long))

        return subtree_starts

    def crossover_population(self, parents: torch.Tensor, crossover_rate: float) -> torch.Tensor:
        B, L = parents.shape
        n_pairs = int(B * 0.5 * crossover_rate)
        if n_pairs == 0: return parents.clone()
        
        perm = torch.randperm(B, device=self.device)
        p1_idx = perm[:n_pairs*2:2]
        p2_idx = perm[1:n_pairs*2:2]
        
        parents_1 = parents[p1_idx]
        parents_2 = parents[p2_idx]
        
        starts_1_mat = self._get_subtree_ranges(parents_1)
        starts_2_mat = self._get_subtree_ranges(parents_2)
        
        valid_mask_1 = (starts_1_mat != -1)
        valid_mask_2 = (starts_2_mat != -1)
        
        # Sample points
        probs_1 = valid_mask_1.float() + 1e-6
        probs_2 = valid_mask_2.float() + 1e-6
        
        end_1 = torch.multinomial(probs_1, 1).squeeze(1)
        end_2 = torch.multinomial(probs_2, 1).squeeze(1)
        
        start_1 = starts_1_mat.gather(1, end_1.unsqueeze(1)).squeeze(1)
        start_2 = starts_2_mat.gather(1, end_2.unsqueeze(1)).squeeze(1)
        
        # Lengths
        len_1_pre = start_1
        len_1_sub = end_1 - start_1 + 1
        len_2_pre = start_2
        len_2_sub = end_2 - start_2 + 1
        
        if RPN_CUDA_AVAILABLE and hasattr(rpn_cuda_native, 'crossover_splicing'):
             # CUDA Fast Splicing
             # Allocate children
             c1 = torch.full((n_pairs, L), PAD_ID, dtype=self.pop_dtype, device=self.device)
             c2 = torch.full((n_pairs, L), PAD_ID, dtype=self.pop_dtype, device=self.device)
             
             rpn_cuda_native.crossover_splicing(
                 parents_1, parents_2,
                 start_1, end_1,
                 start_2, end_2,
                 c1, c2, PAD_ID
             )
        else:
            # PyTorch Fallback
            # Reconstruct Child 1
            grid = torch.arange(L, device=self.device).unsqueeze(0).expand(n_pairs, L)
            
            cut_1 = len_1_pre + len_2_sub
            mask_c1_pre = (grid < len_1_pre.unsqueeze(1))
            mask_c1_mid = (grid >= len_1_pre.unsqueeze(1)) & (grid < cut_1.unsqueeze(1))
            # post is rest
            
            src_idx_c1 = torch.zeros((n_pairs, L), dtype=torch.long, device=self.device)
            term_1 = grid
            term_2 = grid - len_1_pre.unsqueeze(1) + start_2.unsqueeze(1)
            term_3 = grid - cut_1.unsqueeze(1) + end_1.unsqueeze(1) + 1
            
            src_idx_c1 = torch.where(mask_c1_pre, term_1, 
                           torch.where(mask_c1_mid, term_2, term_3))
                           
            # Child 2
            cut_2 = len_2_pre + len_1_sub
            mask_c2_pre = (grid < len_2_pre.unsqueeze(1))
            mask_c2_mid = (grid >= len_2_pre.unsqueeze(1)) & (grid < cut_2.unsqueeze(1))
            
            term_2_1 = grid
            term_2_2 = grid - len_2_pre.unsqueeze(1) + start_1.unsqueeze(1)
            term_2_3 = grid - cut_2.unsqueeze(1) + end_2.unsqueeze(1) + 1
            
            src_idx_c2 = torch.where(mask_c2_pre, term_2_1,
                           torch.where(mask_c2_mid, term_2_2, term_2_3))
            
            sel_c1 = torch.where(mask_c1_mid, torch.tensor(1, device=self.device), torch.tensor(0, device=self.device))
            sel_c2 = torch.where(mask_c2_mid, torch.tensor(0, device=self.device), torch.tensor(1, device=self.device))
            
            def gather_mixed(idx_map, sel_map, t0, t1):
                safe_idx = torch.clamp(idx_map, 0, L-1)
                val0 = t0.gather(1, safe_idx)
                val1 = t1.gather(1, safe_idx)
                res = torch.where(sel_map == 0, val0, val1)
                is_pad = (idx_map < 0) | (idx_map >= L)
                res[is_pad] = PAD_ID
                return res
                
            c1 = gather_mixed(src_idx_c1, sel_c1, parents_1, parents_2)
            c2 = gather_mixed(src_idx_c2, sel_c2, parents_1, parents_2)
        
        # --- USE_HARD_DEPTH_LIMIT: Truncar hijos que excedan el límite duro ---
        if GpuGlobals.USE_HARD_DEPTH_LIMIT:
            hard_limit = GpuGlobals.MAX_TREE_DEPTH_HARD_LIMIT
            c1_len = (c1 != PAD_ID).sum(dim=1)
            c2_len = (c2 != PAD_ID).sum(dim=1)
            # Si un hijo excede el límite, revertir al padre original
            too_long_1 = c1_len > hard_limit
            too_long_2 = c2_len > hard_limit
            if too_long_1.any():
                c1[too_long_1] = parents_1[too_long_1]
            if too_long_2.any():
                c2[too_long_2] = parents_2[too_long_2]
        
        parents[p1_idx] = c1
        parents[p2_idx] = c2
        return parents

    def deduplicate_population(self, population: torch.Tensor, constants: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if not GpuGlobals.PREVENT_DUPLICATES:
            return population, constants, 0
        
        # GPU Accelerated Deduplication (Probabilistic Hashing)
        # 1. Compute Hash for each individual (Row)
        # Using pre-computed random weights for semantic hashing
        if not hasattr(self, 'dedup_weights'):
             # Lazy init
             self.dedup_weights = torch.randint(-9223372036854775807, 9223372036854775807, (self.max_len,), device=self.device, dtype=torch.long)
             
        # Hash: (B, L) * (L,) -> Sum -> (B,)
        # Use simple dot product equivalent with implicit wrapping for int64
        # We assume max_len matches population L. If L < max_len?
        curr_L = population.shape[1]
        if curr_L != self.dedup_weights.shape[0]:
             # Resize or re-init?
             # Just slice if smaller, or re-init if larger
             if curr_L > self.dedup_weights.shape[0]:
                 self.dedup_weights = torch.randint(..., curr_L) # reinit
             weights = self.dedup_weights[:curr_L]
        else:
             weights = self.dedup_weights
             
        hashes = (population * weights).sum(dim=1)
        
        # 2. Unique on Hashes (1D is fast)
        # return_inverse gives indices such that hashes = unique[inverse]
        _, inverse_indices = torch.unique(hashes, sorted=False, return_inverse=True)
        
        # 3. Find Duplicates
        # Sort inverse indices to find groups of identical hashes
        sorted_inv, sorted_idx = torch.sort(inverse_indices)
        
        # Mask where sorted_inv[i] == sorted_inv[i-1] -> Duplicate
        mask_dup = torch.zeros_like(sorted_inv, dtype=torch.bool)
        mask_dup[1:] = (sorted_inv[1:] == sorted_inv[:-1])
        
        # Indices of duplicates (in original population)
        dup_indices = sorted_idx[mask_dup]
        n_dups = dup_indices.shape[0]
        
        if n_dups == 0:
            return population, constants, 0
            
        # In-place replacement — no need to clone entire 1M population
        # Generate replacements
        fresh_pop = self.generate_random_population(n_dups)
        
        # Constants for replacements - use integer range if FORCE_INTEGER_CONSTANTS
        K = constants.shape[1]
        if GpuGlobals.FORCE_INTEGER_CONSTANTS:
            fresh_consts = torch.randint(
                GpuGlobals.CONSTANT_INT_MIN_VALUE, 
                GpuGlobals.CONSTANT_INT_MAX_VALUE + 1,
                (n_dups, K), device=self.device, dtype=torch.long
            ).to(self.dtype)
        else:
            fresh_consts = torch.empty(n_dups, K, device=self.device, dtype=self.dtype).uniform_(GpuGlobals.CONSTANT_MIN_VALUE, GpuGlobals.CONSTANT_MAX_VALUE)
        
        population[dup_indices] = fresh_pop
        constants[dup_indices] = fresh_consts
        
        return population, constants, n_dups

    def tarpeian_control(self, population: torch.Tensor, fitness: torch.Tensor) -> torch.Tensor:
        lengths = (population != PAD_ID).sum(dim=1).float()
        avg_len = lengths.mean()
        oversized = lengths > avg_len * 1.5
        random_mask = torch.rand(population.shape[0], device=self.device) < 0.5
        penalize_mask = oversized & random_mask
        fitness_out = fitness.clone()
        fitness_out[penalize_mask] = 1e30
        return fitness_out

    def replace_nan_inf(self, population: torch.Tensor, fitness: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Implementation if needed, or done in evaluation
        return population, fitness

    def subtree_mutation(self, population: torch.Tensor, mutation_rate: float) -> torch.Tensor:
        """
        Replaces a random subtree with a newly generated random tree.
        Crucial for structural exploration (Bloat control + Innovation).
        """
        B, L = population.shape
        # Identify valid subtrees
        starts_mat = self._get_subtree_ranges(population)
        valid_mask = (starts_mat != -1)
        
        # Decide who to mutate
        mutate_mask = (torch.rand(B, device=self.device) < mutation_rate)
        
        # If no valid subtrees for an individual, we can't mutate (skip)
        has_valid = valid_mask.any(dim=1)
        mutate_mask = mutate_mask & has_valid
        
        if not mutate_mask.any():
            return population
            
        # Indices of mutants
        mutant_indices = torch.nonzero(mutate_mask).squeeze(1)
        n_mut = mutant_indices.numel()
        
        # Select random end point for each mutant
        # We use multinomial on valid positions
        probs = valid_mask[mutate_mask].float() + 1e-6
        ends = torch.multinomial(probs, 1).squeeze(1) # [n_mut]
        
        starts = starts_mat[mutate_mask].gather(1, ends.unsqueeze(1)).squeeze(1) # [n_mut]
        
        # Lengths of subtrees to remove
        remove_lens = ends - starts + 1
        
        # Generate NEW random subtrees (RPN)
        # MAX_TREE_DEPTH_MUTATION limita la profundidad de subtrees generados en mutación.
        # Genera con max_len corto para forzar árboles pequeños.
        max_subtree_len = min(self.max_len, GpuGlobals.MAX_TREE_DEPTH_MUTATION * 2 + 1)
        replacements_raw = self._generate_small_subtrees(n_mut, max_subtree_len)
        # Pad to full max_len for splicing
        replacements = torch.full((n_mut, self.max_len), PAD_ID, dtype=self.pop_dtype, device=self.device)
        replacements[:, :max_subtree_len] = replacements_raw
        # Determine actual length of replacements (until first PAD)
        repl_lengths = (replacements != PAD_ID).sum(dim=1)
        
        # Check size constraints
        # New Len = Old Len - Subtree Len + New Subtree Len
        # Old Len:
        old_lens = (population[mutant_indices] != PAD_ID).sum(dim=1)
        new_total_lens = old_lens - remove_lens + repl_lengths
        
        # Filter those that fit
        fits = new_total_lens <= self.max_len
        
        if not fits.any():
            return population
            
        # Apply only to fitting ones
        # Indices relative to mutant_indices
        valid_idx_rel = torch.nonzero(fits).squeeze(1) 
        
        final_mutant_pop_idx = mutant_indices[valid_idx_rel] # Indices in original pop
        
        # We construct the new sequences for these
        orig_seqs = population[final_mutant_pop_idx]
        new_seqs = replacements[valid_idx_rel]
        
        st = starts[valid_idx_rel]
        en = ends[valid_idx_rel]
        r_len = repl_lengths[valid_idx_rel]
        
        # Construct
        grid = torch.arange(L, device=self.device).unsqueeze(0).expand(len(valid_idx_rel), L)
        
        # Output buffer
        out_seqs = torch.full_like(orig_seqs, PAD_ID)
        
        # Mask Pre: idx < st
        mask_pre = grid < st.unsqueeze(1)
        out_seqs[mask_pre] = orig_seqs[mask_pre]
        
        # Mask New: idx >= st and idx < st + r_len
        limit_new = st + r_len
        mask_new = (grid >= st.unsqueeze(1)) & (grid < limit_new.unsqueeze(1))
        
        # Extract new content
        # idx in new_seqs = grid - st
        gather_idx = grid - st.unsqueeze(1)
        gather_idx = torch.clamp(gather_idx, 0, L-1)
        val_new = new_seqs.gather(1, gather_idx)
        out_seqs[mask_new] = val_new[mask_new]
        
        # Mask Post: idx >= limit_new
        # Source Post start at en + 1
        # Target Post start at st + r_len
        # Shift = (en + 1) - (st + r_len)
        shift = (en + 1) - (st + r_len)
        src_idx_post = grid + shift.unsqueeze(1)
        
        mask_post = (grid >= limit_new.unsqueeze(1))
        # Ensure source valid (within [0, L-1])
        valid_src = (src_idx_post < L) #& (src_idx_post >= 0)
        mask_post = mask_post & valid_src
        
        safe_src = torch.clamp(src_idx_post, 0, L-1)
        val_post = orig_seqs.gather(1, safe_src)
        
        out_seqs[mask_post] = val_post[mask_post]
        
        # Write back
        population[final_mutant_pop_idx] = out_seqs
        
        return population

    def repair_invalid_population(self, population: torch.Tensor, constants: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Replace invalid RPN individuals with fresh random individuals.
        This prevents evolutionary collapse to trivial fallbacks.
        """
        valid_mask = self._validate_rpn_batch(population)
        invalid_mask = ~valid_mask
        n_invalid = int(invalid_mask.sum().item())
        if n_invalid == 0:
            return population, constants, 0

        invalid_idx = invalid_mask.nonzero(as_tuple=True)[0]
        population[invalid_idx] = self.generate_random_population(n_invalid)

        if constants is not None and constants.numel() > 0:
            k = constants.shape[1]
            if GpuGlobals.FORCE_INTEGER_CONSTANTS:
                fresh_consts = torch.randint(
                    GpuGlobals.CONSTANT_INT_MIN_VALUE,
                    GpuGlobals.CONSTANT_INT_MAX_VALUE + 1,
                    (n_invalid, k), device=self.device, dtype=torch.long
                ).to(self.dtype)
            else:
                fresh_consts = torch.empty(n_invalid, k, device=self.device, dtype=self.dtype).uniform_(
                    GpuGlobals.CONSTANT_MIN_VALUE,
                    GpuGlobals.CONSTANT_MAX_VALUE
                )
            constants[invalid_idx] = fresh_consts

        return population, constants, n_invalid
