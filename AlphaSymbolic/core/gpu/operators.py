
import torch
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
        self.max_len = max_len
        self.num_variables = num_variables
        self.pop_size = pop_size
        
        # Pre-allocate memory for random generation
        self.terminal_ids = torch.tensor([self.grammar.token_to_id[t] for t in self.grammar.terminals], device=self.device)
        self.operator_ids = torch.tensor([self.grammar.token_to_id[op] for op in self.grammar.operators], device=self.device)
        
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
            
        self.arity_0_ids = torch.tensor(self.arity_0_ids, device=self.device)
        self.arity_1_ids = torch.tensor(self.arity_1_ids, device=self.device)
        self.arity_2_ids = torch.tensor(self.arity_2_ids, device=self.device)
        
        # Cache int32 arities for CUDA
        self.token_arity_int = self.token_arity.to(dtype=torch.int32)

    def generate_random_population(self, size: int) -> torch.Tensor:
        """
        Helper to generate random RPN population of given size.
        Uses GPU-native generation for speed.
        """
        # Use GPU native generator (fast path)
        return self.generate_random_population_gpu(size)
    
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
        GPU-native random RPN generation. No CPU bottleneck.
        
        Algorithm:
        1. Start with stack=0. We need stack=1 at the end.
        2. For each position, sample a token that keeps the stack valid.
        3. Terminals add +1 to stack, arity-1 ops add 0, arity-2 ops add -1.
        4. We need to ensure we can always reach stack=1 at the end.
        """
        max_len = self.max_len
        device = self.device
        
        # Output tensor
        population = torch.zeros(size, max_len, dtype=torch.long, device=device)
        
        # Stack balance tracker: starts at 0, needs to end at 1
        stack = torch.zeros(size, dtype=torch.long, device=device)
        
        # Token pools with their stack delta (terminals=+1, arity1=0, arity2=-1)
        # Combine all valid tokens into a single pool with weights
        n_terminals = self.terminal_ids.shape[0]
        n_arity1 = self.arity_1_ids.shape[0]
        n_arity2 = self.arity_2_ids.shape[0]
        
        # Create combined token pool
        all_tokens = torch.cat([self.terminal_ids, self.arity_1_ids, self.arity_2_ids])
        n_total = all_tokens.shape[0]
        
        # Stack delta for each token type
        # terminals: +1, arity1: 0 (pops 1, pushes 1), arity2: -1 (pops 2, pushes 1)
        token_deltas = torch.cat([
            torch.ones(n_terminals, dtype=torch.long, device=device),   # terminals: +1
            torch.zeros(n_arity1, dtype=torch.long, device=device),     # arity1: 0
            -torch.ones(n_arity2, dtype=torch.long, device=device)      # arity2: -1
        ])
        
        # For each position, we need to sample valid tokens
        for j in range(max_len):
            remaining = max_len - j - 1  # positions left after this one
            
            # For each individual, determine which tokens are valid
            # Rule: After sampling, the new stack must be >= 0
            #       AND we must be able to reach stack=1 with remaining positions
            
            # Stack after sampling token k: stack + delta[k]
            # With 'remaining' positions, we can add at most +remaining to stack (all terminals)
            # and at least -remaining (not realistic, but lower bound)
            
            # To reach stack=1:
            # new_stack + (adjustment from remaining) = 1
            # We need: new_stack >= 0 (valid RPN)
            # We need: new_stack <= 1 + remaining (can reduce with operators)
            # We need: new_stack >= 1 - remaining (can increase with terminals)
            
            # Create validity mask for each (individual, token) pair
            # new_stack = stack.unsqueeze(1) + token_deltas.unsqueeze(0)  # [size, n_total]
            new_stack = stack.unsqueeze(1) + token_deltas.unsqueeze(0)
            
            # Constraints
            # We need new_stack >= 1 to ensure we had enough operands
            # (Stack 0 -> Term(+1) -> 1. Stack 1 -> Op1(0) -> 1. Stack 2 -> Op2(-1) -> 1)
            # Exception: None? All ops produce stack >= 1 if valid.
            valid_mask = (new_stack >= 1) & \
                         (new_stack <= 1 + remaining) & \
                         (new_stack >= 1 - remaining)
            
            # At position 0, stack=0, so we MUST pick a terminal (delta=+1)
            # At final position (remaining=0), we MUST have new_stack=1
            if remaining == 0:
                valid_mask = valid_mask & (new_stack == 1)
            
            # Convert mask to sampling weights (0 for invalid, 1 for valid)
            weights = valid_mask.float()
            
            # Ensure at least one valid option per row
            # If no valid tokens, fall back to first terminal
            row_sums = weights.sum(dim=1, keepdim=True)
            fallback_mask = (row_sums == 0)
            if fallback_mask.any():
                weights[:, 0] = weights[:, 0] + fallback_mask.squeeze().float()
            
            # Normalize weights to probabilities
            probs = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            
            # Sample token index for each individual
            sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Get actual token IDs and their deltas
            sampled_tokens = all_tokens[sampled_idx]
            sampled_deltas = token_deltas[sampled_idx]
            
            # Update population and stack
            population[:, j] = sampled_tokens
            stack = stack + sampled_deltas
        
        # Final validation: stack should be exactly 1
        valid = (stack == 1)
        n_invalid = (~valid).sum().item()
        
        if n_invalid > 0:
            # Replace invalid with simple "x0" formula
            x0_id = self.grammar.token_to_id.get('x0', self.grammar.token_to_id.get('x', 1))
            population[~valid, 0] = x0_id
            population[~valid, 1:] = PAD_ID
        
        return population

    def _validate_rpn_batch(self, population: torch.Tensor) -> torch.Tensor:
        """
        Validate that each RPN formula has a final stack balance of exactly 1.
        Returns a boolean mask of valid formulas.
        """
        B, L = population.shape
        stack = torch.zeros(B, dtype=torch.long, device=self.device)
        
        for j in range(L):
            tokens = population[:, j]
            # Get arity for each token (0 for PAD)
            arity = self.token_arity[tokens.clamp(0, self.token_arity.shape[0] - 1)]
            # Stack delta: -(arity) + 1 for each token (pops arity, pushes 1)
            # But PAD should have delta 0
            is_pad = (tokens == PAD_ID)
            delta = torch.where(is_pad, torch.zeros_like(arity), 1 - arity)
            stack = stack + delta
        
        return (stack == 1)

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
            return torch.empty((0, self.max_len), device=self.device, dtype=torch.long)
        return torch.tensor(batch_rpn, device=self.device, dtype=torch.long)

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
             c1 = torch.full((n_pairs, L), PAD_ID, dtype=torch.long, device=self.device)
             c2 = torch.full((n_pairs, L), PAD_ID, dtype=torch.long, device=self.device)
             
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
            
        pop_out = population.clone()
        const_out = constants.clone()
        
        # Generate replacements
        fresh_pop = self.generate_random_population(n_dups)
        
        # Constants for replacements?
        # If constants are [B, K]. 
        # fresh_pop needs fresh constants.
        K = constants.shape[1]
        fresh_consts = torch.randn(n_dups, K, device=self.device, dtype=self.dtype)
        
        pop_out[dup_indices] = fresh_pop
        const_out[dup_indices] = fresh_consts
        
        return pop_out, const_out, n_dups

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
        
        n_mut = mutate_mask.sum().item()
        if n_mut == 0:
            return population
            
        # Indices of mutants
        mutant_indices = torch.nonzero(mutate_mask).squeeze(1)
        
        # Select random end point for each mutant
        # We use multinomial on valid positions
        probs = valid_mask[mutate_mask].float() + 1e-6
        ends = torch.multinomial(probs, 1).squeeze(1) # [n_mut]
        
        starts = starts_mat[mutate_mask].gather(1, ends.unsqueeze(1)).squeeze(1) # [n_mut]
        
        # Lengths of subtrees to remove
        remove_lens = ends - starts + 1
        
        # Generate NEW random subtrees (RPN)
        # Depth 2-4 is usually good for subtrees
        # We use generate_random_population but force small trees if possible?
        # Standard gen uses MAX_TREE_DEPTH_INITIAL (8?). A bit large for subtree.
        # But we filter by length below.
        
        # Generate replacements (over-generate slightly if needed? No, just generate N)
        replacements = self.generate_random_population(n_mut) # [n_mut, L]
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
