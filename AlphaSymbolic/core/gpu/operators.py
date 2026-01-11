
import torch
import numpy as np
from typing import List, Tuple
from core.grammar import OPERATORS, ExpressionTree
from .grammar import PAD_ID, GPUGrammar
from .config import GpuGlobals

class GPUOperators:
    def __init__(self, grammar: GPUGrammar, device, pop_size, max_len=30, num_variables=1):
        self.grammar = grammar
        self.device = device
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

    def generate_random_population(self, size: int) -> torch.Tensor:
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
        
        # Helper to convert infix strings to RPN tensor
        # Since we might not have the converter here, we can rely on a utility or pass it.
        # But for valid RPN generation, using the Parser is best.
        # We can reimplement simple infix_to_rpn or rely on one passed in.
        # Or better: ExpressionTree has post-order traversal logic?
        # Actually `engine.py` had `infix_to_rpn` using `ExpressionTree`.
        # Let's include that logic here as a privatish method or utility.
        return self._infix_list_to_rpn(formulas)

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
        mask = torch.rand_like(population, dtype=torch.float32) < mutation_rate
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
        B, L = population.shape
        subtree_starts = torch.full((B, L), -1, device=self.device, dtype=torch.long)
        
        # Arities map
        # Variable/Const/0-Arity -> 1 (Push 1)
        # Unary -> 0 (Pop 1 Push 1)
        # Binary -> -1 (Pop 2 Push 1)
        
        # We need efficient lookup. Using a tensor map is best.
        # self.token_arity has 0/1/2.
        # Net Stack Delta = 1 - Arity
        # 0 -> +1
        # 1 -> 0
        # 2 -> -1
        
        token_net_change = 1 - self.token_arity
        
        arities = token_net_change[population]
        arities[population == PAD_ID] = -999 # Invalid
        
        # Cumulative Sum (Stack Depth Profile)
        safe_arities = arities.clone()
        safe_arities[population == PAD_ID] = 0
        depths = torch.cumsum(safe_arities, dim=1)
        
        for i in range(L):
            is_pad = (population[:, i] == PAD_ID)
            
            # Target: depth[start-1] = depth[i] - 1
            target_depth = depths[:, i] - 1
            
            current_start = torch.full((B,), -1, device=self.device, dtype=torch.long)
            found = torch.zeros(B, dtype=torch.bool, device=self.device)
            
            for j in range(i, -1, -1):
                prev_depth = depths[:, j-1] if j > 0 else torch.zeros(B, device=self.device)
                match = (prev_depth == target_depth)
                new_found = match & (~found)
                current_start[new_found] = j
                found = found | new_found
            
            valid_i = (~is_pad) & found
            subtree_starts[valid_i, i] = current_start[valid_i]
            
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
        fresh_consts = torch.randn(n_dups, K, device=self.device, dtype=torch.float64)
        
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
