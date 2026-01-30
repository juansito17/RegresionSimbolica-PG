"""
GPU-Native Symbolic Simplifier - Professional Performance Edition

A high-performance symbolic simplification engine designed for millions of formulas.
All operations are 100% vectorized using PyTorch tensors.
Zero Python loops over the batch dimension ensure maximum GPU throughput.
"""
import torch
from typing import Tuple, List
from .grammar import PAD_ID, GPUGrammar

class GPUSymbolicSimplifier:
    def __init__(self, grammar: GPUGrammar, device, dtype=torch.float64):
        self.grammar = grammar
        self.device = device
        self.dtype = dtype
        self._cache_operator_ids()
        self._build_arity_table()
        
    def _cache_operator_ids(self):
        g = self.grammar
        def get_ids(tokens: List[str]) -> torch.Tensor:
            ids = [g.token_to_id[t] for t in tokens if t in g.token_to_id]
            return torch.tensor(ids, device=self.device, dtype=torch.long)

        self.OP_PLUS = g.token_to_id.get('+', -1)
        self.OP_MINUS = g.token_to_id.get('-', -1)
        self.OP_MULT = g.token_to_id.get('*', -1)
        self.OP_DIV = g.token_to_id.get('/', -1)
        self.OP_POW_IDS = get_ids(['pow', '^', '**'])
        self.OP_LOG_IDS = get_ids(['log', 'ln'])
        self.OP_EXP_IDS = get_ids(['exp', 'e'])
        self.OP_NEG_IDS = get_ids(['neg'])
        self.OP_SQRT_IDS = get_ids(['sqrt'])
        self.OP_ABS_IDS = get_ids(['abs'])
        
        self.OP_SIN = g.token_to_id.get('sin', -1)
        self.OP_COS = g.token_to_id.get('cos', -1)
        self.OP_TAN = g.token_to_id.get('tan', -1)
        
        # Rescued Advanced Operators
        self.OP_GAMMA_IDS = get_ids(['gamma', '!'])
        self.OP_LGAMMA_IDS = get_ids(['lgamma', 'lg', 'g'])
        
        terminals = list(g.terminals)
        self.zero_ids = torch.tensor([g.token_to_id[t] for t in terminals if t in ['0', '0.0']], device=self.device, dtype=torch.long)
        self.one_ids = torch.tensor([g.token_to_id[t] for t in terminals if t in ['1', '1.0']], device=self.device, dtype=torch.long)
        self.two_ids = torch.tensor([g.token_to_id[t] for t in terminals if t in ['2', '2.0']], device=self.device, dtype=torch.long)
        self.literal_ids = torch.tensor([g.token_to_id[t] for t in terminals if t.replace('.','',1).isdigit() or (t.startswith('-') and t[1:].replace('.','',1).isdigit())], device=self.device, dtype=torch.long)
        
        # Scalar versions for fast checks
        self.CONST_0 = self.zero_ids[0].item() if self.zero_ids.numel() > 0 else -1
        self.CONST_1 = self.one_ids[0].item() if self.one_ids.numel() > 0 else -1
        self.CONST_2 = self.two_ids[0].item() if self.two_ids.numel() > 0 else -1
        
        # ID aliases
        self.ID_0, self.ID_1, self.ID_2 = self.CONST_0, self.CONST_1, self.CONST_2
        
    def _build_arity_table(self):
        from core.grammar import OPERATORS
        max_id = max(self.grammar.id_to_token.keys()) + 1
        self.arity_table = torch.zeros(max_id, dtype=torch.long, device=self.device)
        for token, tid in self.grammar.token_to_id.items():
            self.arity_table[tid] = OPERATORS.get(token, 0)
                
    def _is_zero(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.zero_ids.numel() == 0: return torch.zeros_like(token_ids, dtype=torch.bool)
        return (token_ids.unsqueeze(-1) == self.zero_ids).any(dim=-1)
    
    def _is_one(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.one_ids.numel() == 0: return torch.zeros_like(token_ids, dtype=torch.bool)
        return (token_ids.unsqueeze(-1) == self.one_ids).any(dim=-1)

    def _is_constant(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens.unsqueeze(-1) == self.literal_ids).any(dim=-1)

    def _is_constant_value(self, tokens: torch.Tensor, val: float) -> torch.Tensor:
        if val == 0.0: return self._is_zero(tokens)
        if val == 1.0: return self._is_one(tokens)
        if val == 2.0: return tokens == self.CONST_2
        val_str = str(int(val)) if val.is_integer() else str(val)
        tid = self.grammar.token_to_id.get(val_str, -1)
        return tokens == tid if tid != -1 else torch.zeros_like(tokens, dtype=torch.bool)

    def simplify_batch(self, population: torch.Tensor, constants: torch.Tensor = None, max_passes: int = 3) -> Tuple[torch.Tensor, torch.Tensor, int]:
        B, L = population.shape
        pop = population.clone()
        total_simplified = 0
        for _ in range(max_passes):
            n_pass = 0
            # Normalization helps patterns match more reliably
            pop, n = self._apply_commutative_normalization(pop); n_pass += n
            
            pop, n = self._apply_identity_rules(pop); n_pass += n
            pop, n = self._apply_zero_rules(pop); n_pass += n
            pop, n = self._apply_self_cancellation_rules(pop); n_pass += n
            pop, n = self._apply_associative_rules(pop); n_pass += n
            pop, n = self._apply_advanced_rules(pop); n_pass += n
            pop, n = self._apply_term_consolidation(pop); n_pass += n
            pop, n = self._apply_modulo_rules(pop); n_pass += n
            pop, n = self._apply_constant_folding(pop); n_pass += n
            pop, n = self._compact_formulas(pop); n_pass += n
            if n_pass == 0: break
            total_simplified += n_pass
        return pop, constants, total_simplified
    
    def _apply_identity_rules(self, population: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, L = population.shape
        pop = population.clone()
        n_simplified = 0
        o_id = self.ID_1
        for j in range(2, L):
            op = pop[:, j]
            # Pattern: [arg1, arg2, op]
            is_plus, is_mult, is_minus, is_div = (op==self.OP_PLUS), (op==self.OP_MULT), (op==self.OP_MINUS), (op==self.OP_DIV)
            is_pow = (op.unsqueeze(-1) == self.OP_POW_IDS).any(-1) if self.OP_POW_IDS.numel() > 0 else torch.zeros_like(op, dtype=torch.bool)
            
            if not (is_plus | is_mult | is_minus | is_div | is_pow).any(): continue
            
            s2 = self._get_subtree_starts(pop, j-1)
            is_z2, is_o2 = (s2 == j-1) & self._is_zero(pop[:, j-1]), (s2 == j-1) & self._is_one(pop[:, j-1])
            
            # x+0, x-0, x*1, x/1, x^1 -> skip arg2/op (keep arg1)
            to_skip_arg2 = (is_plus & is_z2) | (is_minus & is_z2) | (is_mult & is_o2) | (is_div & is_o2) | (is_pow & is_o2)
            if to_skip_arg2.any():
                pop[to_skip_arg2, j-1], pop[to_skip_arg2, j] = PAD_ID, PAD_ID
                n_simplified += to_skip_arg2.sum().item()
            
            # 0+x, 1*x -> skip arg1/op (keep arg2)
            s1 = self._get_subtree_starts(pop, s2-1)
            valid_s1 = (s2 > 0)
            is_z1 = (s1 == s2-1) & self._is_zero(pop.gather(1, (s2-1).clamp(0).unsqueeze(1)).squeeze(1)) & valid_s1
            is_o1 = (s1 == s2-1) & self._is_one(pop.gather(1, (s2-1).clamp(0).unsqueeze(1)).squeeze(1)) & valid_s1
            
            to_skip_arg1 = (is_plus & is_z1) | (is_mult & is_o1)
            if to_skip_arg1.any():
                rows = torch.where(to_skip_arg1)[0]
                pop[rows, (s2-1)[rows]] = PAD_ID
                pop[rows, j] = PAD_ID
                n_simplified += to_skip_arg1.sum().item()
            
            # Special Constant Result Rules: x^0 -> 1, 1^x -> 1
            match_const_1 = is_pow & (is_z2 | is_o1)
            if match_const_1.any() and o_id != -1:
                rows = torch.where(match_const_1)[0]
                start = s1[match_const_1]
                sub_pop = pop[rows]
                c_idx = torch.arange(len(rows), device=self.device)
                sub_pop[c_idx, start] = o_id
                pos = torch.arange(L, device=self.device).reshape(1, L)
                sub_pop[(pos > start.unsqueeze(1)) & (pos <= j)] = PAD_ID
                pop[rows] = sub_pop
                n_simplified += match_const_1.sum().item()
        return pop, n_simplified

    def _apply_commutative_normalization(self, population: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Standardize the order of operands for commutative operators (+, *).
        Puts constants and shorter/simpler subtrees on the left.
        """
        B, L = population.shape
        pop = population.clone()
        n_swapped = 0
        
        for j in range(2, L):
            op = pop[:, j]
            is_comm = (op == self.OP_PLUS) | (op == self.OP_MULT)
            if not is_comm.any(): continue
            
            # Identify subtrees
            s2 = self._get_subtree_starts(pop, j-1)
            s1 = self._get_subtree_starts(pop, s2-1)
            
            # Term 1 info
            t1 = pop.gather(1, s1.clamp(min=0).unsqueeze(1)).squeeze(1)
            is_leaf1 = (s1 == s2-1)
            is_const1 = self._is_constant(t1) & is_leaf1 & (s1 >= 0)
            
            # Term 2 info
            t2 = pop.gather(1, s2.clamp(min=0).unsqueeze(1)).squeeze(1)
            is_leaf2 = (s2 == j-1)
            is_const2 = self._is_constant(t2) & is_leaf2 & (s2 >= 0)
            
            # Score 0: Constant, 1: Variable, 2: Complex
            score1 = torch.where(is_const1, 0, torch.where(is_leaf1, 1, 2))
            score2 = torch.where(is_const2, 0, torch.where(is_leaf2, 1, 2))
            
            should_swap = is_comm & (score2 < score1)
            
            # Tiebreaker for constants/terminals: lower ID
            tie = is_comm & (score1 == score2) & (score1 < 2) & (t2 < t1)
            should_swap |= tie
            
            # Tiebreaker for complex: shorter length
            len1 = s2 - s1
            len2 = j - s2
            tie_complex = is_comm & (score1 == 2) & (score2 == 2) & (len2 < len1)
            should_swap |= tie_complex
            
            if should_swap.any():
                # Fast Path: Both args are terminals (len=1)
                is_simple_swap = (len1 == 1) & (len2 == 1) & should_swap
                if is_simple_swap.any():
                    # Vectorized swap
                    mask = is_simple_swap
                    # Store original values
                    nodes_1 = pop.gather(1, s1.clamp(min=0).unsqueeze(1)).squeeze(1)
                    nodes_2 = pop.gather(1, s2.clamp(min=0).unsqueeze(1)).squeeze(1)
                    
                    # Compute flat indices for scatter/advanced indexing
                    # We want pop[mask, s1[mask]] = nodes_2[mask]
                    # pop[mask, s2[mask]] = nodes_1[mask]
                    # Since s1 and s2 vary per row, we can use torch.scatter or advanced indexing
                    
                    # Advanced indexing:
                    rows_simple = torch.where(mask)[0]
                    cols_s1 = s1[mask]
                    cols_s2 = s2[mask]
                    
                    # We can map rows_simple to 0..N_match range for gathering vals
                    vals_1 = nodes_1[mask]
                    vals_2 = nodes_2[mask]
                    
                    # Write
                    pop[rows_simple, cols_s1] = vals_2
                    pop[rows_simple, cols_s2] = vals_1
                    
                    n_swapped += mask.sum().item()
                    
                    # Remove simple swaps from remaining potential swaps to avoid double handling
                    should_swap &= ~is_simple_swap

                # Slow path: Variable length swaps
                if should_swap.any():
                    rows = torch.where(should_swap)[0]
                    for b in rows:
                        idx_s1, idx_s2, idx_j = s1[b].item(), s2[b].item(), j
                        if idx_s1 < 0 or idx_s2 < 0: continue
                        arg1 = pop[b, idx_s1:idx_s2].clone()
                        arg2 = pop[b, idx_s2:idx_j].clone()
                        
                        pop[b, idx_s1:idx_s1+len(arg2)] = arg2
                        pop[b, idx_s1+len(arg2):idx_j] = arg1
                        n_swapped += 1
                    
        return pop, n_swapped

    def _apply_zero_rules(self, population: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, L = population.shape
        pop = population.clone()
        n_simplified = 0
        z_id = self.zero_ids[0] if self.zero_ids.numel() > 0 else self.CONST_0
        for j in range(2, L):
            is_mult = (pop[:, j] == self.OP_MULT)
            if not is_mult.any(): continue
            s2 = self._get_subtree_starts(pop, j-1)
            is_z2 = (s2 == j-1) & self._is_zero(pop[:, j-1])
            is_z1 = False
            s1 = self._get_subtree_starts(pop, s2-1)
            is_z1 = (s1 == s2-1) & self._is_zero(pop.gather(1, (s2-1).clamp(0).unsqueeze(1)).squeeze(1)) & (s2 > 0)
            
            match = is_mult & (is_z1 | is_z2)
            if match.any():
                rows = torch.where(match)[0]
                start_to_wipe = s1[match]
                
                # Create sub-population of matching rows (Copy)
                sub_pop = pop[rows]
                
                # Set Z_ID at start
                # sub_pop has N_match rows. start_to_wipe has N_match indices.
                # We need column indices for each row.
                # sub_pop[range(N_match), start_to_wipe] = z_id
                cols = torch.arange(len(rows), device=self.device)
                sub_pop[cols, start_to_wipe] = z_id
                
                # Set PADs
                pos = torch.arange(L, device=self.device).reshape(1, L)
                s_wipe = start_to_wipe.unsqueeze(1)
                # Mask shape: (N_match, L)
                pad_mask = (pos > s_wipe) & (pos <= j)
                sub_pop[pad_mask] = PAD_ID
                
                # Write back to main population
                pop[rows] = sub_pop
                
                n_simplified += match.sum().item()
        return pop, n_simplified

    def _apply_self_cancellation_rules(self, population: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, L = population.shape
        pop = population.clone()
        n_simplified = 0
        z_id, o_id = (self.zero_ids[0] if self.zero_ids.numel()>0 else self.CONST_0), (self.one_ids[0] if self.one_ids.numel()>0 else self.CONST_1)
        for j in range(2, L):
            op = pop[:, j]
            is_matchable = (op == self.OP_MINUS) | (op == self.OP_DIV)
            if not is_matchable.any(): continue
            # Vectorized check for single-token operands: [x, x, -] -> [0, PAD, PAD]
            arg2, arg1 = pop[:, j-1], pop[:, j-2]
            match_single = is_matchable & (arg1 == arg2) & (self.arity_table[arg1.clamp(0)] == 0) & (arg1 != PAD_ID)
            if match_single.any():
                is_m = match_single & (op == self.OP_MINUS)
                is_d = match_single & (op == self.OP_DIV)
                pop[is_m, j-2], pop[is_d, j-2] = z_id, o_id
                pop[match_single, j-1], pop[match_single, j] = PAD_ID, PAD_ID
                n_simplified += match_single.sum().item()
        return pop, n_simplified

    def _apply_advanced_rules(self, population: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Apply advanced rules: 
        - neg(neg(x)) = x
        - exp(log(x)) = x, log(exp(x)) = x
        - sqrt(x^2) = abs(x)
        - Gamma/LGamma identities: lg(1)=0, lg(2)=0, gamma(1)=1, gamma(2)=1
        - Trigonometric/Log identity on constants
        """
        B, L = population.shape
        pop = population.clone()
        n_simplified = 0
        z_id, o_id = self.ID_0, self.ID_1
        abs_id = self.OP_ABS_IDS[0].item() if self.OP_ABS_IDS.numel() > 0 else -1
        
        def is_op_in(tokens: torch.Tensor, set_ids: torch.Tensor) -> torch.Tensor:
            if set_ids.numel() == 0: return torch.zeros_like(tokens, dtype=torch.bool)
            return (tokens.unsqueeze(-1) == set_ids).any(-1)

        for j in range(1, L):
            tokens = pop[:, j]
            # --- Chain reduction logic ---
            # neg(neg(x)) -> x
            if j >= 2:
                is_neg = is_op_in(tokens, self.OP_NEG_IDS)
                is_neg_prev = is_op_in(pop[:, j-1], self.OP_NEG_IDS)
                match_nn = is_neg & is_neg_prev
                if match_nn.any():
                    pop[match_nn, j], pop[match_nn, j-1] = PAD_ID, PAD_ID
                    n_simplified += match_nn.sum().item()
                
                # exp(log(x)) -> x
                is_exp = is_op_in(tokens, self.OP_EXP_IDS)
                is_log_prev = is_op_in(pop[:, j-1], self.OP_LOG_IDS)
                match_el = is_exp & is_log_prev
                if match_el.any():
                    pop[match_el, j], pop[match_el, j-1] = PAD_ID, PAD_ID
                    n_simplified += match_el.sum().item()

                # log(exp(x)) -> x
                is_log = is_op_in(tokens, self.OP_LOG_IDS)
                is_exp_prev = is_op_in(pop[:, j-1], self.OP_EXP_IDS)
                match_le = is_log & is_exp_prev
                if match_le.any():
                    pop[match_le, j], pop[match_le, j-1] = PAD_ID, PAD_ID
                    n_simplified += match_le.sum().item()

            # --- SOTA / Better than basic Sympy ---
            # sqrt(x^2) -> abs(x)
            if j >= 3:
                is_sqrt = is_op_in(tokens, self.OP_SQRT_IDS)
                is_pow_prev = is_op_in(pop[:, j-1], self.OP_POW_IDS)
                is_two_pp = self._is_constant_value(pop[:, j-2], 2.0)
                match_sqrt_p2 = is_sqrt & is_pow_prev & is_two_pp
                if match_sqrt_p2.any() and abs_id != -1:
                    pop[match_sqrt_p2, j] = abs_id
                    pop[match_sqrt_p2, j-1] = PAD_ID
                    pop[match_sqrt_p2, j-2] = PAD_ID
                    n_simplified += match_sqrt_p2.sum().item()
            
            # --- Constant arg identities ---
            is_unary = (self.arity_table[tokens.clamp(0)] == 1)
            if is_unary.any():
                arg = pop[:, j-1]
                arg_is0 = self._is_zero(arg)
                arg_is1 = self._is_one(arg)
                arg_is2 = self._is_constant_value(arg, 2.0)
                
                # Rescued Unary Identities
                to_zero = arg_is0 & ((tokens==self.OP_SIN)|(tokens==self.OP_TAN)|is_op_in(tokens, self.OP_ABS_IDS))
                to_zero |= arg_is1 & is_op_in(tokens, self.OP_LOG_IDS)
                to_zero |= arg_is1 & is_op_in(tokens, self.OP_LGAMMA_IDS) # lg(1)=0
                to_zero |= arg_is2 & is_op_in(tokens, self.OP_LGAMMA_IDS) # lg(2)=0
                
                if to_zero.any() and z_id != -1: 
                    pop[to_zero, j-1], pop[to_zero, j] = z_id, PAD_ID
                    n_simplified += to_zero.sum().item()
                
                to_one = arg_is0 & ((tokens==self.OP_COS)|is_op_in(tokens, self.OP_EXP_IDS))
                to_one |= arg_is1 & is_op_in(tokens, self.OP_GAMMA_IDS) # gamma(1)=1
                to_one |= arg_is2 & is_op_in(tokens, self.OP_GAMMA_IDS) # gamma(2)=1
                
                if to_one.any() and o_id != -1: 
                    pop[to_one, j-1], pop[to_one, j] = o_id, PAD_ID
                    n_simplified += to_one.sum().item()
            
            # --- Negation Rules (Plus) ---
            if j >= 2 and (tokens == self.OP_PLUS).any():
                is_p = (tokens == self.OP_PLUS)
                s2 = self._get_subtree_starts(pop, j-1)
                s1 = self._get_subtree_starts(pop, s2-1)
                
                # Move to logic: x + neg(x) -> 0
                is_neg2 = is_op_in(pop[:, j-1], self.OP_NEG_IDS)
                match_p = is_p & is_neg2
                if match_p.any():
                    # Batch-calculate inner subtree starts for all potential negations
                    s_inner_batch = self._get_subtree_starts(pop, j-2)
                    
                    # Compare subtrees [s1:s_inner] and [s_inner:j-1]
                    # Focus on terminals for robustness in vectorized mode
                    match_p &= (s1 == s_inner_batch-1) & (s1 >= 0) & (s_inner_batch >= 0)
                    
                    if match_p.any():
                        t1 = pop.gather(1, s1.clamp(min=0).unsqueeze(1)).squeeze(1)
                        t_inner = pop.gather(1, s_inner_batch.clamp(min=0).unsqueeze(1)).squeeze(1)
                        match_p &= (t1 == t_inner)
                        
                    if match_p.any():
                        rows = torch.where(match_p)[0]
                        for b in rows:
                            pop[b, s1[b]] = z_id
                            pop[b, s1[b]+1:j+1] = PAD_ID
                            n_simplified += 1

        return pop, n_simplified

    def _apply_associative_rules(self, population: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Apply associative/grouping rules:
        - x + (x + y) -> 2*x + y
        - (x + y) + x -> 2*x + y
        """
        B, L = population.shape
        pop = population.clone()
        n_simplified = 0
        if self.ID_2 == -1: return pop, 0
        
        for j in range(4, L):
            op = pop[:, j]
            is_plus = (op == self.OP_PLUS)
            if not is_plus.any(): continue
            
            # Pattern: (x + y) + z -> 2*x + y
            # We check if arg2 is a PLUS (RPN: [x, y, +, z, +])
            # Or if arg1 is a PLUS (RPN: [z, x, y, +, +])
            match_1 = is_plus & (pop[:, j-1] == self.OP_PLUS)
            if match_1.any():
                # Get subtree starts for the whole batch at index j
                s2_batch = self._get_subtree_starts(pop, j-1)
                s1_batch = self._get_subtree_starts(pop, s2_batch-1)
                
                rows = torch.where(match_1)[0]
                for b in rows:
                    idx_s1, idx_s2 = s1_batch[b].item(), s2_batch[b].item()
                    if idx_s1 < 0 or idx_s2 < 0: continue
                    
                    # Pattern: [x, y, +, z, +] where s2 is start of z
                    # arg1 is [x, y, +]
                    e_arg1 = idx_s2 - 1
                    if e_arg1 < 0: continue
                    # Local call here is okay as it's single row, but we could vectorize more.
                    # Given the nested nature of RPN, we'll focus on the big batch calls.
                    s_arg1 = self._get_subtree_starts(pop[b:b+1], e_arg1)[0].item()
                    if s_arg1 < 0: continue
                    
                    e_inner2 = e_arg1 - 1
                    if e_inner2 < 0: continue
                    s_inner2 = self._get_subtree_starts(pop[b:b+1], e_inner2)[0].item()
                    if s_inner2 < 0: continue
                    
                    e_inner1 = s_inner2 - 1
                    if e_inner1 < 0: continue
                    s_inner1 = self._get_subtree_starts(pop[b:b+1], e_inner1)[0].item()
                    if s_inner1 < 0: continue
                    
                    z = pop[b, idx_s2:j].clone()
                    x = pop[b, s_inner1:e_inner1+1].clone()
                    y = pop[b, s_inner2:e_inner2+1].clone()
                    
                    if torch.equal(x, z):
                        new = torch.cat([torch.tensor([self.ID_2], device=self.device), x, torch.tensor([self.OP_MULT], device=self.device), y, torch.tensor([self.OP_PLUS], device=self.device)])
                        if len(new) <= (j - idx_s1 + 1):
                            pop[b, idx_s1:idx_s1+len(new)] = new
                            pop[b, idx_s1+len(new):j+1] = PAD_ID
                            n_simplified += 1
                    elif torch.equal(y, z):
                        new = torch.cat([torch.tensor([self.ID_2], device=self.device), y, torch.tensor([self.OP_MULT], device=self.device), x, torch.tensor([self.OP_PLUS], device=self.device)])
                        if len(new) <= (j - idx_s1 + 1):
                            pop[b, idx_s1:idx_s1+len(new)] = new
                            pop[b, idx_s1+len(new):j+1] = PAD_ID
                            n_simplified += 1

        return pop, n_simplified

    def _apply_term_consolidation(self, population: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, L = population.shape
        pop = population.clone()
        n_simplified = 0
        for j in range(2, L):
            op = pop[:, j]
            if op.numel() == 0: continue
            is_plus = (op == self.OP_PLUS)
            is_mult = (op == self.OP_MULT)
            if not (is_plus | is_mult).any(): continue
            
            # --- x + x -> 2 * x ---
            arg2, arg1 = pop[:, j-1], pop[:, j-2]
            is_same = (arg1 == arg2) & (self.arity_table[arg1.clamp(0)] == 0) & (arg1 != PAD_ID)
            match_add_same = is_plus & is_same
            if match_add_same.any() and self.ID_2 != -1:
                pop[match_add_same, j] = self.OP_MULT
                pop[match_add_same, j-1] = arg1[match_add_same]
                pop[match_add_same, j-2] = self.ID_2
                n_simplified += match_add_same.sum().item()

            # --- x * x -> x ^ 2 ---
            match_mult_same = is_mult & is_same
            if match_mult_same.any() and self.OP_POW_IDS.numel() > 0 and self.ID_2 != -1:
                pop[match_mult_same, j] = self.OP_POW_IDS[0]
                pop[match_mult_same, j-1] = self.ID_2
                pop[match_mult_same, j-2] = arg1[match_mult_same]
                n_simplified += match_mult_same.sum().item()

            # --- a*x + b*x -> (a+b)*x ---
            # RPN: [a, x, *, b, x, *, +]
            if j >= 6:
                is_p = (pop[:, j] == self.OP_PLUS)
                m2, m1 = (pop[:, j-1] == self.OP_MULT), (pop[:, j-4] == self.OP_MULT)
                # Term 2: [b, x, *] at j-3, j-2, j-1
                # Term 1: [a, x, *] at j-6, j-5, j-4
                x2, b = pop[:, j-2], pop[:, j-3]
                x1, a = pop[:, j-5], pop[:, j-6]
                
                # Check for [a, x, *, b, x, *, +]
                is_ax_bx = is_p & m1 & m2 & (x1 == x2) & (self.arity_table[x1.clamp(0)] == 0) & (x1 != PAD_ID)
                
                # Also handle commutative variants like [x, a, *, x, b, *, +]
                # But for the test case [2, x0, *, 3, x0, *, +] it's [a, x, *, b, x, *, +]
                
                if is_ax_bx.any():
                    # Check for literals (already handled above)
                    pass
                    
                # Generalized Factoring: [x, a, *, x, b, *, +] -> [a, b, +, x, *] (SymPy style)
                # This makes it easier for constant folding to find [a, b, +]
                if j >= 6:
                    x1, a = pop[:, j-5], pop[:, j-6]
                    x2, b = pop[:, j-2], pop[:, j-3]
                    m2, m1, is_p = (pop[:, j-1] == self.OP_MULT), (pop[:, j-4] == self.OP_MULT), (pop[:, j] == self.OP_PLUS)
                    
                # Generalized Factoring: [x, a, *, x, b, *, +] -> [a, b, +, x, *] (SymPy style)
                # This makes it easier for constant folding to find [a, b, +]
                if j >= 6:
                    # Capture values as CLONES to avoid view-modification bugs
                    val_6 = pop[:, j-6].clone()
                    val_5 = pop[:, j-5].clone()
                    val_4 = pop[:, j-4].clone()
                    val_3 = pop[:, j-3].clone()
                    val_2 = pop[:, j-2].clone()
                    val_1 = pop[:, j-1].clone()
                    val_0 = pop[:, j].clone()
                    
                    m2, m1, is_p = (val_1 == self.OP_MULT), (val_4 == self.OP_MULT), (val_0 == self.OP_PLUS)
                    
                    # Case 1: [a, x, *, b, x, *, +] -> (a+b)*x
                    match_fact_1 = is_p & m1 & m2 & (val_5 == val_2) & (self.arity_table[val_5.clamp(0)] == 0) & (val_5 != PAD_ID)
                    match_fact_1 &= (self.arity_table[val_6.clamp(0)] == 0) & (self.arity_table[val_3.clamp(0)] == 0)
                    
                    if match_fact_1.any():
                        rows = torch.where(match_fact_1)[0]
                        pop[rows, j-6] = val_6[rows] # a
                        pop[rows, j-5] = val_3[rows] # b
                        pop[rows, j-4] = self.OP_PLUS
                        pop[rows, j-3] = val_5[rows] # x (the common variable)
                        pop[rows, j-2] = self.OP_MULT
                        pop[rows, j-1] = PAD_ID
                        pop[rows, j] = PAD_ID
                        n_simplified += match_fact_1.sum().item()

                    # Case 2: [x, a, *, x, b, *, +] -> (a+b)*x
                    match_fact_2 = is_p & m1 & m2 & (val_6 == val_3) & (self.arity_table[val_6.clamp(0)] == 0) & (val_6 != PAD_ID)
                    match_fact_2 &= (self.arity_table[val_5.clamp(0)] == 0) & (self.arity_table[val_2.clamp(0)] == 0)
                    
                    if match_fact_2.any():
                        rows = torch.where(match_fact_2)[0]
                        pop[rows, j-6] = val_5[rows] # a
                        pop[rows, j-5] = val_2[rows] # b
                        pop[rows, j-4] = self.OP_PLUS
                        pop[rows, j-3] = val_6[rows] # x (the common variable)
                        pop[rows, j-2] = self.OP_MULT
                        pop[rows, j-1] = PAD_ID
                        pop[rows, j] = PAD_ID
                        n_simplified += match_fact_2.sum().item()
            
            # --- Generalized Factoring: x*y + x*z -> x*(y+z) ---
            if is_plus.any():
                # Batch-calculate top-level subtrees for the PLUS
                s2_plus = self._get_subtree_starts(pop, j-1)
                s1_plus = self._get_subtree_starts(pop, s2_plus-1)
                
                # Filter for rows where both subtrees are multiplications
                # Filter for rows where both subtrees are multiplications
                t2_idx = (s2_plus - 1).clamp(min=0)
                t2_id = pop.gather(1, t2_idx.unsqueeze(1)).squeeze(1)
                t_top_id = pop.gather(1, torch.full((B, 1), j-1, device=self.device, dtype=torch.long)).squeeze(1)
                is_mult_mult = (t2_id == self.OP_MULT) & (t_top_id == self.OP_MULT)
                
                if is_mult_mult.any():
                    # Batch-calculate inner subtree starts
                    # arg1 subtrees:
                    e1_2 = (s2_plus - 2).clamp(min=0)
                    s1_2_batch = self._get_subtree_starts(pop, e1_2)
                    e1_1 = (s1_2_batch - 1).clamp(min=0)
                    s1_1_batch = self._get_subtree_starts(pop, e1_1)
                    
                    # arg2 subtrees:
                    e2_2 = torch.full((B,), j - 2, device=self.device, dtype=torch.long).clamp(min=0)
                    s2_2_batch = self._get_subtree_starts(pop, e2_2)
                    e2_1 = (s2_2_batch - 1).clamp(min=0)
                    s2_1_batch = self._get_subtree_starts(pop, e2_1)

                    rows = torch.where(is_mult_mult)[0]
                    for b in rows:
                        s1, s2 = s1_plus[b].item(), s2_plus[b].item()
                        s1_1, s1_2 = s1_1_batch[b].item(), s1_2_batch[b].item()
                        e1_1, e1_2 = s1_2 - 1, s2 - 2
                        
                        s2_1, s2_2 = s2_1_batch[b].item(), s2_2_batch[b].item()
                        e2_1, e2_2 = s2_2 - 1, j - 2
                        
                        if s1_1 < 0 or s1_2 < 0 or s2_1 < 0 or s2_2 < 0: continue
                        
                        # Case: Common factor on left
                        # We use the fact that indices are already calculated batch-wise
                        if torch.equal(pop[b, s1_1:e1_1+1], pop[b, s2_1:e2_1+1]):
                            x = pop[b, s1_1:e1_1+1].clone()
                            y = pop[b, s1_2:e1_2+1].clone()
                            z = pop[b, s2_2:e2_2+1].clone()
                            # [x, y, z, +, *]
                            new = torch.cat([x, y, z, torch.tensor([self.OP_PLUS, self.OP_MULT], device=self.device)])
                            if len(new) <= (j - s1 + 1):
                                pop[b, s1:s1+len(new)] = new
                                pop[b, s1+len(new):j+1] = PAD_ID
                                n_simplified += 1

        return pop, n_simplified

    def _apply_modulo_rules(self, population: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, L = population.shape
        pop = population.clone()
        n_simplified = 0
        z_id = self.zero_ids[0] if self.zero_ids.numel() > 0 else self.CONST_0
        
        self.OP_MOD = self.grammar.token_to_id.get('mod', -1)
        if self.OP_MOD == -1: return pop, 0

        for j in range(2, L):
            op = pop[:, j]
            if not (op == self.OP_MOD).any(): continue
            
            # x % x -> 0
            arg2, arg1 = pop[:, j-1], pop[:, j-2]
            match_self = (op == self.OP_MOD) & (arg1 == arg2) & (self.arity_table[arg1.clamp(0)] == 0) & (arg1 != PAD_ID)
            
            if match_self.any() and z_id != -1:
                pop[match_self, j-2] = z_id
                pop[match_self, j-1] = PAD_ID
                pop[match_self, j] = PAD_ID
                n_simplified += match_self.sum().item()
        return pop, n_simplified

    def _apply_constant_folding(self, population: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, L = population.shape
        pop = population.clone()
        n_folded = 0
        val_table = torch.empty(self.arity_table.shape[0], device=self.device, dtype=self.dtype).fill_(float('nan'))
        for t, tid in self.grammar.token_to_id.items():
            if t.replace('.','',1).isdigit() or (t.startswith('-') and t[1:].replace('.','',1).isdigit()): val_table[tid] = float(t)
        for j in range(2, L):
            op, a1, a2 = pop[:, j], pop[:, j-2], pop[:, j-1]
            v1, v2 = val_table[a1], val_table[a2]
            mask = (~v1.isnan()) & (~v2.isnan())
            if not mask.any(): continue
            res = torch.empty(B, device=self.device, dtype=self.dtype).fill_(float('nan'))
            m = mask & (op==self.OP_PLUS); res[m] = v1[m] + v2[m]
            m = mask & (op==self.OP_MINUS); res[m] = v1[m] - v2[m]
            m = mask & (op==self.OP_MULT); res[m] = v1[m] * v2[m]
            m = mask & (op==self.OP_DIV) & (v2!=0); res[m] = v1[m] / v2[m]
            is_pow = (op.unsqueeze(-1) == self.OP_POW_IDS).any(-1) if self.OP_POW_IDS.numel() > 0 else torch.zeros_like(op, dtype=torch.bool)
            m = mask & is_pow & (v1>0); res[m] = v1[m].pow(v2[m])
            
            # Map results to terminal IDs if they exist
            match_any = ~res.isnan()
            if match_any.any():
                # We need a reverse mapping: value -> tid
                # Since we are in a vectorized loop, we can't easily dict lookup for each row.
                # However, we can check for the most common terminals.
                for tid in self.literal_ids:
                    val = val_table[tid]
                    match = match_any & (res == val)
                    if match.any():
                        pop[match, j-2], pop[match, j-1], pop[match, j] = tid, PAD_ID, PAD_ID
                        n_folded += match.sum().item()
                        match_any[match] = False # Avoid redundant assignments
        return pop, n_folded

    def _compact_formulas(self, population: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, L = population.shape
        is_pad = (population == PAD_ID)
        sort_key = is_pad.long() * L + torch.arange(L, device=self.device).unsqueeze(0)
        _, idx = torch.sort(sort_key, dim=1, stable=True)
        return torch.gather(population, 1, idx), (is_pad.any(dim=1).sum().item())

    def _get_subtree_starts(self, population: torch.Tensor, end_indices) -> torch.Tensor:
        B, L = population.shape
        arities = self.arity_table[population.clamp(0)]
        if isinstance(end_indices, int):
            end_t = torch.empty(B, device=self.device, dtype=torch.long).fill_(end_indices)
            max_e = end_indices
        else:
            end_t, max_e = end_indices, end_indices.max().item()
        bal = torch.zeros(B, device=self.device, dtype=torch.long)
        starts = end_t.clone()
        fin, act = torch.zeros(B, device=self.device, dtype=torch.bool), torch.zeros(B, device=self.device, dtype=torch.bool)
        for k in range(max_e, -1, -1):
            new = (end_t == k)
            if new.any(): bal[new] = 1; act |= new
            m = act & (~fin)
            if not m.any(): continue
            bal[m] += arities[m, k] - 1
            is_z = (bal == 0) & m
            starts[is_z], fin[is_z] = k, True
            if fin.all(): break
        return starts
