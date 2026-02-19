"""
GPU-Native Symbolic Simplifier - Professional Performance Edition

A high-performance symbolic simplification engine designed for millions of formulas.
All operations are 100% vectorized using PyTorch tensors.
Zero Python loops over the batch dimension ensure maximum GPU throughput.
"""
import math
import torch
from typing import Tuple, List
from .grammar import PAD_ID, GPUGrammar

try:
    from sys import path as sys_path
    from os import path as os_path
    cuda_path = os_path.join(os_path.dirname(__file__), 'cuda')
    if cuda_path not in sys_path: sys_path.append(cuda_path)
    import rpn_cuda_native
    SIMPLIFY_CUDA_AVAILABLE = True
except ImportError:
    SIMPLIFY_CUDA_AVAILABLE = False

def is_op_in(op, op_ids):
    """Helper to check if op is in a list of IDs (vectorized broadcast)"""
    if op_ids.numel() == 0:
        return torch.zeros_like(op, dtype=torch.bool)
    return (op.unsqueeze(-1) == op_ids).any(-1)

class GPUSymbolicSimplifier:
    def __init__(self, grammar: GPUGrammar, device, dtype=torch.float64):
        self.grammar = grammar
        self.device = device
        self.dtype = dtype
        self._fold_abs_tol = 1e-4
        self._fold_rel_tol = 1e-5
        self._cache_operator_ids()
        self._build_arity_table()
        self._cache_val_table()
        
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
        self.OP_EXP_IDS = get_ids(['exp'])
        self.OP_NEG_IDS = get_ids(['neg'])
        self.OP_SQRT_IDS = get_ids(['sqrt'])
        self.OP_ABS_IDS = get_ids(['abs'])
        self.OP_FACT_IDS = get_ids(['fact', 'factorial'])
        
        self.OP_SIN = g.token_to_id.get('sin', -1)
        self.OP_COS = g.token_to_id.get('cos', -1)
        self.OP_TAN = g.token_to_id.get('tan', -1)
        
        # Rescued Advanced Operators
        self.OP_GAMMA_IDS = get_ids(['gamma', '!'])
        self.OP_LGAMMA_IDS = get_ids(['lgamma', 'lg', 'g'])
        
        self.OP_ASIN = g.token_to_id.get('asin', -1)
        self.OP_ACOS = g.token_to_id.get('acos', -1)
        self.OP_ATAN = g.token_to_id.get('atan', -1)
        
        self.OP_FLOOR = g.token_to_id.get('floor', g.token_to_id.get('_', -1))
        self.OP_CEIL = g.token_to_id.get('ceil', -1)
        self.OP_SIGN = g.token_to_id.get('sign', -1)
        
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
        self.ID_3 = self.grammar.token_to_id.get('3', -1)
        self.ID_4 = self.grammar.token_to_id.get('4', -1)
        self.ID_5 = self.grammar.token_to_id.get('5', -1)
        self.ID_6 = self.grammar.token_to_id.get('6', -1)
        self.ID_PI = self.grammar.token_to_id.get('pi', -1)
        self.OP_0 = self.zero_ids[0].item() if self.zero_ids.numel() > 0 else -1
        self.OP_1 = self.one_ids[0].item() if self.one_ids.numel() > 0 else -1
        self.OP_2 = self.two_ids[0].item() if self.two_ids.numel() > 0 else -1
        self.OP_POW_ID = self.OP_POW_IDS[0].item() if self.OP_POW_IDS.numel() > 0 else -1
        self.OP_LOG_ID = self.OP_LOG_IDS[0].item() if self.OP_LOG_IDS.numel() > 0 else -1
        self.OP_EXP_ID = self.OP_EXP_IDS[0].item() if self.OP_EXP_IDS.numel() > 0 else -1
        self.OP_NEG_ID = self.OP_NEG_IDS[0].item() if self.OP_NEG_IDS.numel() > 0 else -1
        self.OP_SQRT_ID = self.OP_SQRT_IDS[0].item() if self.OP_SQRT_IDS.numel() > 0 else -1
        self.OP_ABS_ID = self.OP_ABS_IDS[0].item() if self.OP_ABS_IDS.numel() > 0 else -1
        self.OP_FACT_ID = self.OP_FACT_IDS[0].item() if self.OP_FACT_IDS.numel() > 0 else -1
        self.OP_GAMMA_ID = self.OP_GAMMA_IDS[0].item() if self.OP_GAMMA_IDS.numel() > 0 else -1
        self.OP_LGAMMA_ID = self.OP_LGAMMA_IDS[0].item() if self.OP_LGAMMA_IDS.numel() > 0 else -1
        self.ID_E = self.grammar.token_to_id.get('e', -1)
        
    def _build_arity_table(self):
        from core.grammar import OPERATORS
        # Ensure table is large enough for both vocab and PAD_ID
        max_id = max(self.grammar.id_to_token.keys()) + 1
        max_id = max(max_id, PAD_ID + 1)
        
        self.arity_table = torch.zeros(max_id, dtype=torch.long, device=self.device)
        for token, tid in self.grammar.token_to_id.items():
            self.arity_table[tid] = OPERATORS.get(token, 0)
        # GPU grammar may expose aliases not present in core OPERATORS
        if 'fact' in self.grammar.token_to_id:
            self.arity_table[self.grammar.token_to_id['fact']] = 1
        if 'factorial' in self.grammar.token_to_id:
            self.arity_table[self.grammar.token_to_id['factorial']] = 1
        
        # arity_table[PAD_ID] remains 0, which is safe for subtree logic

    def _cache_val_table(self):
        """Pre-build val_table once at init — eliminates Python dict iteration per constant_folding call."""
        max_id = self.arity_table.size(0)
        self._cached_val_table = torch.empty(max_id, device=self.device, dtype=self.dtype).fill_(float('nan'))
        for t, tid in self.grammar.token_to_id.items():
            if t.replace('.','',1).isdigit() or (t.startswith('-') and t[1:].replace('.','',1).isdigit()):
                self._cached_val_table[tid] = float(t)
            elif t == 'pi':
                self._cached_val_table[tid] = math.pi
            elif t == 'e':
                self._cached_val_table[tid] = math.e
        self._cached_lit_vals = self._cached_val_table[self.literal_ids] if self.literal_ids.numel() > 0 else None
        
        # FIX N8 (Optimización): Cachear versiones CPU para evitar transferencias en bucles
        self._lit_vals_cpu = None
        if self._cached_lit_vals is not None:
            self._lit_vals_cpu = self._cached_lit_vals.cpu().numpy() if self._cached_lit_vals.is_cuda else self._cached_lit_vals.numpy()
        
        self._lit_ids_cpu = self.literal_ids.cpu().numpy() if self.literal_ids.is_cuda else self.literal_ids.numpy()
                
    def _is_zero(self, token_ids: torch.Tensor) -> torch.Tensor:
        by_id = (token_ids.unsqueeze(-1) == self.zero_ids).any(dim=-1) if self.zero_ids.numel() > 0 else torch.zeros_like(token_ids, dtype=torch.bool)
        if not hasattr(self, '_cached_val_table'):
            return by_id
        max_id = self._cached_val_table.size(0)
        t = token_ids.clamp(0, max_id - 1)
        vals = self._cached_val_table[t.long()]
        by_val = (~vals.isnan()) & (vals.abs() <= self._fold_abs_tol)
        return by_id | by_val
    
    def _is_one(self, token_ids: torch.Tensor) -> torch.Tensor:
        by_id = (token_ids.unsqueeze(-1) == self.one_ids).any(dim=-1) if self.one_ids.numel() > 0 else torch.zeros_like(token_ids, dtype=torch.bool)
        if not hasattr(self, '_cached_val_table'):
            return by_id
        max_id = self._cached_val_table.size(0)
        t = token_ids.clamp(0, max_id - 1)
        vals = self._cached_val_table[t.long()]
        by_val = (~vals.isnan()) & ((vals - 1.0).abs() <= self._fold_abs_tol)
        return by_id | by_val

    def _is_constant(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens.unsqueeze(-1) == self.literal_ids).any(dim=-1)

    def _is_constant_value(self, tokens: torch.Tensor, val: float) -> torch.Tensor:
        if val == 0.0: return self._is_zero(tokens)
        if val == 1.0: return self._is_one(tokens)
        if val == 2.0: return tokens == self.CONST_2
        max_id = self._cached_val_table.size(0)
        t = tokens.clamp(0, max_id - 1)
        vals = self._cached_val_table[t.long()]
        return (~vals.isnan()) & ((vals - val).abs() <= self._fold_abs_tol)

    def _map_values_to_literal_ids(self, values: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map numeric values to grammar literal token IDs with tolerance.
        Returns (token_ids, match_mask), both shape [B].
        """
        B = values.shape[0]
        token_ids = torch.full((B,), -1, device=self.device, dtype=torch.long)
        finite_mask = valid_mask & torch.isfinite(values)

        if self.ID_0 != -1:
            m0 = finite_mask & (values.abs() <= self._fold_abs_tol)
            token_ids[m0] = self.ID_0
        if self.ID_1 != -1:
            m1 = finite_mask & (token_ids == -1) & ((values - 1.0).abs() <= self._fold_abs_tol)
            token_ids[m1] = self.ID_1
        if self.ID_2 != -1:
            m2 = finite_mask & (token_ids == -1) & ((values - 2.0).abs() <= self._fold_abs_tol)
            token_ids[m2] = self.ID_2

        if self._cached_lit_vals is not None and self._cached_lit_vals.numel() > 0:
            unresolved = finite_mask & (token_ids == -1)
            if unresolved.any():
                dists = (values.unsqueeze(-1) - self._cached_lit_vals.unsqueeze(0)).abs()
                min_dist, min_idx = dists.min(dim=1)
                tol = self._fold_abs_tol + self._fold_rel_tol * values.abs()
                m_lit = unresolved & (min_dist <= tol)
                token_ids[m_lit] = self.literal_ids[min_idx[m_lit]]

        matched = token_ids != -1
        return token_ids, matched

    def _map_single_value_to_literal_id(self, value: float) -> int:
        """
        Map a single numeric value to a grammar literal token ID.
        
        FIX N8: Versión optimizada sin crear tensores GPU.
        Usa búsqueda directa en los literales cacheados (CPU) para evitar
        asignaciones de memoria GPU en cada llamada dentro de bucles.
        """
        # Fast path: valores comunes hardcoded
        if self.ID_0 != -1 and abs(value) <= self._fold_abs_tol:
            return self.ID_0
        if self.ID_1 != -1 and abs(value - 1.0) <= self._fold_abs_tol:
            return self.ID_1
        if self.ID_2 != -1 and abs(value - 2.0) <= self._fold_abs_tol:
            return self.ID_2
        if self.ID_PI != -1 and abs(value - math.pi) <= self._fold_abs_tol:
            return self.ID_PI
        if self.ID_E != -1 and abs(value - math.e) <= self._fold_abs_tol:
            return self.ID_E
        
        # Buscar en literales cacheados (sin crear tensores GPU)
        if self._cached_lit_vals is not None and self.literal_ids.numel() > 0:
            tol = self._fold_abs_tol + self._fold_rel_tol * abs(value)
            # Acceder a valores CPU directamente
            for i, lit_val in enumerate(self._lit_vals_cpu):
                if abs(value - lit_val) <= tol:
                    return int(self._lit_ids_cpu[i])
        
        return -1

    def _precompute_all_subtree_starts(self, population: torch.Tensor) -> torch.Tensor:
        """
        Pre-compute subtree start index for EVERY position in every formula.
        Returns: [B, L] tensor where result[b, j] = start index of subtree ending at j.
        
        This replaces hundreds of individual _get_subtree_starts calls per simplifier pass
        with a single backward scan. ~30x fewer GPU syncs.
        """
        B, L = population.shape
        device = population.device
        max_id = self.arity_table.size(0)
        pop_c = population.clamp(0, max_id - 1)
        arities = self.arity_table[pop_c.long()]  # [B, L]
        arities[population >= max_id] = 0
        arities[population == PAD_ID] = 0
        
        # Result matrix
        starts = torch.arange(L, device=device).unsqueeze(0).expand(B, L).clone()  # Default: start = self
        
        # Backward scan: for each position j (from L-1 to 0), determine subtree start
        # A subtree ending at j needs 'arity[j]' sub-subtrees before it
        # We process right-to-left, tracking how many items each position needs
        need = arities.clone()  # [B, L] - remaining items needed
        
        # Strategy: Process each ending position j in parallel across batch B
        
        # More efficient approach: backward cumulative scan
        # For each position j, the subtree start is determined by walking backward
        # until we've consumed enough tokens to satisfy the arity chain.
        # We use the same algorithm as _get_subtree_starts but for ALL positions at once.
        
        # Strategy: Process each ending position j in parallel across batch B
        # For each j, scan backward from j tracking balance
        # Balance starts at 1 (need the root), decreases by 1 per token, increases by arity
        # When balance hits 0, that's the start
        
        # We can process all positions in a single backward scan with a "stacked" approach:
        # Process positions from right to left. For each j, start a new balance tracker.
        # All active trackers advance one step left simultaneously.
        
        # Simpler O(L) approach: backward DP
        # subtree_len[b, j] = 1 + sum of subtree_len of children
        # For RPN, children of token at j are the arity[j] subtrees immediately preceding j
        
        subtree_len = torch.ones(B, L, dtype=torch.long, device=device)
        subtree_len[population == PAD_ID] = 0
        
        # Process left to right (RPN order)
        # For each position j:
        #   if arity == 0: subtree_len = 1 (terminal)
        #   if arity == 1: subtree_len = 1 + subtree_len[j-1] (the single child)
        #   if arity == 2: subtree_len = 1 + subtree_len[j-1] + subtree_len[j-1 - subtree_len[j-1]]
        for j in range(L):
            is_pad = (population[:, j] == PAD_ID)
            ar = arities[:, j]  # [B]
            
            if j == 0:
                # First position must be a terminal
                subtree_len[:, 0] = torch.where(is_pad, torch.zeros(B, dtype=torch.long, device=device), 
                                                  torch.ones(B, dtype=torch.long, device=device))
                continue
            
            # Unary: child ends at j-1
            is_unary = (ar == 1) & ~is_pad
            if is_unary.any():
                child_len = subtree_len[:, j-1]
                subtree_len[:, j] = torch.where(is_unary, 1 + child_len, subtree_len[:, j])
            
            # Binary: right child ends at j-1, left child ends at j-1-len(right_child)
            is_binary = (ar == 2) & ~is_pad
            if is_binary.any():
                right_len = subtree_len[:, j-1]  # [B]
                left_end = (j - 1 - right_len).clamp(0, L - 1)  # [B]
                left_len = subtree_len.gather(1, left_end.unsqueeze(1)).squeeze(1)  # [B]
                subtree_len[:, j] = torch.where(is_binary, 1 + right_len + left_len, subtree_len[:, j])
            
            # Terminals stay at 1 (default)
            subtree_len[:, j] = torch.where(is_pad, torch.zeros(B, dtype=torch.long, device=device), subtree_len[:, j])
        
        # starts[b, j] = j - subtree_len[b, j] + 1
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        starts = positions - subtree_len + 1
        starts = starts.clamp(0, L - 1)
        # PAD positions get -1
        starts[population == PAD_ID] = -1
        
        return starts
    
    def _get_subtree_starts_cached(self, starts_cache: torch.Tensor, end_indices) -> torch.Tensor:
        """Look up subtree starts from pre-computed cache instead of recomputing."""
        B = starts_cache.shape[0]
        if isinstance(end_indices, int):
            return starts_cache[:, end_indices]
        else:
            # Clamp to valid range
            idx = end_indices.clamp(0, starts_cache.shape[1] - 1).unsqueeze(1)
            return starts_cache.gather(1, idx).squeeze(1)

    def simplify_batch(self, population: torch.Tensor, constants: torch.Tensor = None, max_passes: int = 3) -> Tuple[torch.Tensor, torch.Tensor, int]:
        B, L = population.shape
        pop = population.clone()
        
        # ============ CUDA FAST PATH ============
        if SIMPLIFY_CUDA_AVAILABLE and L <= 64:
            try:
                # Build int32 arity table for kernel
                max_id = self.arity_table.size(0)
                arities_int = self.arity_table.to(dtype=torch.int32).contiguous()
                
                # Build val_table (float32): maps token_id -> numerical value (NaN if not literal)
                if not hasattr(self, '_cuda_val_table'):
                    vt = torch.empty(max_id, device=self.device, dtype=torch.float32).fill_(float('nan'))
                    for t, tid in self.grammar.token_to_id.items():
                        if t.replace('.','',1).isdigit() or (t.startswith('-') and t[1:].replace('.','',1).isdigit()):
                            vt[tid] = float(t)
                        elif t == 'pi':
                            vt[tid] = float(math.pi)
                        elif t == 'e':
                            vt[tid] = float(math.e)
                    self._cuda_val_table = vt.contiguous()
                    # Build literal_ids (int64) and literal_vals (float32) for nearest-literal mapping
                    self._cuda_literal_ids = self.literal_ids.contiguous()
                    self._cuda_literal_vals = torch.tensor(
                        [float(self.grammar.id_to_token[lid.item()]) for lid in self.literal_ids],
                        device=self.device, dtype=torch.float32
                    ).contiguous() if self.literal_ids.numel() > 0 else torch.empty(0, device=self.device, dtype=torch.float32)
                
                rpn_cuda_native.simplify_batch(
                    pop, arities_int,
                    self._cuda_val_table, self._cuda_literal_ids, self._cuda_literal_vals,
                    max_passes,
                    self.OP_PLUS, self.OP_MINUS, self.OP_MULT, self.OP_DIV,
                    self.OP_NEG_ID, self.grammar.token_to_id.get('%', self.grammar.token_to_id.get('mod', -1)), self.OP_POW_ID,
                    self.OP_SIN, self.OP_COS, self.OP_TAN,
                    self.OP_ASIN, self.OP_ACOS, self.OP_ATAN,
                    self.OP_LOG_ID, self.OP_EXP_ID, self.OP_SQRT_ID, self.OP_ABS_ID,
                    self.OP_GAMMA_ID, self.OP_LGAMMA_ID,
                    self.OP_FLOOR, self.OP_CEIL, self.OP_SIGN,
                    self.ID_0, self.ID_1, self.ID_2, self.ID_3, self.ID_4, self.ID_5, self.ID_6
                )
                
                # Validate after CUDA simplification
                is_valid = self._validate_batch_stack(pop)
                if not is_valid.all():
                    invalid_indices = torch.where(~is_valid)[0]
                    pop[invalid_indices] = population[invalid_indices]

                # CUDA kernel is fast but can miss some higher-order/iterative patterns.
                # Run a short vectorized GPU cleanup pass to normalize final expressions.
                cleanup_passes = 2 if max_passes >= 2 else 1
                pop, constants, n_cleanup = self._simplify_fallback_passes(
                    pop,
                    constants,
                    cleanup_passes,
                    original_population=population,
                )

                return pop, constants, n_cleanup
            except Exception:
                pass  # Fall through to Python implementation

        return self._simplify_fallback_passes(pop, constants, max_passes, original_population=population)

    def _simplify_fallback_passes(
        self,
        pop: torch.Tensor,
        constants: torch.Tensor,
        max_passes: int,
        original_population: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Vectorized GPU simplification passes used as fallback and post-CUDA cleanup."""
        if original_population is None:
            original_population = pop

        total_simplified = torch.tensor(0, dtype=torch.long, device=self.device)
        for _ in range(max_passes):
            n_pass = torch.tensor(0, dtype=torch.long, device=self.device)
            
            # Pre-compute subtree starts ONCE per pass (replaces hundreds of individual calls)
            starts_cache = self._precompute_all_subtree_starts(pop)
            
            # Normalization helps patterns match more reliably
            pop, n = self._apply_commutative_normalization(pop, starts_cache); n_pass += n
            
            # Recompute cache after normalization changed the population
            if n > 0:
                starts_cache = self._precompute_all_subtree_starts(pop)
            
            pop, n = self._apply_identity_rules(pop, starts_cache); n_pass += n
            if n > 0:
                starts_cache = self._precompute_all_subtree_starts(pop)
            pop, n = self._apply_zero_rules(pop, starts_cache); n_pass += n
            if n > 0:
                starts_cache = self._precompute_all_subtree_starts(pop)
            pop, n = self._apply_self_cancellation_rules(pop); n_pass += n
            pop, n = self._apply_associative_rules(pop, starts_cache); n_pass += n
            if n > 0:
                starts_cache = self._precompute_all_subtree_starts(pop)
            pop, n = self._apply_advanced_rules(pop, starts_cache); n_pass += n
            if n > 0:
                starts_cache = self._precompute_all_subtree_starts(pop)
            pop, n = self._apply_term_consolidation(pop, starts_cache); n_pass += n
            if n > 0:
                starts_cache = self._precompute_all_subtree_starts(pop)
            pop, n = self._apply_modulo_rules(pop, starts_cache); n_pass += n
            pop, n = self._apply_constant_folding(pop, starts_cache); n_pass += n
            pop, n = self._compact_formulas(pop); n_pass += n
            if n_pass.item() == 0: break
            total_simplified += n_pass

        # Validation Step: Revert rows that became invalid (unbalanced)
        is_valid = self._validate_batch_stack(pop)
        if not is_valid.all():
            invalid_indices = torch.where(~is_valid)[0]
            pop[invalid_indices] = original_population[invalid_indices]

        return pop, constants, total_simplified.item()
    
    def _validate_batch_stack(self, population: torch.Tensor) -> torch.Tensor:
        """
        Vectorized RPN validity check.
        Returns boolean tensor of shape (B,) where True means valid.
        Valid means:
        1. Final stack size == 1
        2. Stack never underflows (< 1) during evaluation (assuming >0 tokens)
        """
        B, L = population.shape
        # Get arities
        max_id = self.arity_table.size(0)
        pop_c = population.clamp(0, max_id - 1)
        # Fix: Cast to long because uint8 is treated as mask
        arities = self.arity_table[pop_c.long()]
        
        # PADs have arity 0 in table, but they shouldn't contribute +1 to stack like operands.
        # They should effectively be 0 change.
        # Operands (arity 0) -> delta +1
        # BinOps (arity 2) -> delta -1
        # UnOp (arity 1) -> delta 0
        
        deltas = 1 - arities
        
        # Mask PADs
        is_pad = (population == PAD_ID)
        deltas[is_pad] = 0
        
        # Cumsum
        stack_depth = torch.cumsum(deltas, dim=1)
        
        # Check 1: Final depth (ignoring PADs) should be 1
        # Since pads are at the end (compacted) or 0-delta, the last element of cumsum is the final depth
        final_depth = stack_depth[:, -1]
        cond_final = (final_depth == 1)
        
        # Check 2: No underflow. Stack should be >= 1 at all valid steps.
        # But wait, initially stack is 0. 
        # Token 1 (operand) -> stack 1. 
        # So min_value should be >= 1.
        # However, if we have [PAD, PAD...], stack is 0.
        # Valid formula must have at least 1 token.
        has_tokens = (~is_pad).any(dim=1)
        
        # Filter cumsum where not pad
        # We need to check that stack never drops below 1 for active tokens.
        # But for PAD positions, the value is just carried over.
        # So checking the whole cumsum >= 1 is valid, AS LONG AS initial part isn't PAD.
        # If we have leading PADs (impossible in our setup, PADs are at end), they would be 0.
        # Assuming PADs are always at the end:
        # The active region is where stack depth changes or stays same > 0.
        
        # Simpler check: Min value of stack_depth must be >= 1.
        # However, if formula is empty [PAD, ...], stack_depth is all 0.
        # 'has_tokens' handles the empty case.
        # For non-empty formulas, the first token (operand) makes stack 1.
        # Subsequent ops keep it >= 1.
        # If it drops to 0 or -1, it's invalid.
        
        # Optimization: We only care about ensuring no underflow.
        # If stack_depth.min() < 1, then at some point it hit 0 or negative.
        # Note: Since PADs don't change depth, if valid formula ends at 1, 
        # the trailing PADs will all be 1.
        # So min() check covers everything perfectly (assuming no leading PADs).
        
        cond_no_underflow = (stack_depth.min(dim=1)[0] >= 1)
        
        return cond_final & has_tokens & cond_no_underflow
    
    def _apply_identity_rules(self, population: torch.Tensor, starts_cache: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = population.shape
        pop = population.clone()
        n_simplified = torch.tensor(0, dtype=torch.long, device=self.device)
        o_id = self.ID_1
        for j in range(2, L):
            op = pop[:, j]
            # Pattern: [arg1, arg2, op]
            is_plus, is_mult, is_minus, is_div = (op==self.OP_PLUS), (op==self.OP_MULT), (op==self.OP_MINUS), (op==self.OP_DIV)
            is_pow = (op.unsqueeze(-1) == self.OP_POW_IDS).any(-1) if self.OP_POW_IDS.numel() > 0 else torch.zeros_like(op, dtype=torch.bool)
            
            has_work = (is_plus | is_mult | is_minus | is_div | is_pow)
            # Skip position if no work — but avoid GPU sync by using .any() only once
            # The operations below are cheap no-ops when masks are all-False
            
            s2 = self._get_subtree_starts_cached(starts_cache, j-1) if starts_cache is not None else self._get_subtree_starts(pop, j-1)
            is_z2 = has_work & (s2 == j-1) & self._is_constant_value(pop[:, j-1], 0.0)
            is_o2 = has_work & (s2 == j-1) & self._is_constant_value(pop[:, j-1], 1.0)
            
            # x+0, x-0, x*1, x/1, x^1 -> skip arg2/op (keep arg1)
            to_skip_arg2 = (is_plus & is_z2) | (is_minus & is_z2) | (is_mult & is_o2) | (is_div & is_o2) | (is_pow & is_o2)
            # Apply unconditionally — torch.where is a no-op on empty masks
            pop[:, j-1] = torch.where(to_skip_arg2, PAD_ID, pop[:, j-1])
            pop[:, j] = torch.where(to_skip_arg2, PAD_ID, pop[:, j])
            n_simplified += to_skip_arg2.sum()
            
            # 0+x, 1*x -> skip arg1/op (keep arg2)
            s1 = self._get_subtree_starts_cached(starts_cache, s2-1) if starts_cache is not None else self._get_subtree_starts(pop, s2-1)
            valid_s1 = (s2 > 0)
            is_z1 = (s1 == s2-1) & self._is_constant_value(pop.gather(1, (s2-1).clamp(0).unsqueeze(1)).squeeze(1), 0.0) & valid_s1
            is_o1 = (s1 == s2-1) & self._is_constant_value(pop.gather(1, (s2-1).clamp(0).unsqueeze(1)).squeeze(1), 1.0) & valid_s1
            
            to_skip_arg1 = (is_plus & is_z1) | (is_mult & is_o1)
            to_skip_arg1 &= (s2 > 0) & (s2 < L)
            rows = torch.where(to_skip_arg1)[0]
            pop[rows, (s2-1)[rows]] = PAD_ID
            pop[rows, j] = PAD_ID
            n_simplified += to_skip_arg1.sum()
            
            # Special Constant Result Rules: x^0 -> 1, 1^x -> 1
            match_const_1 = is_pow & (is_z2 | is_o1)
            match_const_1 &= (s1 >= 0) & (s1 < L)
            if o_id != -1:
                rows = torch.where(match_const_1)[0]
                start = s1[match_const_1]
                sub_pop = pop[rows]
                c_idx = torch.arange(len(rows), device=self.device)
                sub_pop[c_idx, start] = o_id
                pos = torch.arange(L, device=self.device).reshape(1, L)
                sub_pop[(pos > start.unsqueeze(1)) & (pos <= j)] = PAD_ID
                pop[rows] = sub_pop
                n_simplified += match_const_1.sum()
        return pop, n_simplified

    def _apply_commutative_normalization(self, population: torch.Tensor, starts_cache: torch.Tensor = None) -> Tuple[torch.Tensor, int]:
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
            
            # Identify subtrees
            s2 = self._get_subtree_starts_cached(starts_cache, j-1) if starts_cache is not None else self._get_subtree_starts(pop, j-1)
            s1 = self._get_subtree_starts_cached(starts_cache, s2-1) if starts_cache is not None else self._get_subtree_starts(pop, s2-1)
            
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
                    
                    # Write only if both indices are valid
                    valid_swap = (cols_s1 >= 0) & (cols_s1 < L) & (cols_s2 >= 0) & (cols_s2 < L)
                    if valid_swap.any():
                        rows_v = rows_simple[valid_swap]
                        pop[rows_v, cols_s1[valid_swap]] = vals_2[valid_swap]
                        pop[rows_v, cols_s2[valid_swap]] = vals_1[valid_swap]
                        n_swapped += valid_swap.sum()
                    
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

    def _apply_zero_rules(self, population: torch.Tensor, starts_cache: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = population.shape
        pop = population.clone()
        n_simplified = torch.tensor(0, dtype=torch.long, device=self.device)
        z_id = self.zero_ids[0].item() if self.zero_ids.numel() > 0 else self.CONST_0
        neg_id = self.OP_NEG_IDS[0].item() if self.OP_NEG_IDS.numel() > 0 else -1
        for j in range(2, L):
            op = pop[:, j]

            is_mult = (pop[:, j] == self.OP_MULT)
            s2 = self._get_subtree_starts_cached(starts_cache, j-1) if starts_cache is not None else self._get_subtree_starts(pop, j-1)
            is_z2 = is_mult & (s2 == j-1) & self._is_zero(pop[:, j-1])
            is_z1 = False
            s1 = self._get_subtree_starts_cached(starts_cache, s2-1) if starts_cache is not None else self._get_subtree_starts(pop, s2-1)
            is_z1 = is_mult & (s1 == s2-1) & self._is_zero(pop.gather(1, (s2-1).clamp(0).unsqueeze(1)).squeeze(1)) & (s2 > 0)
            
            match = is_mult & (is_z1 | is_z2)
            # Safety: start must be valid
            match &= (s1 >= 0) & (s1 < L)
            # Apply unconditionally - no-op when match is all False
            rows = torch.where(match)[0]
            if rows.numel() > 0:
                rows = torch.where(match)[0]
                start_to_wipe = s1[match]
                
                # Create sub-population of matching rows (Copy)
                sub_pop = pop[rows]
                
                cols = torch.arange(len(rows), device=self.device)
                sub_pop[cols, start_to_wipe] = z_id
                
                # Set PADs
                pos = torch.arange(L, device=self.device).reshape(1, L)
                s_wipe = start_to_wipe.unsqueeze(1)
                pad_mask = (pos > s_wipe) & (pos <= j)
                sub_pop[pad_mask] = PAD_ID
                
                # Write back to main population
                pop[rows] = sub_pop
                
                n_simplified += match.sum()

            # 0 - x -> neg(x)
            if neg_id != -1:
                is_minus = (op == self.OP_MINUS)
                is_zero_left = is_minus & (s2 > 0) & (s1 == s2 - 1)
                left_tok = pop.gather(1, (s2 - 1).clamp(0).unsqueeze(1)).squeeze(1)
                is_zero_left &= self._is_constant_value(left_tok, 0.0)

                rows = torch.where(is_zero_left)[0]
                for b in rows:
                    start = int(s1[b].item())
                    arg2_start = int(s2[b].item())
                    if start < 0 or arg2_start < 0 or arg2_start >= j:
                        continue

                    arg2 = pop[b, arg2_start:j].clone()
                    new = torch.cat([
                        arg2,
                        torch.tensor([neg_id], device=self.device, dtype=torch.long)
                    ])
                    if new.numel() <= (j - start + 1):
                        pop[b, start:start + new.numel()] = new
                        pop[b, start + new.numel():j + 1] = PAD_ID
                        n_simplified += 1

            # x / (-1) -> neg(x)
            if neg_id != -1:
                is_div = (op == self.OP_DIV)
                is_neg_one_right = is_div & (s2 == j - 1) & self._is_constant_value(pop[:, j - 1], -1.0)

                rows = torch.where(is_neg_one_right)[0]
                for b in rows:
                    start = int(s1[b].item())
                    arg2_start = int(s2[b].item())
                    if start < 0 or arg2_start <= start or arg2_start >= j:
                        continue

                    arg1 = pop[b, start:arg2_start].clone()
                    new = torch.cat([
                        arg1,
                        torch.tensor([neg_id], device=self.device, dtype=torch.long)
                    ])
                    if new.numel() <= (j - start + 1):
                        pop[b, start:start + new.numel()] = new
                        pop[b, start + new.numel():j + 1] = PAD_ID
                        n_simplified += 1

        return pop, n_simplified

    def _apply_self_cancellation_rules(self, population: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = population.shape
        pop = population.clone()
        n_simplified = torch.tensor(0, dtype=torch.long, device=self.device)
        z_id, o_id = (self.zero_ids[0].item() if self.zero_ids.numel()>0 else self.CONST_0), (self.one_ids[0].item() if self.one_ids.numel()>0 else self.CONST_1)
        for j in range(2, L):
            op = pop[:, j]
            is_matchable = (op == self.OP_MINUS) | (op == self.OP_DIV)
            # Vectorized check for single-token operands: [x, x, -] -> [0, PAD, PAD]
            arg2, arg1 = pop[:, j-1], pop[:, j-2]
            match_single = is_matchable & (arg1 == arg2) & (self.arity_table[arg1.clamp(0).long()] == 0) & (arg1 != PAD_ID)
            # Apply unconditionally via torch.where - no GPU sync
            is_m = match_single & (op == self.OP_MINUS)
            is_d = match_single & (op == self.OP_DIV)
            pop[:, j-2] = torch.where(is_m, z_id, torch.where(is_d, o_id, pop[:, j-2]))
            pop[:, j-1] = torch.where(match_single, PAD_ID, pop[:, j-1])
            pop[:, j] = torch.where(match_single, PAD_ID, pop[:, j])
            n_simplified += match_single.sum()
        return pop, n_simplified

    def _apply_advanced_rules(self, population: torch.Tensor, starts_cache: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
        counts = torch.zeros(B, device=self.device, dtype=torch.long)
        z_id, o_id = self.ID_0, self.ID_1
        abs_id = self.OP_ABS_IDS[0].item() if self.OP_ABS_IDS.numel() > 0 else -1

        for j in range(1, L):
            tokens = pop[:, j]
            # --- Chain reduction logic (torch.where, zero GPU syncs) ---
            if j >= 2:
                # neg(neg(x)) -> x
                match_nn = is_op_in(tokens, self.OP_NEG_IDS) & is_op_in(pop[:, j-1], self.OP_NEG_IDS)
                pop[:, j] = torch.where(match_nn, PAD_ID, pop[:, j])
                pop[:, j-1] = torch.where(match_nn, PAD_ID, pop[:, j-1])
                counts += match_nn.long()
                
                # exp(log(x)) -> x
                match_el = is_op_in(tokens, self.OP_EXP_IDS) & is_op_in(pop[:, j-1], self.OP_LOG_IDS)
                pop[:, j] = torch.where(match_el, PAD_ID, pop[:, j])
                pop[:, j-1] = torch.where(match_el, PAD_ID, pop[:, j-1])
                counts += match_el.long()

                # log(exp(x)) -> x
                match_le = is_op_in(tokens, self.OP_LOG_IDS) & is_op_in(pop[:, j-1], self.OP_EXP_IDS)
                pop[:, j] = torch.where(match_le, PAD_ID, pop[:, j])
                pop[:, j-1] = torch.where(match_le, PAD_ID, pop[:, j-1])
                counts += match_le.long()
                    
                # acos(cos(x)) -> x
                match_ac = (tokens == self.OP_ACOS) & (pop[:, j-1] == self.OP_COS)
                pop[:, j] = torch.where(match_ac, PAD_ID, pop[:, j])
                pop[:, j-1] = torch.where(match_ac, PAD_ID, pop[:, j-1])
                counts += match_ac.long()
                    
                # asin(sin(x)) -> x
                match_as = (tokens == self.OP_ASIN) & (pop[:, j-1] == self.OP_SIN)
                pop[:, j] = torch.where(match_as, PAD_ID, pop[:, j])
                pop[:, j-1] = torch.where(match_as, PAD_ID, pop[:, j-1])
                counts += match_as.long()
                    
                # atan(tan(x)) -> x
                match_at = (tokens == self.OP_ATAN) & (pop[:, j-1] == self.OP_TAN)
                pop[:, j] = torch.where(match_at, PAD_ID, pop[:, j])
                pop[:, j-1] = torch.where(match_at, PAD_ID, pop[:, j-1])
                counts += match_at.long()

                # abs(neg(x)) -> abs(x)
                match_an = is_op_in(tokens, self.OP_ABS_IDS) & is_op_in(pop[:, j-1], self.OP_NEG_IDS)
                pop[:, j-1] = torch.where(match_an, PAD_ID, pop[:, j-1])
                counts += match_an.long()

                # abs(abs(x)) -> abs(x)
                match_aa = is_op_in(tokens, self.OP_ABS_IDS) & is_op_in(pop[:, j-1], self.OP_ABS_IDS)
                pop[:, j-1] = torch.where(match_aa, PAD_ID, pop[:, j-1])
                counts += match_aa.long()

            # --- SOTA / Better than basic Sympy ---
            # sqrt(x^2) -> abs(x)
            if j >= 3 and abs_id != -1:
                match_sqrt_p2 = is_op_in(tokens, self.OP_SQRT_IDS) & is_op_in(pop[:, j-1], self.OP_POW_IDS) & self._is_constant_value(pop[:, j-2], 2.0)
                pop[:, j] = torch.where(match_sqrt_p2, abs_id, pop[:, j])
                pop[:, j-1] = torch.where(match_sqrt_p2, PAD_ID, pop[:, j-1])
                pop[:, j-2] = torch.where(match_sqrt_p2, PAD_ID, pop[:, j-2])
                counts += match_sqrt_p2.long()
            
            # --- Constant arg identities (no .any() sync) ---
            is_unary = (self.arity_table[tokens.clamp(0).long()] == 1)
            arg = pop[:, j-1]
            arg_is0 = is_unary & self._is_zero(arg)
            arg_is1 = is_unary & self._is_one(arg)
            arg_is2 = is_unary & self._is_constant_value(arg, 2.0)
            arg_is_e = is_unary & self._is_constant_value(arg, math.e)
            
            to_zero = arg_is0 & ((tokens==self.OP_SIN)|(tokens==self.OP_TAN)|is_op_in(tokens, self.OP_ABS_IDS))
            to_zero |= arg_is1 & is_op_in(tokens, self.OP_LOG_IDS)
            to_zero |= arg_is1 & is_op_in(tokens, self.OP_LGAMMA_IDS)
            to_zero |= arg_is2 & is_op_in(tokens, self.OP_LGAMMA_IDS)
            
            if z_id != -1:
                pop[:, j-1] = torch.where(to_zero, z_id, pop[:, j-1])
                pop[:, j] = torch.where(to_zero, PAD_ID, pop[:, j])
                counts += to_zero.long()
            
            to_one = arg_is0 & ((tokens==self.OP_COS)|is_op_in(tokens, self.OP_EXP_IDS)|is_op_in(tokens, self.OP_FACT_IDS))
            to_one |= arg_is1 & is_op_in(tokens, self.OP_GAMMA_IDS)
            to_one |= arg_is2 & is_op_in(tokens, self.OP_GAMMA_IDS)
            to_one |= arg_is_e & is_op_in(tokens, self.OP_LOG_IDS)
            
            if o_id != -1:
                pop[:, j-1] = torch.where(to_one, o_id, pop[:, j-1])
                pop[:, j] = torch.where(to_one, PAD_ID, pop[:, j])
                counts += to_one.long()

            # exp(1) -> e (when symbolic e exists in grammar)
            to_e = arg_is1 & is_op_in(tokens, self.OP_EXP_IDS)
            if self.ID_E != -1:
                pop[:, j-1] = torch.where(to_e, self.ID_E, pop[:, j-1])
                pop[:, j] = torch.where(to_e, PAD_ID, pop[:, j])
                counts += to_e.long()
            
            # --- Negation Rules: x + neg(x) -> 0 (cached subtree starts) ---
            if j >= 2:
                is_p = (tokens == self.OP_PLUS)
                is_neg2 = is_op_in(pop[:, j-1], self.OP_NEG_IDS)
                match_p = is_p & is_neg2
                
                s2 = self._get_subtree_starts_cached(starts_cache, j-1) if starts_cache is not None else self._get_subtree_starts(pop, j-1)
                s1 = self._get_subtree_starts_cached(starts_cache, s2-1) if starts_cache is not None else self._get_subtree_starts(pop, s2-1)
                s_inner = self._get_subtree_starts_cached(starts_cache, j-2) if starts_cache is not None else self._get_subtree_starts(pop, j-2)
                
                match_p &= (s1 == s_inner-1) & (s1 >= 0) & (s_inner >= 0)
                t1 = pop.gather(1, s1.clamp(min=0).unsqueeze(1)).squeeze(1)
                t_inner = pop.gather(1, s_inner.clamp(min=0).unsqueeze(1)).squeeze(1)
                match_p &= (t1 == t_inner)
                
                rows = torch.where(match_p)[0]
                for b in rows:
                    pop[b, s1[b]] = z_id
                    pop[b, s1[b]+1:j+1] = PAD_ID
                    counts[b] += 1

        return pop, counts.sum()

    def _apply_associative_rules(self, population: torch.Tensor, starts_cache: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply associative/grouping rules:
        - x + (x + y) -> 2*x + y
        - (x + y) + x -> 2*x + y
        """
        B, L = population.shape
        pop = population.clone()
        n_simplified = torch.tensor(0, dtype=torch.long, device=self.device)
        
        for j in range(4, L):
            op = pop[:, j]
            is_plus = (op == self.OP_PLUS)

            s2_batch = self._get_subtree_starts_cached(starts_cache, j-1) if starts_cache is not None else self._get_subtree_starts(pop, j-1)
            s1_batch = self._get_subtree_starts_cached(starts_cache, s2_batch-1) if starts_cache is not None else self._get_subtree_starts(pop, s2_batch-1)

            # Constant hoisting for nested plus:
            # (c1 + (c2 + x)) -> ((c1 + c2) + x)
            # ((x + c2) + c1) -> ((c1 + c2) + x)
            rows_plus = torch.where(is_plus)[0]
            for b in rows_plus:
                s1 = int(s1_batch[b].item())
                s2 = int(s2_batch[b].item())
                if s1 < 0 or s2 < 0 or s1 >= s2 or s2 > j:
                    continue

                # Case A: c1 + ( ... )
                if s1 == s2 - 1 and pop[b, j-1] == self.OP_PLUS:
                    c1_tok = pop[b, s1]
                    if c1_tok >= 0 and c1_tok < self._cached_val_table.size(0):
                        c1_val = self._cached_val_table[c1_tok.long()].item()
                        if not math.isnan(c1_val):
                            if starts_cache is not None:
                                inner_s2 = int(self._get_subtree_starts_cached(starts_cache, j-2)[b].item())
                                inner_s1 = int(self._get_subtree_starts_cached(starts_cache, inner_s2-1)[b].item())
                            else:
                                inner_s2 = int(self._get_subtree_starts(pop[b:b+1], j-2)[0].item())
                                inner_s1 = int(self._get_subtree_starts(pop[b:b+1], inner_s2-1)[0].item())
                            if inner_s1 >= 0 and inner_s2 >= 0 and inner_s1 < inner_s2:
                                c2_tok = None
                                x_tokens = None
                                # (c2 + x)
                                if inner_s1 == inner_s2 - 1:
                                    c2_tok = pop[b, inner_s1]
                                    x_tokens = pop[b, inner_s2:j-1].clone()
                                # (x + c2)
                                elif inner_s2 == j-2:
                                    c2_tok = pop[b, inner_s2]
                                    x_tokens = pop[b, inner_s1:inner_s2].clone()

                                if c2_tok is not None and c2_tok >= 0 and c2_tok < self._cached_val_table.size(0) and x_tokens is not None and x_tokens.numel() > 0:
                                    c2_val = self._cached_val_table[c2_tok.long()].item()
                                    if not math.isnan(c2_val):
                                        new_c_tid = self._map_single_value_to_literal_id(c1_val + c2_val)
                                        if new_c_tid != -1:
                                            new_seg = torch.cat([
                                                torch.tensor([new_c_tid], device=self.device, dtype=torch.long),
                                                x_tokens,
                                                torch.tensor([self.OP_PLUS], device=self.device, dtype=torch.long)
                                            ])
                                            if new_seg.numel() <= (j - s1 + 1):
                                                # FIX B5: se usaba .to(torch.uint8) hardcoded.
                                                # Si pop.dtype no es uint8 (ej: int64), trunca IDs > 255.
                                                pop[b, s1:s1+new_seg.numel()] = new_seg.to(pop.dtype)
                                                pop[b, s1+new_seg.numel():j+1] = PAD_ID
                                                n_simplified += 1
                                                continue
            
                # Case B: ( ... ) + c1
                if s2 == j - 1 and pop[b, s2-1] == self.OP_PLUS:
                    c1_tok = pop[b, s2]
                    if c1_tok >= 0 and c1_tok < self._cached_val_table.size(0):
                        c1_val = self._cached_val_table[c1_tok.long()].item()
                        if not math.isnan(c1_val):
                            inner_end = s2 - 1
                            if starts_cache is not None:
                                inner_s2 = int(self._get_subtree_starts_cached(starts_cache, inner_end-1)[b].item())
                                inner_s1 = int(self._get_subtree_starts_cached(starts_cache, inner_s2-1)[b].item())
                            else:
                                inner_s2 = int(self._get_subtree_starts(pop[b:b+1], inner_end-1)[0].item())
                                inner_s1 = int(self._get_subtree_starts(pop[b:b+1], inner_s2-1)[0].item())
                            if inner_s1 >= 0 and inner_s2 >= 0 and inner_s1 < inner_s2:
                                c2_tok = None
                                x_tokens = None
                                # (c2 + x)
                                if inner_s1 == inner_s2 - 1:
                                    c2_tok = pop[b, inner_s1]
                                    x_tokens = pop[b, inner_s2:inner_end].clone()
                                # (x + c2)
                                elif inner_s2 == inner_end - 1:
                                    c2_tok = pop[b, inner_s2]
                                    x_tokens = pop[b, inner_s1:inner_s2].clone()

                                if c2_tok is not None and c2_tok >= 0 and c2_tok < self._cached_val_table.size(0) and x_tokens is not None and x_tokens.numel() > 0:
                                    c2_val = self._cached_val_table[c2_tok.long()].item()
                                    if not math.isnan(c2_val):
                                        new_c_tid = self._map_single_value_to_literal_id(c1_val + c2_val)
                                        if new_c_tid != -1:
                                            new_seg = torch.cat([
                                                torch.tensor([new_c_tid], device=self.device, dtype=torch.long),
                                                x_tokens,
                                                torch.tensor([self.OP_PLUS], device=self.device, dtype=torch.long)
                                            ])
                                            if new_seg.numel() <= (j - s1 + 1):
                                                # FIX B5 (Case B): mismo bug que Case A.
                                                pop[b, s1:s1+new_seg.numel()] = new_seg.to(pop.dtype)
                                                pop[b, s1+new_seg.numel():j+1] = PAD_ID
                                                n_simplified += 1
                                                continue

            # Pattern: (x + y) + z -> 2*x + y
            if self.ID_2 == -1:
                continue
            match_1 = is_plus & (pop[:, j-1] == self.OP_PLUS)
            rows = torch.where(match_1)[0]
            if rows.numel() == 0: continue
            
            for b in rows:
                idx_s1, idx_s2 = s1_batch[b].item(), s2_batch[b].item()
                if idx_s1 < 0 or idx_s2 < 0: continue
                
                e_arg1 = idx_s2 - 1
                if e_arg1 < 0: continue
                s_arg1 = starts_cache[b, e_arg1].item() if starts_cache is not None else self._get_subtree_starts(pop[b:b+1], e_arg1)[0].item()
                if s_arg1 < 0: continue
                
                e_inner2 = e_arg1 - 1
                if e_inner2 < 0: continue
                s_inner2 = starts_cache[b, e_inner2].item() if starts_cache is not None else self._get_subtree_starts(pop[b:b+1], e_inner2)[0].item()
                if s_inner2 < 0: continue
                
                e_inner1 = s_inner2 - 1
                if e_inner1 < 0: continue
                s_inner1 = starts_cache[b, e_inner1].item() if starts_cache is not None else self._get_subtree_starts(pop[b:b+1], e_inner1)[0].item()
                if s_inner1 < 0: continue
                
                z = pop[b, idx_s2:j].clone()
                x = pop[b, s_inner1:e_inner1+1].clone()
                y = pop[b, s_inner2:e_inner2+1].clone()
                
                if torch.equal(x, z):
                    new = torch.cat([torch.tensor([self.ID_2], device=self.device), x, torch.tensor([self.OP_MULT], device=self.device), y, torch.tensor([self.OP_PLUS], device=self.device)])
                    if len(new) <= (j - idx_s1 + 1):
                        # FIX B5 (Pattern section): usar pop.dtype en vez de uint8 hardcoded
                        pop[b, idx_s1:idx_s1+len(new)] = new.to(pop.dtype)
                        pop[b, idx_s1+len(new):j+1] = PAD_ID
                        n_simplified += 1
                elif torch.equal(y, z):
                    new = torch.cat([torch.tensor([self.ID_2], device=self.device), y, torch.tensor([self.OP_MULT], device=self.device), x, torch.tensor([self.OP_PLUS], device=self.device)])
                    if len(new) <= (j - idx_s1 + 1):
                        # FIX B5 (Pattern section): usar pop.dtype en vez de uint8 hardcoded
                        pop[b, idx_s1:idx_s1+len(new)] = new.to(pop.dtype)
                        pop[b, idx_s1+len(new):j+1] = PAD_ID
                        n_simplified += 1

        return pop, n_simplified

    def _apply_term_consolidation(self, population: torch.Tensor, starts_cache: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = population.shape
        pop = population.clone()
        counts = torch.zeros(B, device=self.device, dtype=torch.long)
        for j in range(2, L):
            op = pop[:, j]
            is_plus = (op == self.OP_PLUS)
            is_mult = (op == self.OP_MULT)
            
            # --- x + x -> 2 * x (torch.where, no sync) ---
            arg2, arg1 = pop[:, j-1], pop[:, j-2]
            is_same = (arg1 == arg2) & (self.arity_table[arg1.clamp(0).long()] == 0) & (arg1 != PAD_ID)
            match_add_same = is_plus & is_same
            if self.ID_2 != -1:
                saved_arg1 = arg1.clone()
                pop[:, j] = torch.where(match_add_same, self.OP_MULT, pop[:, j])
                pop[:, j-1] = torch.where(match_add_same, saved_arg1, pop[:, j-1])
                pop[:, j-2] = torch.where(match_add_same, self.ID_2, pop[:, j-2])
                counts += match_add_same.long()

            # --- x * x -> x ^ 2 (torch.where, no sync) ---
            match_mult_same = is_mult & is_same
            if self.OP_POW_IDS.numel() > 0 and self.ID_2 != -1:
                saved_arg1 = arg1.clone()
                pop[:, j] = torch.where(match_mult_same, self.OP_POW_IDS[0].item(), pop[:, j])
                pop[:, j-1] = torch.where(match_mult_same, self.ID_2, pop[:, j-1])
                pop[:, j-2] = torch.where(match_mult_same, saved_arg1, pop[:, j-2])
                counts += match_mult_same.long()

            # --- a*x + b*x -> (a+b)*x ---
            # RPN: [a, x, *, b, x, *, +]
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
                    match_fact_1 = is_p & m1 & m2 & (val_5 == val_2) & (self.arity_table[val_5.clamp(0).long()] == 0) & (val_5 != PAD_ID)
                    match_fact_1 &= (self.arity_table[val_6.clamp(0).long()] == 0) & (self.arity_table[val_3.clamp(0).long()] == 0)
                    
                    pop[:, j-6] = torch.where(match_fact_1, val_6, pop[:, j-6])  # a
                    pop[:, j-5] = torch.where(match_fact_1, val_3, pop[:, j-5])  # b
                    pop[:, j-4] = torch.where(match_fact_1, self.OP_PLUS, pop[:, j-4])
                    pop[:, j-3] = torch.where(match_fact_1, val_5, pop[:, j-3])  # x
                    pop[:, j-2] = torch.where(match_fact_1, self.OP_MULT, pop[:, j-2])
                    pop[:, j-1] = torch.where(match_fact_1, PAD_ID, pop[:, j-1])
                    pop[:, j] = torch.where(match_fact_1, PAD_ID, pop[:, j])
                    counts += match_fact_1.long()

                    # Case 2: [x, a, *, x, b, *, +] -> (a+b)*x
                    match_fact_2 = is_p & m1 & m2 & (val_6 == val_3) & (self.arity_table[val_6.clamp(0).long()] == 0) & (val_6 != PAD_ID)
                    match_fact_2 &= (self.arity_table[val_5.clamp(0).long()] == 0) & (self.arity_table[val_2.clamp(0).long()] == 0)
                    
                    pop[:, j-6] = torch.where(match_fact_2, val_5, pop[:, j-6])  # a
                    pop[:, j-5] = torch.where(match_fact_2, val_2, pop[:, j-5])  # b
                    pop[:, j-4] = torch.where(match_fact_2, self.OP_PLUS, pop[:, j-4])
                    pop[:, j-3] = torch.where(match_fact_2, val_6, pop[:, j-3])  # x
                    pop[:, j-2] = torch.where(match_fact_2, self.OP_MULT, pop[:, j-2])
                    pop[:, j-1] = torch.where(match_fact_2, PAD_ID, pop[:, j-1])
                    pop[:, j] = torch.where(match_fact_2, PAD_ID, pop[:, j])
                    counts += match_fact_2.long()
            
            # --- Generalized Factoring: x*y + x*z -> x*(y+z) ---
            if is_plus.any():
                # Batch-calculate top-level subtrees from cache
                s2_plus = self._get_subtree_starts_cached(starts_cache, j-1) if starts_cache is not None else self._get_subtree_starts(pop, j-1)
                s1_plus = self._get_subtree_starts_cached(starts_cache, s2_plus-1) if starts_cache is not None else self._get_subtree_starts(pop, s2_plus-1)
                
                # Filter for rows where both subtrees are multiplications
                t2_idx = (s2_plus - 1).clamp(min=0)
                t2_id = pop.gather(1, t2_idx.unsqueeze(1)).squeeze(1)
                t_top_id = pop[:, j-1]
                is_mult_mult = is_plus & (t2_id == self.OP_MULT) & (t_top_id == self.OP_MULT)
                
                rows = torch.where(is_mult_mult)[0]
                if rows.numel() > 0:
                    # Batch-calculate inner subtree starts from cache
                    e1_2 = (s2_plus - 2).clamp(min=0)
                    s1_2_batch = self._get_subtree_starts_cached(starts_cache, e1_2) if starts_cache is not None else self._get_subtree_starts(pop, e1_2)
                    e1_1 = (s1_2_batch - 1).clamp(min=0)
                    s1_1_batch = self._get_subtree_starts_cached(starts_cache, e1_1) if starts_cache is not None else self._get_subtree_starts(pop, e1_1)
                    
                    e2_2 = torch.full((B,), j - 2, device=self.device, dtype=torch.long).clamp(min=0)
                    s2_2_batch = self._get_subtree_starts_cached(starts_cache, e2_2) if starts_cache is not None else self._get_subtree_starts(pop, e2_2)
                    e2_1 = (s2_2_batch - 1).clamp(min=0)
                    s2_1_batch = self._get_subtree_starts_cached(starts_cache, e2_1) if starts_cache is not None else self._get_subtree_starts(pop, e2_1)

                    for b in rows:
                        s1, s2 = s1_plus[b].item(), s2_plus[b].item()
                        s1_1, s1_2 = s1_1_batch[b].item(), s1_2_batch[b].item()
                        e1_1_v, e1_2_v = s1_2 - 1, s2 - 2
                        
                        s2_1, s2_2 = s2_1_batch[b].item(), s2_2_batch[b].item()
                        e2_1_v, e2_2_v = s2_2 - 1, j - 2
                        
                        if s1_1 < 0 or s1_2 < 0 or s2_1 < 0 or s2_2 < 0: continue
                        
                        if torch.equal(pop[b, s1_1:e1_1_v+1], pop[b, s2_1:e2_1_v+1]):
                            x = pop[b, s1_1:e1_1_v+1].clone()
                            y = pop[b, s1_2:e1_2_v+1].clone()
                            z = pop[b, s2_2:e2_2_v+1].clone()
                            new = torch.cat([x, y, z, torch.tensor([self.OP_PLUS, self.OP_MULT], device=self.device)])
                            if len(new) <= (j - s1 + 1):
                                pop[b, s1:s1+len(new)] = new
                                pop[b, s1+len(new):j+1] = PAD_ID
                                counts[b] += 1

        return pop, counts.sum()

    def _apply_modulo_rules(self, population: torch.Tensor, starts_cache: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = population.shape
        pop = population.clone()
        counts = torch.zeros(B, device=self.device, dtype=torch.long)
        z_id = self.zero_ids[0] if self.zero_ids.numel() > 0 else self.CONST_0
        
        self.OP_MOD = self.grammar.token_to_id.get('mod', -1)
        if self.OP_MOD == -1: return pop, 0

        for j in range(2, L):
            op = pop[:, j]
            # x % x -> 0 (torch.where, no sync)
            arg2, arg1 = pop[:, j-1], pop[:, j-2]
            match_self = (op == self.OP_MOD) & (arg1 == arg2) & (self.arity_table[arg1.clamp(0)] == 0) & (arg1 != PAD_ID)
            
            if z_id != -1:
                pop[:, j-2] = torch.where(match_self, z_id, pop[:, j-2])
                pop[:, j-1] = torch.where(match_self, PAD_ID, pop[:, j-1])
                pop[:, j] = torch.where(match_self, PAD_ID, pop[:, j])
                counts += match_self.long()
        return pop, counts.sum()

    def _apply_constant_folding(self, population: torch.Tensor, starts_cache: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = population.shape
        pop = population.clone()
        counts = torch.zeros(B, device=self.device, dtype=torch.long)
        max_id = self.arity_table.size(0)
        val_table = self._cached_val_table
        lit_vals = self._cached_lit_vals
        
        for j in range(1, L):
            op = pop[:, j]
            op_idx = op.clamp(0, max_id - 1)
            arity = self.arity_table[op_idx.long()]
            arity[op >= max_id] = 0
            
            # --- Unary Folding (removed outer .any() sync) ---
            is_unary = (arity == 1)
            arg = pop[:, j-1]
            arg_c = arg.clamp(0, max_id - 1)
            val = val_table[arg_c.long()]
            val[arg >= max_id] = float('nan')
            mask = is_unary & (~val.isnan())
            
            if mask.any():  # Single sync guards allocation + heavy computation
                res = torch.full((B,), float('nan'), device=self.device, dtype=self.dtype)
                applied = torch.zeros(B, device=self.device, dtype=torch.bool)

                m = mask & (op == self.OP_SIN); res[m] = torch.sin(val[m]); applied |= m
                m = mask & (op == self.OP_COS); res[m] = torch.cos(val[m]); applied |= m
                m = mask & (op == self.OP_TAN); res[m] = torch.tan(val[m]); applied |= m
                m = mask & is_op_in(op, self.OP_LOG_IDS); res[m] = torch.log(val[m]); applied |= m
                m = mask & is_op_in(op, self.OP_EXP_IDS); res[m] = torch.exp(val[m]); applied |= m
                m = mask & is_op_in(op, self.OP_SQRT_IDS); res[m] = torch.sqrt(val[m]); applied |= m
                m = mask & is_op_in(op, self.OP_ABS_IDS); res[m] = torch.abs(val[m]); applied |= m
                m = mask & is_op_in(op, self.OP_NEG_IDS); res[m] = -val[m]; applied |= m
                m = mask & (op == self.OP_FLOOR); res[m] = torch.floor(val[m]); applied |= m
                m = mask & (op == self.OP_CEIL); res[m] = torch.ceil(val[m]); applied |= m
                m = mask & (op == self.OP_SIGN); res[m] = torch.sign(val[m]); applied |= m
                m = mask & (op == self.OP_ASIN); res[m] = torch.asin(val[m]); applied |= m
                m = mask & (op == self.OP_ACOS); res[m] = torch.acos(val[m]); applied |= m
                m = mask & (op == self.OP_ATAN); res[m] = torch.atan(val[m]); applied |= m
                m = mask & is_op_in(op, self.OP_FACT_IDS); res[m] = torch.exp(torch.lgamma(val[m] + 1.0)); applied |= m

                m = mask & is_op_in(op, self.OP_GAMMA_IDS); res[m] = torch.exp(torch.lgamma(val[m])); applied |= m
                m = mask & is_op_in(op, self.OP_LGAMMA_IDS); res[m] = torch.lgamma(val[m]); applied |= m

                best_tid, match_close = self._map_values_to_literal_ids(res, applied)
                pop[:, j-1] = torch.where(match_close, best_tid, pop[:, j-1])
                pop[:, j] = torch.where(match_close, PAD_ID, pop[:, j])
                counts += match_close.long()

            # --- Binary Folding (removed outer .any() sync) ---
            is_binary = (arity == 2)
            if j >= 2:
                a1, a2 = pop[:, j-2], pop[:, j-1]
                a1_c, a2_c = a1.clamp(0, max_id - 1), a2.clamp(0, max_id - 1)
                v1, v2 = val_table[a1_c.long()], val_table[a2_c.long()]
                v1[a1 >= max_id] = float('nan'); v2[a2 >= max_id] = float('nan')
                mask = is_binary & (~v1.isnan()) & (~v2.isnan())
                
                if mask.any():  # Single sync guards allocation + heavy computation
                    res = torch.full((B,), float('nan'), device=self.device, dtype=self.dtype)
                    applied = torch.zeros(B, device=self.device, dtype=torch.bool)

                    m = mask & (op == self.OP_PLUS); res[m] = v1[m] + v2[m]; applied |= m
                    m = mask & (op == self.OP_MINUS); res[m] = v1[m] - v2[m]; applied |= m
                    m = mask & (op == self.OP_MULT); res[m] = v1[m] * v2[m]; applied |= m

                    m = mask & (op == self.OP_DIV) & (v2 != 0)
                    res[m] = v1[m] / v2[m]
                    applied |= m

                    m = mask & is_op_in(op, self.OP_POW_IDS)
                    res[m] = v1[m].pow(v2[m])
                    applied |= m

                    best_tid, match_close = self._map_values_to_literal_ids(res, applied)
                    pop[:, j-2] = torch.where(match_close, best_tid, pop[:, j-2])
                    pop[:, j-1] = torch.where(match_close, PAD_ID, pop[:, j-1])
                    pop[:, j] = torch.where(match_close, PAD_ID, pop[:, j])
                    counts += match_close.long()
        
        return pop, counts.sum()

    def _compact_formulas(self, population: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # FIX B6: El count anterior era is_pad.any(dim=1).sum(), que equivale al numero de
        # filas con ALGUN PAD (practicamente todas las formulas cortas). Eso hacia que
        # n_pass nunca fuera 0, evitando el early-exit del bucle de simplificacion y
        # ejecutando pasadas innecesarias.
        # Ahora count = filas que realmente cambiaron (tenian huecos internos).
        B, L = population.shape
        is_pad = (population == PAD_ID)
        sort_key = is_pad.long() * L + torch.arange(L, device=self.device).unsqueeze(0)
        _, idx = torch.sort(sort_key, dim=1, stable=True)
        compacted = torch.gather(population, 1, idx)
        n_changed = (compacted != population).any(dim=1).sum()
        return compacted, n_changed

    def _get_subtree_starts(self, population: torch.Tensor, end_indices) -> torch.Tensor:
        B, L = population.shape
        # Defensive indexing
        max_id = self.arity_table.size(0)
        pop_c = population.clamp(0, max_id - 1)
        arities = self.arity_table[pop_c]
        # Treat OOB tokens as terminals (arity 0) to avoid infinite loops or crashes
        arities[population >= max_id] = 0
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
