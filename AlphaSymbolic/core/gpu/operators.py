
import torch
import random
import numpy as np
from typing import List, Tuple
from AlphaSymbolic.core.grammar import OPERATORS, ExpressionTree
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
        Uses CUDA kernel for variable presence detection when available.
        """
        if self.num_variables <= 1 or GpuGlobals.VAR_FORCE_SEED_PERCENT <= 0:
            return population
        
        size = population.shape[0]
        device = self.device
        n_force = int(size * GpuGlobals.VAR_FORCE_SEED_PERCENT)
        if n_force < 1:
            return population
        
        subset = population[:n_force]  # [n_force, max_len]
        
        # ============ CUDA FAST PATH ============
        if RPN_CUDA_AVAILABLE and hasattr(rpn_cuda_native, 'compute_var_presence') and population.is_cuda:
            try:
                import rpn_cuda_native
                
                # Compute variable presence for all individuals at once
                var_presence = torch.empty(n_force, dtype=torch.int32, device=device)
                id_x_start = self.grammar.token_to_id.get('x0', 
                            self.grammar.token_to_id.get('x', 1))
                
                rpn_cuda_native.compute_var_presence(
                    subset, var_presence, PAD_ID, id_x_start, self.num_variables
                )
                
                # Target mask: all variables present
                all_vars_mask = (1 << self.num_variables) - 1  # e.g., 3 variables -> 0b111
                
                # Find individuals missing variables
                missing_mask = var_presence != all_vars_mask
                
                # Process each variable that's missing
                for vi in range(self.num_variables):
                    var_name = f'x{vi}'
                    vid = self.grammar.token_to_id.get(var_name, -1)
                    if vid <= 0:
                        continue
                    
                    var_bit = 1 << vi
                    
                    # Check if this variable is missing
                    missing_this = (var_presence & var_bit) == 0
                    n_missing = missing_this.sum().item()
                    
                    if n_missing == 0:
                        continue
                    
                    missing_idx = missing_this.nonzero(as_tuple=True)[0]
                    missing_pop = subset[missing_idx]
                    
                    # Find terminal positions (arity-0 tokens, non-PAD)
                    is_term = torch.zeros_like(missing_pop, dtype=torch.bool)
                    for tid in self.arity_0_ids:
                        is_term |= (missing_pop == tid.item())
                    
                    has_any = is_term.any(dim=1)
                    if not has_any.any():
                        continue
                    
                    valid_idx = has_any.nonzero(as_tuple=True)[0]
                    valid_pop = missing_pop[valid_idx]
                    valid_term = is_term[valid_idx]
                    
                    rand_w = torch.rand_like(valid_term.float()) * valid_term.float()
                    pos = rand_w.argmax(dim=1)
                    valid_pop[torch.arange(valid_idx.shape[0], device=device), pos] = vid
                    
                    missing_pop[valid_idx] = valid_pop
                    
                    # Update var_presence for newly added variable
                    var_presence[missing_idx[valid_idx]] |= var_bit
                
                return population
                
            except Exception:
                pass  # Fall through to PyTorch fallback
        
        # ============ PyTorch FALLBACK ============
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
                
                # OPTIMIZED: calcular pesos de categoría desde OPERATOR_WEIGHTS (bug fix: antes 1.0 fijo)
                _op_w = GpuGlobals.OPERATOR_WEIGHTS
                _bin_sum = sum(_op_w[:6])           # +,-,*,/,**,%
                _una_sum = sum(_op_w[6:])           # sin,cos,tan,log,exp,fact,...,sqrt,abs
                _op_sum = max(_bin_sum + _una_sum, 1e-6)
                _t_frac = float(GpuGlobals.TERMINAL_VS_VARIABLE_PROB)
                _o_frac = 1.0 - _t_frac
                _gen_term_w  = _t_frac
                _gen_unary_w = float(_o_frac * _una_sum / _op_sum)
                _gen_bin_w   = float(_o_frac * _bin_sum / _op_sum)

                rpn_cuda_native.generate_random_rpn(
                    population, t_ids, u_ids, b_ids, seed,
                    _gen_term_w, _gen_unary_w, _gen_bin_w
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
        Optimized via C++ CUDA Kernel.
        """
        if RPN_CUDA_AVAILABLE and hasattr(rpn_cuda_native, 'validate_rpn_batch') and population.is_cuda:
            try:
                B, L = population.shape
                valid_mask = torch.zeros(B, dtype=torch.bool, device=self.device)
                if not hasattr(self, 'token_arity_int'): self.token_arity_int = self.token_arity.to(dtype=torch.int32)
                rpn_cuda_native.validate_rpn_batch(population, self.token_arity_int, valid_mask, PAD_ID)
                return valid_mask
            except Exception as e:
                pass # Fallback

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
        stack_prefix = deltas.cumsum(dim=1)
        non_pad = ~is_pad
        prefix_ok = torch.where(non_pad, stack_prefix >= 1, torch.ones_like(non_pad))
        no_underflow = prefix_ok.all(dim=1)

        pad_seen_before = is_pad.to(torch.int32).cumsum(dim=1) > 0
        nonpad_after_pad = non_pad & pad_seen_before
        contiguous_ok = ~nonpad_after_pad.any(dim=1)

        return (final_stack == 1) & no_underflow & contiguous_ok

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
                # OPTIMIZED: mismos pesos de categoría que generate_random_population_gpu
                _op_w = GpuGlobals.OPERATOR_WEIGHTS
                _bin_sum = sum(_op_w[:6])
                _una_sum = sum(_op_w[6:])
                _op_sum = max(_bin_sum + _una_sum, 1e-6)
                _t_frac = float(GpuGlobals.TERMINAL_VS_VARIABLE_PROB)
                _o_frac = 1.0 - _t_frac
                rpn_cuda_native.generate_random_rpn(
                    population, t_ids, u_ids, b_ids, seed,
                    _t_frac,
                    float(_o_frac * _una_sum / _op_sum),
                    float(_o_frac * _bin_sum / _op_sum)
                )
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
        arities = self.token_arity[population.clamp(0, self.token_arity.shape[0] - 1).long()]
        deltas = 1 - arities
        is_pad = (population == PAD_ID)
        deltas[is_pad] = 0
        final_stack = deltas.sum(dim=1)
        stack_prefix = deltas.cumsum(dim=1)
        non_pad = ~is_pad
        prefix_ok = torch.where(non_pad, stack_prefix >= 1, torch.ones_like(non_pad))
        no_underflow = prefix_ok.all(dim=1)

        pad_seen_before = is_pad.to(torch.int32).cumsum(dim=1) > 0
        nonpad_after_pad = non_pad & pad_seen_before
        contiguous_ok = ~nonpad_after_pad.any(dim=1)

        return (final_stack == 1) & no_underflow & contiguous_ok

    def _compute_depth(self, population: torch.Tensor, starts_mat: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the maximum depth of each RPN tree in the population.
        Vectorized O(L) forward pass.
        Returns tensor [B] with max depth (1-indexed) per tree.
        """
        B, L = population.shape
        device = self.device
        if starts_mat is None:
            starts_mat = self._get_subtree_ranges(population)
            
        arities = self.token_arity[population.clamp(0, self.token_arity.shape[0] - 1).long()]
        arities[population == PAD_ID] = 0
        is_pad = (population == PAD_ID)
        
        depths = torch.zeros(B, L, dtype=torch.long, device=device)
        
        for j in range(L):
            ar = arities[:, j]
            if j == 0:
                depths[:, 0] = torch.where(is_pad[:, 0], 0, 1)
                continue
                
            is_term = (ar == 0) & ~is_pad[:, j]
            is_unary = (ar == 1) & ~is_pad[:, j]
            is_binary = (ar == 2) & ~is_pad[:, j]
            
            cur_depth = torch.where(is_term, 1, 0)
            
            if is_unary.any():
                cur_depth = torch.where(is_unary, 1 + depths[:, j-1], cur_depth)
                
            if is_binary.any():
                right_depth = depths[:, j-1]
                left_child_end = (starts_mat[:, j-1] - 1).clamp(0, L - 1)
                left_depth = depths.gather(1, left_child_end.unsqueeze(1)).squeeze(1)
                max_child_depth = torch.max(right_depth, left_depth)
                cur_depth = torch.where(is_binary, 1 + max_child_depth, cur_depth)
                
            depths[:, j] = torch.where(is_pad[:, j], 0, cur_depth)
            
        return depths.max(dim=1).values

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

    def mutate_population(self, population: torch.Tensor, constants_or_rate, mutation_rate: float = None):
        """
        Performs arity-safe mutation on the population.
        Now preserves constants for point mutations.
        """
        legacy_no_constants = mutation_rate is None
        if legacy_no_constants:
            mutation_rate = float(constants_or_rate)
            constants = torch.zeros((population.shape[0], 1), device=self.device, dtype=self.dtype)
        else:
            constants = constants_or_rate

        population = population.clone()
        constants = constants.clone()
        
        # BUG FIX: Check if population is on CUDA before using CUDA kernel
        if RPN_CUDA_AVAILABLE and hasattr(rpn_cuda_native, 'mutate_population') and population.is_cuda:
             # CUDA Fast Path
             # Note: Original native kernel might not handle constants yet.
             # If so, we'll need to repair after returning.
             B, L = population.shape
             rand_floats = torch.rand(population.shape, device=self.device, dtype=torch.float32)
             rand_ints = torch.randint(0, 2**30, population.shape, device=self.device, dtype=torch.long)
             
             rpn_cuda_native.mutate_population(
                 population, rand_floats, rand_ints,
                 self.token_arity_int,
                 self.arity_0_ids, self.arity_1_ids, self.arity_2_ids,
                 mutation_rate, PAD_ID
             )
             
             # Repair constants alignment if RPN changed
             # (Point mutation doesn't break RPN structure, but can change 'C' count)
             population, constants, _ = self.repair_invalid_population(population, constants)
             return population if legacy_no_constants else (population, constants)

        # Fallback to PyTorch
        B, L = population.shape
        K = constants.shape[1]
        mask = torch.rand_like(population, dtype=self.dtype) < mutation_rate
        mask = mask & (population != PAD_ID)
        
        token_arity_local = self.token_arity.to(population.device)
        current_arities = token_arity_local[population.long()]
        
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
             
        # Guardar constantes originales para mapeo
        id_C = self.grammar.token_to_id.get('C', PAD_ID)
        is_c_orig = (population == id_C)
        c_map_orig = torch.cumsum(is_c_orig.long(), dim=1) - 1
        
        # Aplicar mutación a RPN
        mask_0 = mask & (current_arities == 0)
        population = torch.where(mask_0, replacements_0, population)
        
        mask_1 = mask & (current_arities == 1)
        population = torch.where(mask_1, replacements_1, population)
        
        mask_2 = mask & (current_arities == 2)
        population = torch.where(mask_2, replacements_2, population)

        if legacy_no_constants:
            return population
        
        # Realinear constantes
        is_c_new = (population == id_C)
        next_c = torch.zeros_like(constants)
        
        if is_c_new.any():
            c_idx_new = torch.cumsum(is_c_new.long(), dim=1) - 1
            
            # Si una posición era 'C' y sigue siendo 'C', mantiene el valor.
            # Si era otra cosa y se convirtió en 'C', recibe valor aleatorio.
            # Si era 'C' y se convirtió en otra cosa, el valor se pierde.
            
            # Mascara de los que eran 'C' y siguen siéndolo
            still_c = is_c_orig & is_c_new
            # Mascara de los nuevos 'C'
            became_c = (~is_c_orig) & is_c_new
            
            # Map constants from old to new
            if still_c.any():
                src_idx = c_map_orig[still_c]
                dst_idx = c_idx_new[still_c]
                rows = torch.nonzero(still_c, as_tuple=True)[0]
                
                # Scatter values
                next_c[rows, dst_idx] = constants[rows, src_idx]
                
            if became_c.any():
                # Random initialization for new constants
                rows = torch.nonzero(became_c, as_tuple=True)[0]
                dst_idx = c_idx_new[became_c]
                
                # Jitter original or uniform? Let's use uniform as it is a "new" constant.
                new_vals = torch.empty(len(rows), device=self.device, dtype=self.dtype).uniform_(
                    GpuGlobals.CONSTANT_MIN_VALUE, GpuGlobals.CONSTANT_MAX_VALUE
                )
                next_c[rows, dst_idx] = new_vals

        return population if legacy_no_constants else (population, next_c)


    def _get_subtree_ranges(self, population: torch.Tensor) -> torch.Tensor:
        """
        Calculates the start index of the subtree ending at each position.
        Returns tensor [B, L] where value is start_index, or -1 if invalid/padding.
        """
        # FIX CUDA-CPU: Solo usar el kernel CUDA si el tensor está en dispositivo CUDA.
        # Antes, el kernel se llamaba incluso con tensores CPU, causando RuntimeError.
        if RPN_CUDA_AVAILABLE and hasattr(rpn_cuda_native, 'find_subtree_ranges') and population.is_cuda:
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
        # FIX B1-idx: Si population es uint8, PyTorch lo interpreta como mascara booleana
        # en lugar de indices. Castear a long() garantiza indexacion correcta.
        arities = token_net_change[population.long()]
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

    def _depth_fair_sample(self, parents: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Depth-Fair crossover point sampling.

        Standard multinomial over nodes is biased toward large subtrees
        (more tokens → more often picked). This method equalizes by:
          1. Compute depth of each RPN node using cumulative arity stack.
          2. Pick a random depth d uniformly in [0, max_depth].
          3. Pick uniformly among valid nodes at that depth.

        Args:
            parents:    [B, L] RPN integer tensor.
            valid_mask: [B, L] bool, True where crossover point is valid.

        Returns:
            end_idx: [B] long tensor — the sampled end-of-subtree index.
        """
        B, L = parents.shape
        device = self.device

        # Compute per-token stack deltas: terminal(+1), unary(0), binary(-1), pad(0)
        arities = self.token_arity[parents.clamp(0, self.token_arity.shape[0] - 1).long()]
        deltas = (1 - arities).long()  # [B, L]
        is_pad = (parents == PAD_ID)
        deltas[is_pad] = 0

        # Cumulative stack depth at each position = "depth level" proxy
        # Stack value before consuming position i = cumsum up to i (exclusive) + 1
        # We use cumsum(deltas) and shift: depth[i] = cumsum(deltas)[:i+1].sum()
        # Simplified: use running stack height as depth proxy
        depths = torch.cumsum(deltas, dim=1)  # [B, L], stack height after position i

        # Clamp to non-negative (invalid RPNs may go negative)
        depths = depths.clamp(min=0)

        # Mask out non-valid positions
        # valid_mask: True where subtree_start != -1
        depths_valid = depths * valid_mask.long()  # zero invalid positions

        # For each individual: how many distinct depth levels exist?
        max_depths = depths_valid.max(dim=1).values  # [B]

        # Sample target depth: uniform in [0, max_depth] (using rand * (max+1) as integer)
        # +1 to include depth=0 (leaf nodes at the root level when stack=1)
        target_depth = (torch.rand(B, device=device) * (max_depths.float() + 1)).long()  # [B]

        # Build a per-position mask: node is at target depth AND valid
        at_target_depth = (depths_valid == target_depth.unsqueeze(1)) & valid_mask  # [B, L]

        # Fallback: if no nodes match target depth, use all valid nodes
        no_match = ~at_target_depth.any(dim=1)  # [B]
        if no_match.any():
            at_target_depth[no_match] = valid_mask[no_match]

        # Uniform sample from matching nodes
        sample_probs = at_target_depth.float() + 1e-9
        end_idx = torch.multinomial(sample_probs, 1).squeeze(1)  # [B]

        return end_idx

    def crossover_population(self, parents: torch.Tensor, constants_or_rate, crossover_rate: float = None):
        legacy_no_constants = crossover_rate is None
        if legacy_no_constants:
            crossover_rate = float(constants_or_rate)
            constants = torch.zeros((parents.shape[0], 1), device=self.device, dtype=self.dtype)
        else:
            constants = constants_or_rate

        # FIX B1: Clonar entrada para evitar mutacion in-place del tensor original.
        parents = parents.clone()
        constants = constants.clone()
        B, L = parents.shape
        K = constants.shape[1]
        n_pairs = int(B * 0.5 * crossover_rate)
        if n_pairs == 0:
            return parents.clone() if legacy_no_constants else (parents.clone(), constants.clone())
        
        perm = torch.randperm(B, device=self.device)
        p1_idx = perm[:n_pairs*2:2]
        p2_idx = perm[1:n_pairs*2:2]
        
        parents_1 = parents[p1_idx]
        parents_2 = parents[p2_idx]
        consts_1 = constants[p1_idx]
        consts_2 = constants[p2_idx]
        
        starts_1_mat = self._get_subtree_ranges(parents_1)
        starts_2_mat = self._get_subtree_ranges(parents_2)
        
        valid_mask_1 = (starts_1_mat != -1)
        valid_mask_2 = (starts_2_mat != -1)
        
        # Sample crossover points
        _depth_fair = getattr(GpuGlobals, 'DEPTH_FAIR_CROSSOVER', False)
        if _depth_fair:
            try:
                end_1 = self._depth_fair_sample(parents_1, valid_mask_1)
                end_2 = self._depth_fair_sample(parents_2, valid_mask_2)
            except Exception:
                probs_1 = valid_mask_1.float() + 1e-6
                probs_2 = valid_mask_2.float() + 1e-6
                end_1 = torch.multinomial(probs_1, 1).squeeze(1)
                end_2 = torch.multinomial(probs_2, 1).squeeze(1)
        else:
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
        
        id_C = self.grammar.token_to_id.get('C', PAD_ID)
        
        if RPN_CUDA_AVAILABLE and hasattr(rpn_cuda_native, 'crossover_splicing') and parents.is_cuda:
             # CUDA Fast Splicing
             c1 = torch.full((n_pairs, L), PAD_ID, dtype=self.pop_dtype, device=self.device)
             c2 = torch.full((n_pairs, L), PAD_ID, dtype=self.pop_dtype, device=self.device)
             c1_c = torch.zeros((n_pairs, K), dtype=self.dtype, device=self.device)
             c2_c = torch.zeros((n_pairs, K), dtype=self.dtype, device=self.device)
             
             rpn_cuda_native.crossover_splicing(
                 parents_1, parents_2,
                 start_1, end_1,
                 start_2, end_2,
                 c1, c2, PAD_ID
             )
             
             # Native Constant Splicing
             if hasattr(rpn_cuda_native, 'crossover_constants'):
                 rpn_cuda_native.crossover_constants(
                     parents_1, parents_2,
                     consts_1, consts_2,
                     start_1, end_1, start_2, end_2,
                     c1_c, c2_c,
                     n_pairs, L, K, id_C
                 )
             else:
                 c1_c = consts_1.clone()
                 c2_c = consts_2.clone()
        else:
            # PyTorch Fallback
            grid = torch.arange(L, device=self.device).unsqueeze(0).expand(n_pairs, L)
            
            # Child 1: pre from p1, mid from p2, post from p1
            cut_1 = len_1_pre + len_2_sub
            mask_c1_pre = (grid < len_1_pre.unsqueeze(1))
            mask_c1_mid = (grid >= len_1_pre.unsqueeze(1)) & (grid < cut_1.unsqueeze(1))
            mask_c1_post = (grid >= cut_1.unsqueeze(1))
            
            idx_from_p1_c1 = torch.where(mask_c1_pre, grid, 
                                torch.where(mask_c1_post, grid - cut_1.unsqueeze(1) + end_1.unsqueeze(1) + 1, grid))
            idx_from_p2_c1 = grid - len_1_pre.unsqueeze(1) + start_2.unsqueeze(1)
            
            use_p2_c1 = mask_c1_mid.long() 
            
            safe_idx_p1_c1 = torch.clamp(idx_from_p1_c1, 0, L-1)
            safe_idx_p2_c1 = torch.clamp(idx_from_p2_c1, 0, L-1)
            val_p1_c1 = parents_1.gather(1, safe_idx_p1_c1)
            val_p2_c1 = parents_2.gather(1, safe_idx_p2_c1)
            c1 = torch.where(use_p2_c1 == 1, val_p2_c1, val_p1_c1)
            
            len_p1 = (parents_1 != PAD_ID).sum(dim=1)
            len_p2 = (parents_2 != PAD_ID).sum(dim=1)
            
            invalid_c1 = (mask_c1_pre & (grid >= len_p1.unsqueeze(1))) | \
                         (mask_c1_mid & (idx_from_p2_c1 >= len_p2.unsqueeze(1))) | \
                         (mask_c1_post & (idx_from_p1_c1 >= len_p1.unsqueeze(1)))
            c1[invalid_c1] = PAD_ID
            
            # --- Constant Splicing Child 1 ---
            is_c_p1 = (parents_1 == id_C)
            is_c_p2 = (parents_2 == id_C)
            c_map_p1 = torch.cumsum(is_c_p1.long(), dim=1) - 1
            c_map_p2 = torch.cumsum(is_c_p2.long(), dim=1) - 1
            
            is_c_c1 = (c1 == id_C)
            c1_c = torch.zeros((n_pairs, K), device=self.device, dtype=self.dtype)
            
            if is_c_c1.any():
                idx_in_vcl_1 = torch.where(use_p2_c1 == 0, 
                                          c_map_p1.gather(1, safe_idx_p1_c1),
                                          c_map_p2.gather(1, safe_idx_p2_c1))
                c1_c_idx = torch.cumsum(is_c_c1.long(), dim=1) - 1
                mask_valid_c1 = is_c_c1 & (c1_c_idx < K) & (idx_in_vcl_1 >= 0) & (idx_in_vcl_1 < K)
                if mask_valid_c1.any():
                    vals_from_source = torch.where(use_p2_c1 == 0,
                                                  consts_1.gather(1, idx_in_vcl_1.clamp(0, K-1)),
                                                  consts_2.gather(1, idx_in_vcl_1.clamp(0, K-1)))
                    # Use (row, col) indexing for correct 2D assignment
                    row_ids, _ = torch.nonzero(mask_valid_c1, as_tuple=True)
                    col_ids = c1_c_idx[mask_valid_c1].clamp(0, K-1)
                    c1_c[row_ids, col_ids] = vals_from_source[mask_valid_c1]

            # Child 2 logic
            cut_2 = len_2_pre + len_1_sub
            mask_c2_pre = (grid < len_2_pre.unsqueeze(1))
            mask_c2_mid = (grid >= len_2_pre.unsqueeze(1)) & (grid < cut_2.unsqueeze(1))
            mask_c2_post = (grid >= cut_2.unsqueeze(1))
            
            idx_from_p2_c2 = torch.where(mask_c2_pre, grid,
                                torch.where(mask_c2_post, grid - cut_2.unsqueeze(1) + end_2.unsqueeze(1) + 1, grid))
            idx_from_p1_c2 = grid - len_2_pre.unsqueeze(1) + start_1.unsqueeze(1)
            
            use_p1_c2 = mask_c2_mid.long() 
            
            safe_idx_p2_c2 = torch.clamp(idx_from_p2_c2, 0, L-1)
            safe_idx_p1_c2 = torch.clamp(idx_from_p1_c2, 0, L-1)
            val_p2_c2 = parents_2.gather(1, safe_idx_p2_c2)
            val_p1_c2 = parents_1.gather(1, safe_idx_p1_c2)
            c2 = torch.where(use_p1_c2 == 1, val_p1_c2, val_p2_c2)
            
            invalid_c2 = (mask_c2_pre & (grid >= len_p2.unsqueeze(1))) | \
                         (mask_c2_mid & (idx_from_p1_c2 >= len_p1.unsqueeze(1))) | \
                         (mask_c2_post & (idx_from_p2_c2 >= len_p2.unsqueeze(1)))
            c2[invalid_c2] = PAD_ID
            
            # --- Constant Splicing Child 2 ---
            is_c_c2 = (c2 == id_C)
            c2_c = torch.zeros((n_pairs, K), device=self.device, dtype=self.dtype)
            if is_c_c2.any():
                idx_in_vcl_2 = torch.where(use_p1_c2 == 0,
                                          c_map_p2.gather(1, safe_idx_p2_c2),
                                          c_map_p1.gather(1, safe_idx_p1_c2))
                c2_c_idx = torch.cumsum(is_c_c2.long(), dim=1) - 1
                mask_valid_c2 = is_c_c2 & (c2_c_idx < K) & (idx_in_vcl_2 >= 0) & (idx_in_vcl_2 < K)
                if mask_valid_c2.any():
                    vals_from_source = torch.where(use_p1_c2 == 0,
                                                  consts_2.gather(1, idx_in_vcl_2.clamp(0, K-1)),
                                                  consts_1.gather(1, idx_in_vcl_2.clamp(0, K-1)))
                    row_ids, _ = torch.nonzero(mask_valid_c2, as_tuple=True)
                    col_ids = c2_c_idx[mask_valid_c2].clamp(0, K-1)
                    c2_c[row_ids, col_ids] = vals_from_source[mask_valid_c2]

        # --- Depth Limit Check ---
        if GpuGlobals.USE_HARD_DEPTH_LIMIT:
            hard_limit = GpuGlobals.MAX_TREE_DEPTH_HARD_LIMIT
            c1_depth = self._compute_depth(c1)
            c2_depth = self._compute_depth(c2)
            too_long_1 = c1_depth > hard_limit
            too_long_2 = c2_depth > hard_limit
            if too_long_1.any():
                c1[too_long_1] = parents_1[too_long_1]
                c1_c[too_long_1] = consts_1[too_long_1]
            if too_long_2.any():
                c2[too_long_2] = parents_2[too_long_2]
                c2_c[too_long_2] = consts_2[too_long_2]
        
        parents[p1_idx] = c1
        parents[p2_idx] = c2
        constants[p1_idx] = c1_c
        constants[p2_idx] = c2_c
        
        valid_mask = self._validate_rpn_batch(parents)
        n_invalid = (~valid_mask).sum().item()
        if n_invalid > 0:
            parents, constants, _ = self.repair_invalid_population(parents, constants)
        
        return parents if legacy_no_constants else (parents, constants)


    def deduplicate_population(self, population: torch.Tensor, constants: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if not GpuGlobals.PREVENT_DUPLICATES:
            return population, constants, 0
        
        B = population.shape[0]
        curr_L = population.shape[1]
        
        # ============ CUDA FAST PATH (No CPU Sync) ============
        if RPN_CUDA_AVAILABLE and hasattr(rpn_cuda_native, 'compute_population_hashes') and population.is_cuda:
            try:
                # 1. Compute structural hashes on GPU
                hashes = torch.empty(B, dtype=torch.long, device=self.device)
                var_presence = torch.empty(B, dtype=torch.int32, device=self.device)
                
                # Get variable token start ID
                id_x_start = self.grammar.token_to_id.get('x0', 
                            self.grammar.token_to_id.get('x', 1))
                
                rpn_cuda_native.compute_population_hashes(
                    population, hashes, var_presence,
                    PAD_ID, id_x_start, self.num_variables
                )
                
                # 2. Structural dedup on GPU (atomic hash table)
                # Hash table size: 2^20 = 1M entries
                HASH_TABLE_SIZE = 1 << 20
                hash_table = torch.full((HASH_TABLE_SIZE,), -1, dtype=torch.long, device=self.device)
                duplicate_mask = torch.zeros(B, dtype=torch.int32, device=self.device)
                original_index = torch.zeros(B, dtype=torch.long, device=self.device)
                
                rpn_cuda_native.structural_dedup(hashes, hash_table, duplicate_mask, original_index)
                
                # 3. Get replacement positions on GPU
                replacement_positions = torch.empty(B, dtype=torch.long, device=self.device)
                n_replacements = torch.zeros(1, dtype=torch.long, device=self.device)
                
                n_dups = rpn_cuda_native.get_replacement_positions(
                    duplicate_mask, replacement_positions, n_replacements
                )
                
                if n_dups == 0:
                    return population, constants, 0
                
                # 4. Generate replacements
                dup_indices = replacement_positions[:n_dups]
                fresh_pop = self.generate_random_population(n_dups)

                # Handle shape mismatch
                if fresh_pop.shape[1] != curr_L:
                    if fresh_pop.shape[1] < curr_L:
                        pad = torch.full(
                            (n_dups, curr_L - fresh_pop.shape[1]), PAD_ID,
                            dtype=fresh_pop.dtype, device=self.device
                        )
                        fresh_pop = torch.cat([fresh_pop, pad], dim=1)
                    else:
                        fresh_pop = fresh_pop[:, :curr_L]

                # Constants for replacements
                K = constants.shape[1]
                if GpuGlobals.FORCE_INTEGER_CONSTANTS:
                    fresh_consts = torch.randint(
                        GpuGlobals.CONSTANT_INT_MIN_VALUE, 
                        GpuGlobals.CONSTANT_INT_MAX_VALUE + 1,
                        (n_dups, K), device=self.device, dtype=torch.long
                    ).to(self.dtype)
                else:
                    fresh_consts = torch.empty(n_dups, K, device=self.device, dtype=self.dtype).uniform_(
                        GpuGlobals.CONSTANT_MIN_VALUE, GpuGlobals.CONSTANT_MAX_VALUE
                    )
                
                population[dup_indices] = fresh_pop
                constants[dup_indices] = fresh_consts
                
                return population, constants, n_dups
                
            except Exception as e:
                # Fall through to PyTorch fallback
                pass
        
        # ============ PyTorch FALLBACK ============
        # Collision-free structural dedup (exact RPN row equality).
        # This avoids false positives when different individuals share the same hash.
        _, inverse_indices = torch.unique(population, dim=0, sorted=False, return_inverse=True)
        sorted_inv, sorted_idx = torch.sort(inverse_indices)
        mask_dup = torch.zeros_like(sorted_inv, dtype=torch.bool)
        mask_dup[1:] = (sorted_inv[1:] == sorted_inv[:-1])
        dup_indices = sorted_idx[mask_dup]
        n_dups = dup_indices.shape[0]
        
        if n_dups == 0:
            return population, constants, 0
            
        fresh_pop = self.generate_random_population(n_dups)
        if fresh_pop.shape[1] != curr_L:
            if fresh_pop.shape[1] < curr_L:
                pad = torch.full(
                    (n_dups, curr_L - fresh_pop.shape[1]), PAD_ID,
                    dtype=fresh_pop.dtype, device=self.device
                )
                fresh_pop = torch.cat([fresh_pop, pad], dim=1)
            else:
                fresh_pop = fresh_pop[:, :curr_L]

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

    def subtree_mutation(self, population: torch.Tensor, constants_or_rate, mutation_rate: float = None):
        """
        Replaces a random subtree with a newly generated random tree.
        Crucial for structural exploration (Bloat control + Innovation).
        Now preserves constants in Python fallback.
        """
        legacy_no_constants = mutation_rate is None
        if legacy_no_constants:
            mutation_rate = float(constants_or_rate)
            constants = torch.zeros((population.shape[0], 1), device=self.device, dtype=self.dtype)
        else:
            constants = constants_or_rate

        population = population.clone()
        constants = constants.clone()
        B, L = population.shape
        K = constants.shape[1]
        
        starts_mat = self._get_subtree_ranges(population)
        valid_mask = (starts_mat != -1)
        mutate_mask = (torch.rand(B, device=self.device) < mutation_rate) & valid_mask.any(dim=1)
        
        if not mutate_mask.any():
            return population if legacy_no_constants else (population, constants)
            
        mutant_indices = torch.nonzero(mutate_mask).squeeze(1)
        n_mut = mutant_indices.numel()
        
        probs = valid_mask[mutate_mask].float() + 1e-6
        ends = torch.multinomial(probs, 1).squeeze(1)
        starts = starts_mat[mutate_mask].gather(1, ends.unsqueeze(1)).squeeze(1)
        remove_lens = ends - starts + 1
        
        max_subtree_len = min(L, GpuGlobals.MAX_TREE_DEPTH_MUTATION * 2 + 1)
        # Generate new RPN subtrees
        replacements_raw = self._generate_small_subtrees(n_mut, max_subtree_len)
        # Generate matching random constants for the replacement subtrees
        if GpuGlobals.FORCE_INTEGER_CONSTANTS:
            repl_consts_raw = torch.randint(
                GpuGlobals.CONSTANT_INT_MIN_VALUE,
                GpuGlobals.CONSTANT_INT_MAX_VALUE + 1,
                (n_mut, K), device=self.device, dtype=torch.long
            ).to(self.dtype)
        else:
            repl_consts_raw = torch.empty(n_mut, K, device=self.device, dtype=self.dtype).uniform_(
                GpuGlobals.CONSTANT_MIN_VALUE, GpuGlobals.CONSTANT_MAX_VALUE
            )
        
        replacements = torch.full((n_mut, L), PAD_ID, dtype=self.pop_dtype, device=self.device)
        replacements[:, :max_subtree_len] = replacements_raw
        repl_lengths = (replacements != PAD_ID).sum(dim=1)
        
        old_lens = (population[mutant_indices] != PAD_ID).sum(dim=1)
        new_total_lens = old_lens - remove_lens + repl_lengths
        fits = new_total_lens <= L
        
        if not fits.any():
            return population if legacy_no_constants else (population, constants)
            
        valid_idx_rel = torch.nonzero(fits).squeeze(1) 
        final_mutant_pop_idx = mutant_indices[valid_idx_rel]
        
        orig_seqs = population[final_mutant_pop_idx]
        new_seqs = replacements[valid_idx_rel]
        new_c_repl = repl_consts_raw[valid_idx_rel]
        
        st = starts[valid_idx_rel]
        en = ends[valid_idx_rel]
        r_len = repl_lengths[valid_idx_rel]
        
        # Build new sequences and constants
        grid = torch.arange(L, device=self.device).unsqueeze(0).expand(len(valid_idx_rel), L)
        out_seqs = torch.full_like(orig_seqs, PAD_ID)
        out_consts = torch.zeros((len(valid_idx_rel), K), device=self.device, dtype=self.dtype)
        
        id_C = self.grammar.token_to_id.get('C', PAD_ID)
        orig_consts = constants[final_mutant_pop_idx]
        
        # Splicing RPN
        mask_pre = grid < st.unsqueeze(1)
        limit_new = st + r_len
        mask_new = (grid >= st.unsqueeze(1)) & (grid < limit_new.unsqueeze(1))
        mask_post = (grid >= limit_new.unsqueeze(1))
        
        shift = (en + 1) - limit_new
        src_idx_post = grid + shift.unsqueeze(1)
        valid_src = (src_idx_post < L)
        mask_post = mask_post & valid_src
        
        out_seqs[mask_pre] = orig_seqs[mask_pre]
        
        gather_idx_new = (grid - st.unsqueeze(1)).clamp(0, L-1)
        new_seqs_vals = new_seqs.gather(1, gather_idx_new).to(out_seqs.dtype)
        out_seqs[mask_new] = new_seqs_vals[mask_new]
        
        gather_idx_post = torch.clamp(src_idx_post, 0, L-1)
        out_seqs[mask_post] = orig_seqs.gather(1, gather_idx_post)[mask_post]
        
        # Splicing Constants
        # 1. Map 'C' in sources
        c_map_orig = torch.cumsum((orig_seqs == id_C).long(), dim=1) - 1
        c_map_new = torch.cumsum((new_seqs == id_C).long(), dim=1) - 1
        
        # 2. Map 'C' in output
        is_c_out = (out_seqs == id_C)
        if is_c_out.any():
            c_out_idx = torch.cumsum(is_c_out.long(), dim=1) - 1
            
            # Determinar fuente constante para cada posición 'C' en la salida
            # Pre: orig_consts[c_map_orig[grid]]
            # New: new_c_repl[c_map_new[grid-st]]
            # Post: orig_consts[c_map_orig[grid+shift]]
            
            src_c_idx = torch.where(mask_pre, c_map_orig,
                                   torch.where(mask_new, c_map_new.gather(1, gather_idx_new),
                                              c_map_orig.gather(1, gather_idx_post)))
            
            mask_valid_c = is_c_out & (c_out_idx < K) & (src_c_idx >= 0) & (src_c_idx < K)
            if mask_valid_c.any():
                vals_from_source = torch.where(mask_pre, 
                                             orig_consts.gather(1, src_c_idx.clamp(0, K-1)),
                                             torch.where(mask_new,
                                                        new_c_repl.gather(1, src_c_idx.clamp(0, K-1)),
                                                        orig_consts.gather(1, src_c_idx.clamp(0, K-1))))
                # Use (row, col) indexing for correct 2D assignment
                row_ids, _ = torch.nonzero(mask_valid_c, as_tuple=True)
                col_ids = c_out_idx[mask_valid_c].clamp(0, K-1)
                out_consts[row_ids, col_ids] = vals_from_source[mask_valid_c]
        
        population[final_mutant_pop_idx] = out_seqs
        constants[final_mutant_pop_idx] = out_consts
        
        return population if legacy_no_constants else (population, constants)


    def repair_invalid_population(self, population: torch.Tensor, constants: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Replace invalid RPN individuals with fresh random individuals.
        This prevents evolutionary collapse to trivial fallbacks.
        """
        valid_mask = self._validate_rpn_batch(population)
        invalid_mask = ~valid_mask
        
        # Asynchronous evaluation: no CPU sync via .item() unless absolutely needed
        if not invalid_mask.any():
            return population, constants, 0
            
        n_invalid = int(invalid_mask.sum().item())

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

    # ================================================================
    #   SOTA P0 — Constant Perturbation
    # ================================================================
    def constant_perturbation(
        self,
        constants: torch.Tensor,
        rate: float = 0.05,
        sigma: float = 0.01,
    ) -> torch.Tensor:
        """
        Apply multiplicative Gaussian noise to a fraction of constants.

        For each constant c, the perturbed value is:
            c' = c + N(0, |c| * sigma + eps)

        This keeps constants near their current value while exploring
        the local neighbourhood — complementary to PSO which explores
        globally.

        Args:
            constants: [B, K] float tensor of current constant values.
            rate:  Fraction of individuals that receive perturbation.
            sigma: Relative noise magnitude (0.01 = 1% of |c|).

        Returns:
            constants tensor (modified in-place, also returned for chaining).
        """
        B, K = constants.shape
        # Decide which individuals get perturbed
        perturb_mask = torch.rand(B, device=self.device) < rate  # [B]
        if not perturb_mask.any():
            return constants

        # Build scale: |c| * sigma + small absolute floor (avoids 0-noise for c≈0)
        eps = 1e-4
        scale = constants[perturb_mask].abs() * sigma + eps  # [n_pert, K]

        noise = torch.randn(scale.shape, device=self.device, dtype=constants.dtype) * scale
        constants[perturb_mask] = constants[perturb_mask] + noise

        # Clamp to valid constant range
        c_min = float(getattr(GpuGlobals, 'CONSTANT_MIN_VALUE', -10.0))
        c_max = float(getattr(GpuGlobals, 'CONSTANT_MAX_VALUE', 10.0))
        constants.clamp_(c_min, c_max)

        return constants

    # ================================================================
    #   SOTA P0 — Headless Chicken Crossover
    # ================================================================
    def headless_chicken_crossover(
        self,
        population: torch.Tensor,
        constants: torch.Tensor,
        crossover_rate: float,
        chicken_rate: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Headless Chicken Crossover. Now preserves constants.
        """
        population = population.clone()
        constants = constants.clone()
        B, L = population.shape
        K = constants.shape[1]
        n_pairs = int(B * 0.5 * crossover_rate)
        if n_pairs == 0:
            return population, constants

        perm = torch.randperm(B, device=self.device)
        p1_idx = perm[:n_pairs * 2:2]
        p2_idx = perm[1:n_pairs * 2:2]

        n_p = p1_idx.shape[0]
        chicken_mask = torch.rand(n_p, device=self.device) < chicken_rate

        parents_1 = population[p1_idx].clone()
        parents_2 = population[p2_idx].clone()
        consts_1 = constants[p1_idx].clone()
        consts_2 = constants[p2_idx].clone()

        if chicken_mask.any():
            n_chicken = int(chicken_mask.sum().item())
            fresh_pop = self.generate_random_population(n_chicken)
            if fresh_pop.shape[1] < L:
                pad = torch.full((n_chicken, L - fresh_pop.shape[1]), PAD_ID,
                                 dtype=self.pop_dtype, device=self.device)
                fresh_pop = torch.cat([fresh_pop, pad], dim=1)
            elif fresh_pop.shape[1] > L:
                fresh_pop = fresh_pop[:, :L]
            # Fresh constants for the random parents
            if GpuGlobals.FORCE_INTEGER_CONSTANTS:
                fresh_consts = torch.randint(
                    GpuGlobals.CONSTANT_INT_MIN_VALUE,
                    GpuGlobals.CONSTANT_INT_MAX_VALUE + 1,
                    (n_chicken, K), device=self.device, dtype=torch.long
                ).to(self.dtype)
            else:
                fresh_consts = torch.empty(n_chicken, K, device=self.device, dtype=self.dtype).uniform_(
                    GpuGlobals.CONSTANT_MIN_VALUE, GpuGlobals.CONSTANT_MAX_VALUE
                )

            parents_2[chicken_mask] = fresh_pop
            consts_2[chicken_mask] = fresh_consts

        # Interleave RPN
        interleaved_rpn = torch.empty((2 * n_p, L), dtype=self.pop_dtype, device=self.device)
        interleaved_rpn[0::2] = parents_1
        interleaved_rpn[1::2] = parents_2
        
        # Interleave Constants
        interleaved_consts = torch.empty((2 * n_p, K), dtype=self.dtype, device=self.device)
        interleaved_consts[0::2] = consts_1
        interleaved_consts[1::2] = consts_2

        crossed_rpn, crossed_consts = self.crossover_population(interleaved_rpn, interleaved_consts, crossover_rate=1.0)

        population[p1_idx] = crossed_rpn[0::2]
        population[p2_idx] = crossed_rpn[1::2]
        constants[p1_idx] = crossed_consts[0::2]
        constants[p2_idx] = crossed_consts[1::2]

        return population, constants

