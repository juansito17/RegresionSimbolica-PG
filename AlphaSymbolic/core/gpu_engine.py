
# DEPRECATED: Use core.gpu instead
from AlphaSymbolic.core.gpu import TensorGeneticEngine

# We map every token to an integer ID.
# 0 is padding/null.
PAD_ID = 0
# 1..N are operators and terminals.

class GPUGrammar:
    def __init__(self, num_variables=1):
        self.token_to_id = {'<PAD>': PAD_ID}
        self.id_to_token = {PAD_ID: '<PAD>'}
        self.next_id = 1
        
        # Terminals (Variables + Constants)
        # Only include variables compliant with num_variables
        self.active_variables = ['x0'] # Always support x0
        if num_variables > 1:
            self.active_variables = [f'x{i}' for i in range(num_variables)]
        elif num_variables == 1:
            self.active_variables = ['x', 'x0'] # Support both for 1D

        self.terminals = self.active_variables + ['C', '1', '2', '3', '5', 'pi', 'e']
        for t in self.terminals:
            self.token_to_id[t] = self.next_id
            self.id_to_token[self.next_id] = t
            self.next_id += 1
            
        # Operators
        # Map operator string to ID
        self.operators = list(OPERATORS.keys())
        for op in self.operators:
            self.token_to_id[op] = self.next_id
            self.id_to_token[self.next_id] = op
            self.next_id += 1
            
        self.vocab_size = self.next_id
        
        # Precompute arithmetic mappings for faster lookup in eval loop
        # We need to know which ID corresponds to which operation type
        self.op_ids = {op: self.token_to_id[op] for op in self.operators}
        self.arity = {self.token_to_id[op]: OPERATORS[op] for op in self.operators}

class TensorGeneticEngine:
    def __init__(self, device: torch.device = None, pop_size=10000, max_len=30, num_variables=1, max_constants=5, n_islands=5):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.grammar = GPUGrammar(num_variables)
        
        # Adjust pop_size to be divisible by n_islands
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

    def optimize_constants(self, population: torch.Tensor, constants: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor, steps=10, lr=0.1):
        """
        Refine constants using Gradient Descent.
        population: [K, L]
        constants: [K, MaxConstants]
        """
        # Clone constants to leaf tensor with grad
        optimized_consts = constants.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([optimized_consts], lr=lr)
        
        best_mse = torch.full((population.shape[0],), float('inf'), device=self.device)
        best_consts = constants.clone().detach() # Fallback
        
        for _ in range(steps):
            optimizer.zero_grad()
            
            # Forward (Differentiable)
            mse, _ = self.evaluate_differentiable(population, optimized_consts, x, y_target)
            
            # Mask NaNs (invalid formulas don't train)
            valid_mask = ~torch.isnan(mse)
            if not valid_mask.any(): break
            
            # Keep best known constants per individual
            improved = (mse < best_mse) & valid_mask
            if improved.any():
                best_mse[improved] = mse[improved].detach()
                best_consts[improved] = optimized_consts[improved].detach()
            
            # Loss = Sum(MSE_valid)
            loss = mse[valid_mask].sum()
            
            if not loss.requires_grad:
                # This happens if no individual uses 'C' (graph disconnected)
                break
                
            loss.backward()
            optimizer.step()
            
        return best_consts, best_mse

    def infix_to_rpn(self, formulas: List[str]) -> torch.Tensor:
        """
        Converts a list of infix strings to a padded RPN tensor [B, L].
        """
        batch_rpn = []
        for f in formulas:
            try:
                # Use shared ExpressionTree to parse infix -> tree -> postfix(ish)
                # But ExpressionTree is prefix. We need Postfix for stack eval.
                # Let's do a simple recursive implementation here or leverage parsed tree.
                tree = ExpressionTree.from_infix(f)
                if not tree.is_valid:
                    batch_rpn.append([PAD_ID]*self.max_len)
                    continue
                
                # Conversion: Tree -> Postfix
                rpn_tokens = []
                def traverse(node):
                    if not node: return
                    for child in node.children:
                        traverse(child)
                    rpn_tokens.append(node.value)
                
                traverse(tree.root)
                
                # Convert to IDs
                ids = [self.grammar.token_to_id.get(t, PAD_ID) for t in rpn_tokens]
                # Pad/Truncate
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

    def rpn_to_infix(self, rpn_tensor: torch.Tensor) -> str:
        """
        Decodes a single RPN tensor row back to infix string.
        """
        ids = rpn_tensor.squeeze().cpu().numpy()
        stack = []
        
        for id in ids:
            if id == PAD_ID: continue
            token = self.grammar.id_to_token.get(id, '?')
            
            if token in OPERATORS:
                arity = OPERATORS[token]
            if token in OPERATORS:
                arity = OPERATORS[token]
                if len(stack) < arity: 
                    # Skip invalid op, just like GPU engine does
                    continue
                
                args = [stack.pop() for _ in range(arity)]
                args.reverse()
                
                # Infix string construction
                if arity == 2:
                    if token == 'pow': elem = f"pow({args[0]}, {args[1]})"
                    else: elem = f"({args[0]} {token} {args[1]})"
                else:
                    elem = f"{token}({args[0]})"
                stack.append(elem)
            else:
                stack.append(token)
                
        if len(stack) >= 1:
            return stack[-1]
        # print(f"DEBUG: RPN Decode Failed. IDs: {ids} Stack: {stack}")
        return "Invalid"

    def evaluate_batch(self, population: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the RPN population on the GPU.
        population: [PopSize, MaxLen] (Integers)
        x: [DataSize] (Floats)
        y_target: [DataSize] (Floats)
        
        Returns: RMSE per individual [PopSize]
        """
        B, L = population.shape
        D = x.shape[0]
        MAX_STACK = 10
        
        # Stack: [B, D, MAX_STACK]
        # We need D because each data point evaluates differently.
        # But wait, the structure is the same. Just the values differ.
        # We can treat B*D as the batch dimension for the stack operations to simplify?
        # PopSize=10k, Data=20 -> 200k items. Easy for GPU.
        
        # Reshape inputs for "Batch of Data Points"
        # Effective Batch Size = B * D
        eff_B = B * D
        
        # Expand population to match data: [B, 1, L] -> [B, D, L] -> [B*D, L]
        pop_expanded = population.unsqueeze(1).expand(-1, D, -1).reshape(eff_B, L)
        
        # Expand x to match population: [D] -> [1, D] -> [B, D] -> [B*D]
        x_expanded = x.unsqueeze(0).expand(B, -1).reshape(eff_B)
        
        # Stack tensor: [EffectiveBatch, StackDepth]
        stack = torch.zeros(eff_B, MAX_STACK, device=self.device, dtype=torch.float32)
        sp = torch.zeros(eff_B, device=self.device, dtype=torch.long) # Stack pointer (next empty slot)
        
        # DEBUG
        # print(f"DEBUG: Eval Batch B={B} L={L} EffB={eff_B}")
        
        # Constants lookup (naive)
        pi_val = torch.tensor(np.pi, device=self.device)
        e_val = torch.tensor(np.e, device=self.device)
        
        # Precompute IDs for speed
        id_x = self.grammar.token_to_id.get('x', -100)
        id_x0 = self.grammar.token_to_id.get('x0', -100)
        id_C = self.grammar.token_to_id.get('C', -100)
        id_pi = self.grammar.token_to_id.get('pi', -100)
        id_e = self.grammar.token_to_id.get('e', -100)
        
        # Binary Ops
        op_add = self.grammar.token_to_id.get('+', -100)
        op_sub = self.grammar.token_to_id.get('-', -100)
        op_mul = self.grammar.token_to_id.get('*', -100)
        op_div = self.grammar.token_to_id.get('/', -100)
        op_pow = self.grammar.token_to_id.get('pow', -100)
        
        # Unary Ops
        # Unary Ops
        op_sin = self.grammar.token_to_id.get('sin', -100)
        op_cos = self.grammar.token_to_id.get('cos', -100)
        op_tan = self.grammar.token_to_id.get('tan', -100)
        
        op_asin = self.grammar.token_to_id.get('asin', -100)
        op_acos = self.grammar.token_to_id.get('acos', -100)
        op_atan = self.grammar.token_to_id.get('atan', -100)
        
        op_exp = self.grammar.token_to_id.get('exp', -100)
        op_log = self.grammar.token_to_id.get('log', -100)
        op_sqrt = self.grammar.token_to_id.get('sqrt', -100)
        op_abs = self.grammar.token_to_id.get('abs', -100)
        op_neg = self.grammar.token_to_id.get('neg', -100)

        # Loop over RPN tokens
        for i in range(L):
            token = pop_expanded[:, i] # [EffectiveBatch]
            
            # Mask: Is this row active? (Not PAD)
            # PAD=0. If PAD, we do nothing (stack remains same)
            active_mask = (token != PAD_ID)
            if not active_mask.any(): continue
            
            # 1. Handle Operands (Push)
            # -------------------------
            # We calculate "value to push" for everyone, then apply.
            push_vals = torch.zeros(eff_B, device=self.device)
            is_operand = torch.zeros(eff_B, dtype=torch.bool, device=self.device)
            
            # x
            mask = (token == id_x) | (token == id_x0)
            if mask.any():
                push_vals[mask] = x_expanded[mask]
                is_operand = is_operand | mask
                
            # Constants
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
                push_vals[mask] = 1.0 # Default C=1.0 for GPU Search (optimization is hard here)
                is_operand = is_operand | mask
                
            # Numeric Literals (1..5)
            # (Assuming ids mapped sequentially or we map individually)
            # Simpler: Check range if mapped sequentially, or just discrete checks
            for val_str in ['1', '2', '3', '5']:
                vid = self.grammar.token_to_id.get(val_str, -999)
                mask = (token == vid)
                if mask.any():
                    push_vals[mask] = float(val_str)
                    is_operand = is_operand | mask

            # Apply Push
            if is_operand.any():
                # stack[b, sp[b]] = val
                # Safe scatter
                safe_sp = torch.clamp(sp, 0, MAX_STACK-1)
                stack.scatter_(1, safe_sp.unsqueeze(1), push_vals.unsqueeze(1))
                # Increment SP
                sp = sp + is_operand.long()


            # 2. Handle Binary Ops (Pop 2, Push 1)
            # ------------------------------------
            is_binary = (token == op_add) | (token == op_sub) | (token == op_mul) | (token == op_div) | (token == op_pow)
            
            if is_binary.any():
                # We need at least 2 items. If sp < 2, it's invalid.
                valid_op = is_binary & (sp >= 2)
                
                if valid_op.any():
                    # Calculate indices safely (clamp to valid range [0, 9] even if invalid row)
                    # We will mask out the result later, so garbage input is fine, but SEGV isn't.
                    safe_sp_minus_1 = torch.clamp(sp - 1, 0, MAX_STACK - 1)
                    safe_sp_minus_2 = torch.clamp(sp - 2, 0, MAX_STACK - 1)
                    
                    # Pop B (Top)
                    idx_b = safe_sp_minus_1.unsqueeze(1)
                    val_b = stack.gather(1, idx_b).squeeze(1)
                    
                    # Pop A (Second)
                    idx_a = safe_sp_minus_2.unsqueeze(1)
                    val_a = stack.gather(1, idx_a).squeeze(1)
                    
                    res = torch.zeros_like(val_a)
                    
                    # Compute
                    mask = (token == op_add) & valid_op
                    if mask.any(): res[mask] = val_a[mask] + val_b[mask]
                    
                    mask = (token == op_sub) & valid_op
                    if mask.any(): res[mask] = val_a[mask] - val_b[mask]
                    
                    mask = (token == op_mul) & valid_op
                    if mask.any(): res[mask] = val_a[mask] * val_b[mask]
                    
                    mask = (token == op_div) & valid_op
                    if mask.any(): 
                        # Protected Div
                        denom = val_b[mask]
                        denom = torch.where(denom.abs() < 1e-6, torch.tensor(1.0, device=self.device), denom)
                        res[mask] = val_a[mask] / denom
                        
                    mask = (token == op_pow) & valid_op
                    if mask.any():
                        # Protected Pow
                        base = val_a[mask].abs() + 1e-6
                        expon = torch.clamp(val_b[mask], -10, 10)
                        res[mask] = torch.pow(base, expon)
                    
                    # Push Result (at pos sp-2)
                    write_val = res
                    # Write pos must be valid too
                    write_pos = torch.clamp(sp - 2, 0, MAX_STACK-1)
                    
                    # Blend: Only update if valid_op
                    current_at_pos = stack.gather(1, write_pos.unsqueeze(1)).squeeze(1)
                    final_write_val = torch.where(valid_op, write_val, current_at_pos)
                    
                    stack.scatter_(1, write_pos.unsqueeze(1), final_write_val.unsqueeze(1))
                    
                    # Decrement SP by 1 (Pop 2, Push 1 = Net -1)
                    sp = sp - valid_op.long()


            # 3. Handle Unary Ops (Pop 1, Push 1)
            # -----------------------------------
            is_unary = (token == op_sin) | (token == op_cos) | (token == op_tan) | \
                       (token == op_asin) | (token == op_acos) | (token == op_atan) | \
                       (token == op_exp) | (token == op_log) | \
                       (token == op_sqrt) | (token == op_abs) | (token == op_neg)
                       
            if is_unary.any():
                valid_op = is_unary & (sp >= 1)
                
                if valid_op.any():
                    # Index safety
                    safe_sp_minus_1 = torch.clamp(sp - 1, 0, MAX_STACK - 1)
                    
                    # Peek Top (at sp-1)
                    idx_a = safe_sp_minus_1.unsqueeze(1)
                    val_a = stack.gather(1, idx_a).squeeze(1)
                    
                    res = torch.zeros_like(val_a)
                    
                    mask = (token == op_sin) & valid_op
                    if mask.any(): res[mask] = torch.sin(val_a[mask])
                    
                    mask = (token == op_cos) & valid_op
                    if mask.any(): res[mask] = torch.cos(val_a[mask])
                    
                    mask = (token == op_tan) & valid_op
                    if mask.any(): res[mask] = torch.tan(val_a[mask])
                    
                    mask = (token == op_asin) & valid_op
                    if mask.any():
                        # Clamp for safety
                        clamped = torch.clamp(val_a[mask], -0.999, 0.999) 
                        res[mask] = torch.asin(clamped)
                        
                    mask = (token == op_acos) & valid_op
                    if mask.any():
                        clamped = torch.clamp(val_a[mask], -0.999, 0.999)
                        res[mask] = torch.acos(clamped)
                        
                    mask = (token == op_atan) & valid_op
                    if mask.any(): res[mask] = torch.atan(val_a[mask])
                    
                    mask = (token == op_exp) & valid_op
                    if mask.any(): res[mask] = torch.exp(torch.clamp(val_a[mask], -20, 20))
                    
                    mask = (token == op_log) & valid_op
                    if mask.any(): res[mask] = torch.log(val_a[mask].abs() + 1e-6)
                    
                    mask = (token == op_sqrt) & valid_op
                    if mask.any(): res[mask] = torch.sqrt(val_a[mask].abs())
                    
                    mask = (token == op_abs) & valid_op
                    if mask.any(): res[mask] = torch.abs(val_a[mask])
                    
                    mask = (token == op_neg) & valid_op
                    if mask.any(): res[mask] = -val_a[mask]
                    
                    # Overwrite Top
                    write_pos = safe_sp_minus_1
                    current_at_pos = stack.gather(1, write_pos.unsqueeze(1)).squeeze(1)
                    final_write_val = torch.where(valid_op, res, current_at_pos)
                    
                    stack.scatter_(1, write_pos.unsqueeze(1), final_write_val.unsqueeze(1))
                    
                    # SP stays same

        
        # End of Loop
        # Result is at stack[0] (if valid)
        # Check validity: sp should be 1
        
        is_valid = (sp == 1)
        
        # Extract result
        # final_preds: [EffectiveBatch]
        final_preds = stack[:, 0]
        
        # For invalid, set to NaN or huge error
        final_preds = torch.where(is_valid, final_preds, torch.tensor(float('nan'), device=self.device))
        
        # Reshape back to [B, D]
        # [eff_B] -> [B, D]
        preds_matrix = final_preds.view(B, D)
        
        # Compute RMSE
        # y_target: [D] -> [1, D] -> [B, D]
        target_matrix = y_target.unsqueeze(0).expand(B, -1)
        
        # MSE: mean over dim 1 (Data points)
        diff = preds_matrix - target_matrix
        mse = torch.mean(diff**2, dim=1) # [B]
        
        # Handle NaNs (invalid formulas)
        mse = torch.where(torch.isnan(mse), torch.tensor(1e9, device=self.device), mse)
        rmse = torch.sqrt(mse)
        
        return rmse

    def evaluate_differentiable(self, population: torch.Tensor, constants: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autograd-compatible evaluation for constant optimization.
        Only run this on a small subset (e.g. Top-K) due to memory cost of tracing.
        
        population: [B, L] (Long)
        constants: [B, MaxConstants] (Float, Requires Grad)
        x: [D]
        y_target: [D]
        
        Returns: (RMSE [B], Predictions [B, D])
        """
        B, L = population.shape
        D = x.shape[0]
        MAX_STACK = 10
        eff_B = B * D
        
        # Reshape inputs
        pop_expanded = population.unsqueeze(1).expand(-1, D, -1).reshape(eff_B, L)
        x_expanded = x.unsqueeze(0).expand(B, -1).reshape(eff_B)
        
        # Expand constants: [B, K] -> [B, D, K] -> [B*D, K]
        constants_expanded = constants.unsqueeze(1).expand(-1, D, -1).reshape(eff_B, -1)
        
        # Initial State (Functional, no in-place)
        stack = torch.zeros(eff_B, MAX_STACK, device=self.device, dtype=torch.float32)
        sp = torch.zeros(eff_B, device=self.device, dtype=torch.long)
        c_ptr = torch.zeros(eff_B, device=self.device, dtype=torch.long) # Pointer to which constant to use next
        
        # Constants lookup
        pi_val = torch.tensor(np.pi, device=self.device)
        e_val = torch.tensor(np.e, device=self.device)
        
        # IDs
        id_x = self.grammar.token_to_id.get('x', -100)
        id_x0 = self.grammar.token_to_id.get('x0', -100)
        id_C = self.grammar.token_to_id.get('C', -100)
        id_pi = self.grammar.token_to_id.get('pi', -100)
        id_e = self.grammar.token_to_id.get('e', -100)
        
        # Binary Ops
        op_add = self.grammar.token_to_id.get('+', -100)
        op_sub = self.grammar.token_to_id.get('-', -100)
        op_mul = self.grammar.token_to_id.get('*', -100)
        op_div = self.grammar.token_to_id.get('/', -100)
        op_pow = self.grammar.token_to_id.get('pow', -100)
        
        # Unary Ops
        op_sin = self.grammar.token_to_id.get('sin', -100)
        op_cos = self.grammar.token_to_id.get('cos', -100)
        op_tan = self.grammar.token_to_id.get('tan', -100)
        op_asin = self.grammar.token_to_id.get('asin', -100)
        op_acos = self.grammar.token_to_id.get('acos', -100)
        op_atan = self.grammar.token_to_id.get('atan', -100)
        op_exp = self.grammar.token_to_id.get('exp', -100)
        op_log = self.grammar.token_to_id.get('log', -100)
        op_sqrt = self.grammar.token_to_id.get('sqrt', -100)
        op_abs = self.grammar.token_to_id.get('abs', -100)
        op_neg = self.grammar.token_to_id.get('neg', -100)
        
        import torch.nn.functional as F

        for i in range(L):
            token = pop_expanded[:, i]
            active_mask = (token != PAD_ID)
            if not active_mask.any(): continue
            
            # --- 1. Push Operands ---
            push_vals = torch.zeros(eff_B, device=self.device)
            is_operand = torch.zeros(eff_B, dtype=torch.bool, device=self.device)
            
            # x
            mask = (token == id_x) | (token == id_x0)
            if mask.any():
                push_vals = torch.where(mask, x_expanded, push_vals)
                is_operand = is_operand | mask
                
            # Learnable Constants 'C'
            mask = (token == id_C)
            if mask.any():
                # Gather from constants buffer using c_ptr
                # safe_ptr = c_ptr.clamp(0, K-1)
                safe_ptr = torch.clamp(c_ptr, 0, constants_expanded.shape[1]-1)
                
                # Gather: constants[batch, ptr]
                # Gather requires [B, 1] index
                val_c = torch.gather(constants_expanded, 1, safe_ptr.unsqueeze(1)).squeeze(1)
                
                push_vals = torch.where(mask, val_c, push_vals)
                is_operand = is_operand | mask
                
                # Update pointer only for those who used C
                c_ptr = c_ptr + mask.long()
                
            # Fixed Constants
            mask = (token == id_pi)
            if mask.any():
                push_vals = torch.where(mask, pi_val, push_vals)
                is_operand = is_operand | mask
                
            mask = (token == id_e)
            if mask.any():
                push_vals = torch.where(mask, e_val, push_vals)
                is_operand = is_operand | mask
                
            # Literals
            for val_str in ['1', '2', '3', '5']:
                vid = self.grammar.token_to_id.get(val_str, -999)
                mask = (token == vid)
                if mask.any():
                    push_vals = torch.where(mask, torch.tensor(float(val_str), device=self.device), push_vals)
                    is_operand = is_operand | mask
            
            # Update Stack (Functional)
            if is_operand.any():
                # One-hot encoding of SP position
                # safe_sp = sp.clamp(0, MAX_STACK - 1)
                # target_mask: [B, MAX_STACK]
                target_mask = F.one_hot(torch.clamp(sp, 0, MAX_STACK-1), num_classes=MAX_STACK).bool()
                
                # Logic: if is_operand, replace stack[sp] with push_val
                # stack_new = stack * (~(is_operand & target_mask)) + push_vals * (is_operand & target_mask)
                # But is_operand is [B], target_mask is [B, 10].
                
                update_mask = target_mask & is_operand.unsqueeze(1) # [B, 10]
                
                # Expand push_vals to [B, 10]
                vals_expanded = push_vals.unsqueeze(1).expand(-1, MAX_STACK)
                
                stack = torch.where(update_mask, vals_expanded, stack)
                sp = sp + is_operand.long()
                
            # --- 2. Binary Ops ---
            is_binary = (token == op_add) | (token == op_sub) | (token == op_mul) | (token == op_div) | (token == op_pow)
            valid_op = is_binary & (sp >= 2)
            
            if valid_op.any():
                sp_1 = torch.clamp(sp - 1, 0, MAX_STACK-1)
                sp_2 = torch.clamp(sp - 2, 0, MAX_STACK-1)
                
                # Gather operands
                idx_b = F.one_hot(sp_1, MAX_STACK).bool()
                val_b = (stack * idx_b).sum(dim=1) # Differentiable gather
                
                idx_a = F.one_hot(sp_2, MAX_STACK).bool()
                val_a = (stack * idx_a).sum(dim=1)
                
                res = torch.zeros_like(val_a)
                
                mask = (token == op_add) & valid_op
                if mask.any(): res = torch.where(mask, val_a + val_b, res)
                
                mask = (token == op_sub) & valid_op
                if mask.any(): res = torch.where(mask, val_a - val_b, res)
                
                mask = (token == op_mul) & valid_op
                if mask.any(): res = torch.where(mask, val_a * val_b, res)
                
                mask = (token == op_div) & valid_op
                if mask.any():
                    denom = torch.where(val_b.abs() < 1e-6, torch.tensor(1.0, device=self.device), val_b)
                    res = torch.where(mask, val_a / denom, res)
                    
                mask = (token == op_pow) & valid_op
                if mask.any():
                    base = val_a.abs() + 1e-6
                    expon = torch.clamp(val_b, -10, 10)
                    res = torch.where(mask, torch.pow(base, expon), res)
                
                # Write back to sp-2
                write_pos = sp_2
                target_mask = F.one_hot(write_pos, MAX_STACK).bool()
                update_mask = target_mask & valid_op.unsqueeze(1)
                vals_expanded = res.unsqueeze(1).expand(-1, MAX_STACK)
                
                stack = torch.where(update_mask, vals_expanded, stack)
                sp = sp - valid_op.long()
                
            # --- 3. Unary Ops ---
            is_unary = (token == op_sin) | (token == op_cos) | (token == op_tan) | \
                       (token == op_asin) | (token == op_acos) | (token == op_atan) | \
                       (token == op_exp) | (token == op_log) | \
                       (token == op_sqrt) | (token == op_abs) | (token == op_neg)
            valid_op = is_unary & (sp >= 1)
            
            if valid_op.any():
                sp_1 = torch.clamp(sp - 1, 0, MAX_STACK-1)
                idx_a = F.one_hot(sp_1, MAX_STACK).bool()
                val_a = (stack * idx_a).sum(dim=1)
                
                res = torch.zeros_like(val_a)
                
                mask = (token == op_sin) & valid_op
                if mask.any(): res = torch.where(mask, torch.sin(val_a), res)
                
                mask = (token == op_cos) & valid_op
                if mask.any(): res = torch.where(mask, torch.cos(val_a), res)
                
                mask = (token == op_tan) & valid_op
                if mask.any(): res = torch.where(mask, torch.tan(val_a), res)
                
                mask = (token == op_asin) & valid_op
                if mask.any():
                    clamped = torch.clamp(val_a, -0.999, 0.999)
                    res = torch.where(mask, torch.asin(clamped), res)
                    
                mask = (token == op_acos) & valid_op
                if mask.any():
                    clamped = torch.clamp(val_a, -0.999, 0.999)
                    res = torch.where(mask, torch.acos(clamped), res)
                    
                mask = (token == op_atan) & valid_op
                if mask.any(): res = torch.where(mask, torch.atan(val_a), res)
                
                mask = (token == op_exp) & valid_op
                if mask.any(): res = torch.where(mask, torch.exp(torch.clamp(val_a, -20, 20)), res)
                
                mask = (token == op_log) & valid_op
                if mask.any(): res = torch.where(mask, torch.log(val_a.abs() + 1e-6), res)
                
                mask = (token == op_sqrt) & valid_op
                if mask.any(): res = torch.where(mask, torch.sqrt(val_a.abs()), res)
                
                mask = (token == op_abs) & valid_op
                if mask.any(): res = torch.where(mask, torch.abs(val_a), res)
                
                mask = (token == op_neg) & valid_op
                if mask.any(): res = torch.where(mask, -val_a, res)
                
                # Write back
                target_mask = F.one_hot(sp_1, MAX_STACK).bool()
                update_mask = target_mask & valid_op.unsqueeze(1)
                vals_expanded = res.unsqueeze(1).expand(-1, MAX_STACK)
                
                stack = torch.where(update_mask, vals_expanded, stack)

        # Final Extract
        is_valid = (sp == 1)
        final_preds = stack[:, 0]
        final_preds = torch.where(is_valid, final_preds, torch.tensor(float('nan'), device=self.device))
        
        preds_matrix = final_preds.view(B, D)
        
        # Loss
        target_matrix = y_target.unsqueeze(0).expand(B, -1)
        mse = torch.mean((preds_matrix - target_matrix)**2, dim=1) # [B]
        
        # Handling NaNs for gradient? 
        # If NaN, we can't backprop. Mask them out.
        # But we mostly optimize valid formulas.
        
        return mse, preds_matrix

    def run(self, x_data: List[float], y_data: List[float], seeds: List[str], timeout_sec=10) -> Optional[str]:
        """
        Main entry point.
        """
    def rpn_to_infix(self, rpn_tensor: torch.Tensor, constants: torch.Tensor = None) -> str:
        """
        Decodes RPN tensor to Infix string (CPU-style formatting).
        """
        vocab = self.grammar.id_to_token
        stack = []
        const_idx = 0
        
        def format_const(val):
            # Match C++ format_constant
            if abs(val - round(val)) < 1e-9:
                return str(int(round(val)))
            if abs(val) >= 1e6 or abs(val) <= 1e-6:
                return f"{val:.8e}"
            s = f"{val:.8f}"
            s = s.rstrip('0').rstrip('.')
            return s if s else "0"

        for token_id in rpn_tensor:
            token_id = token_id.item()
            if token_id == PAD_ID: break
            
            token = vocab.get(token_id, "")
            
            if token in self.grammar.OPERATORS:
                arity = self.grammar.token_arity.get(token, 2)
                if arity == 1:
                    if not stack: return "Invalid"
                    a = stack.pop()
                    if token == 's': stack.append(f"sin({a})")
                    elif token == 'c': stack.append(f"cos({a})")
                    elif token == 'l': stack.append(f"log({a})")
                    elif token == 'e': stack.append(f"exp({a})")
                    elif token == 'q': stack.append(f"sqrt({a})")
                    elif token == 'a': stack.append(f"abs({a})")
                    elif token == 'n': stack.append(f"sign({a})")
                    elif token == '_': stack.append(f"floor({a})")
                    elif token == '!': stack.append(f"({a})!")
                    else: stack.append(f"{token}({a})")
                else: # Binary
                    if len(stack) < 2: return "Invalid"
                    b = stack.pop()
                    a = stack.pop()
                    
                    # Handle A + (-B) -> (A - B)
                    # Handle 0 - B -> (-B)
                    if token == '+' and b.startswith("-") and not b.startswith("(-"):
                         stack.append(f"({a} - {b[1:]})")
                    elif token == '-' and a == "0":
                         stack.append(f"(-{b})")
                    else:
                         stack.append(f"({a} {token} {b})")
            elif token == 'C':
                val = 1.0
                if constants is not None and const_idx < len(constants):
                    val = constants[const_idx].item()
                    const_idx += 1
                stack.append(format_const(val))
            elif token.startswith('x'):
                # Handle x0, x1
                # If token is just 'x', assume x0
                if token == 'x': stack.append("x0")
                else: stack.append(token)
            else:
                stack.append(str(token))
                
        if len(stack) == 1:
            return stack[0]
        return "Invalid"


    def run(self, x_values: List[float], y_targets: List[float], seeds: List[str], timeout_sec=10, callback=None) -> Optional[str]:
        """
        Main evolutionary loop on GPU.
        callback: function(gen, best_mse, best_rpn, best_consts, is_new_best) -> None
        """
        start_time = time.time()
        
        # 1. Setup Data
        x_t = torch.tensor(x_values, device=self.device, dtype=torch.float32)
        y_t = torch.tensor(y_targets, device=self.device, dtype=torch.float32)
        
        if x_t.ndim > 1: x_t = x_t.flatten() 
        if y_t.ndim > 1: y_t = y_t.flatten()

        # print(f"[GPU Worker] Initializing Tensor Population ({self.pop_size})...")
        
        # --- 0. Target Pattern Detection ("The Sniper") ---
        # Swiftly check for trivial Linear or Geometric patterns
        if x_t.shape[0] > 2:
            try:
                # Prepare X matrix [N, 2] for (slope, intercept)
                X_mat = torch.stack([x_t, torch.ones_like(x_t)], dim=1)
                
                # A. Linear Check (y = mx + c)
                # Solve: X * [m, c] = y
                try:
                    solution = torch.linalg.lstsq(X_mat, y_t).solution
                    m, c = solution[0].item(), solution[1].item()
                    y_pred = m * x_t + c
                    
                    # Check residuals (Relative Error or R2?)
                    # Use Normalized RMSE
                    res_std = torch.std(y_t - y_pred)
                    y_std = torch.std(y_t)
                    if y_std > 1e-9 and (res_std / y_std) < 1e-4:
                        # Found Linear!
                        # print(f"[GPU Sniper] Detected Linear Pattern: {m:.4f}*x + {c:.4f}")
                        if abs(c) < 1e-5: return f"({m:.4f} * x)"
                        return f"(({m:.4f} * x) + {c:.4f})"
                except: pass

                # B. Geometric Check (y = A * e^(Bx) -> log(y) = log(A) + Bx)
                if torch.all(y_t > 0):
                    log_y = torch.log(y_t)
                    solution_g = torch.linalg.lstsq(X_mat, log_y).solution
                    B, log_A = solution_g[0].item(), solution_g[1].item()
                    
                    y_pred_log = B * x_t + log_A
                    res_std_log = torch.std(log_y - y_pred_log)
                    
                    if res_std_log < 1e-4:
                        # Found Geometric!
                        # Formula: exp(log_A + Bx)
                        # print(f"[GPU Sniper] Detected Geometric Pattern.")
                        return f"exp(({B:.4f} * x) + {log_A:.4f})"
            except Exception as e:
                pass

        
        # 2. Initialize Population & Constants
        seed_tensor = self.infix_to_rpn(seeds)
        num_seeds = seed_tensor.shape[0]
        
        population = torch.zeros(self.pop_size, self.max_len, device=self.device, dtype=torch.long)
        pop_constants = torch.randn(self.pop_size, self.max_constants, device=self.device) # Learnable Constants
        
        # Fill seeds
        population[:num_seeds] = seed_tensor
        
        # Fill rest
        if num_seeds > 0:
            remaining = self.pop_size - num_seeds
            src_indices = torch.randint(0, num_seeds, (remaining,), device=self.device)
            population[num_seeds:] = seed_tensor[src_indices]
            
            mutation_mask = torch.rand(remaining, self.max_len, device=self.device) < 0.3
            random_tokens = torch.randint(1, self.grammar.vocab_size, (remaining, self.max_len), device=self.device)
            population[num_seeds:] = torch.where(mutation_mask, random_tokens, population[num_seeds:])
        else:
             print("[GPU Worker] No seeds provided.")
             return None

        best_formula_str = None
        best_rmse = float('inf')
        
        generations = 0
        COMPLEXITY_PENALTY = 0.01
        
        # --- Dynamic Adaptation ("The Thermostat") ---
        stagnation_counter = 0
        current_mutation_rate = 0.15  # Base rate
        current_chaos_rate = 0.01     # Base chaos
        last_improvement_gen = 0

        
        while time.time() - start_time < timeout_sec:
            generations += 1
            
            # A. Evaluate (Standard)
            fitness_rmse = self.evaluate_batch(population, x_t, y_t)
            
            # Calculate Complexity (Length)
            # Penalize longer formulas to encourage simplicity (Occam's Razor)
            lengths = (population != PAD_ID).sum(dim=1).float()
            # fitness = rmse * (1 + penalty * length) + length * epsilon (for 0-rmse ties)
            fitness_penalized = fitness_rmse * (1.0 + COMPLEXITY_PENALTY * lengths) + lengths * 1e-6
            
            # B. Constant Optimization (Elitism)
            # Pick Top 50 candidates based on PENALIZED fitness to refine
            k_opt = 50
            top_vals, top_indices = torch.topk(fitness_penalized, k_opt, largest=False)
            
            # Run Gradient Descent on these constants
            refined_consts, refined_mse = self.optimize_constants(
                population[top_indices], 
                pop_constants[top_indices], 
                x_t, y_t, steps=10, lr=0.1
            )
            
            # Write back results
            pop_constants[top_indices] = refined_consts.detach()
            fitness_rmse[top_indices] = refined_mse.detach()
            
            # Re-calculate penalized fitness for optimized ones
            refined_lengths = lengths[top_indices]
            fitness_penalized[top_indices] = refined_mse.detach() * (1.0 + COMPLEXITY_PENALTY * refined_lengths) + refined_lengths * 1e-6
            
            # --- Algebraic Simplification (The Cleaner) ---
            # Every 5 generations, simplify the top elites to remove clutter (x*1 -> x)
            if generations % 5 == 0:
                try:
                    import sympy
                    # Simplify the optimization candidates (which are already best)
                    # We operate in-place on the population
                    for idx_in_top in range(len(top_indices)):
                        pop_idx = top_indices[idx_in_top]
                        
                        # 1. Decode to string (with optimized constants)
                        rpn = population[pop_idx].unsqueeze(0)
                        consts = pop_constants[pop_idx]
                        expr_str = self.rpn_to_infix(rpn, consts)
                        
                        if expr_str == "Invalid": continue
                        
                        # 2. Simplify with SymPy
                        try:
                            # Parse and simplify
                            sym_expr = sympy.sympify(expr_str)
                            simplified_sym = sympy.simplify(sym_expr)
                            
                            # 3. Re-encode to RPN + Constants
                            new_rpn_ids, new_consts_vals = self.sympy_to_rpn(simplified_sym)
                            
                            # Update if valid and fits
                            if len(new_rpn_ids) <= self.max_len:
                                # Overwrite population
                                population[pop_idx] = torch.tensor(new_rpn_ids + [PAD_ID]*(self.max_len - len(new_rpn_ids)), device=self.device)
                                
                                # Overwrite constants
                                new_c_tensor = torch.zeros(self.max_constants, device=self.device)
                                num_c = min(len(new_consts_vals), self.max_constants)
                                if num_c > 0:
                                    new_c_tensor[:num_c] = torch.tensor(new_consts_vals[:num_c], device=self.device)
                                pop_constants[pop_idx] = new_c_tensor
                                
                                # Note: Fitness needs update? 
                                # Simplification should preserve semantics, so RMSE is same. 
                                # But length might decrease, so fitness improves!
                                # Let's re-eval next generation or now?
                                # For safety, we leave it. It will be re-evaluated next gen or by selection if we updated lengths.
                                lengths[pop_idx] = len(new_rpn_ids) # Approximate update
                                
                        except Exception as e:
                            # print(f"Simplification failed for {expr_str}: {e}")
                            pass
                except ImportError:
                    pass

            # Check Best (based on Raw RMSE, but maybe Length matters for user? stick to RMSE)
            min_rmse, min_idx = torch.min(fitness_rmse, dim=0)
            if min_rmse.item() < best_rmse:
                best_rmse = min_rmse.item()
                best_rpn = population[min_idx].unsqueeze(0)
                best_consts_vec = pop_constants[min_idx]
                best_formula_str = self.rpn_to_infix(best_rpn, best_consts_vec)
                # print(f"[GPU Worker] New Best: {best_formula_str} (RMSE: {best_rmse:.5f})")
                
                if callback:
                    callback(generations, best_rmse, best_rpn, best_consts_vec, True)
                
                # Reset Stagnation
                stagnation_counter = 0
                current_mutation_rate = 0.15
                current_chaos_rate = 0.01
                last_improvement_gen = generations
            else:
                stagnation_counter += 1
                
            if callback and (generations % 100 == 0 or generations == 1) and best_rpn is not None:
                 # Pass current global best
                 callback(generations, best_rmse, best_rpn, best_consts_vec, False) # False = not new best, just update
                 
            # Adaptation Logic


                
            # Adaptation Logic
            if stagnation_counter > 20:
                # Boost Mutation/Chaos incrementally
                current_mutation_rate = min(0.40, current_mutation_rate + 0.02)
                current_chaos_rate = min(0.05, current_chaos_rate + 0.005)
                
            # --- Island Cataclysm (Nuclear Reset) ---
            if stagnation_counter >= 50:
                 # print(f"[GPU Worker] CATACLYSM! Global Stagnation {stagnation_counter}. Resetting population.")
                 # Keep Top 1 (min_idx)
                 # We need to construct a new population where index 0 is best, rest random.
                 
                 # 1. Save Best
                 saved_best_rpn = population[min_idx].clone()
                 saved_best_c = pop_constants[min_idx].clone()
                 
                 # 2. Randomize All
                 population = torch.randint(1, self.grammar.vocab_size, (self.pop_size, self.max_len), device=self.device)
                 pop_constants = torch.randn(self.pop_size, self.max_constants, device=self.device)
                 
                 # 3. Restore Best at 0
                 population[0] = saved_best_rpn
                 pop_constants[0] = saved_best_c
                 
                 # 4. Reset Stats
                 stagnation_counter = 0
                 current_mutation_rate = 0.15
                 current_chaos_rate = 0.01
                 
                 # Force re-eval? Next loop will evaluate.


            
            # C. Island Selection & Tournament (Vectorized)
            # 1. Reshape to [NumIslands, IslandSize]
            view_fit = fitness_penalized.view(self.n_islands, self.island_size)
            view_pop = population.view(self.n_islands, self.island_size, self.max_len)
            view_const = pop_constants.view(self.n_islands, self.island_size, self.max_constants)

            # 2. Elitism per Island
            k_elite_island = max(1, int(self.island_size * 0.1))
            # topk returns indices relative to the island
            elite_vals, elite_local_idx = torch.topk(view_fit, k_elite_island, dim=1, largest=False)

            # Gather Elites
            # Expansion for gather: [Islands, K, L]
            gather_idx_pop = elite_local_idx.unsqueeze(-1).expand(-1, -1, self.max_len)
            elites_pop = torch.gather(view_pop, 1, gather_idx_pop)
            
            gather_idx_c = elite_local_idx.unsqueeze(-1).expand(-1, -1, self.max_constants)
            elites_c = torch.gather(view_const, 1, gather_idx_c)

            # 3. Tournament for Offspring
            num_offspring = self.island_size - k_elite_island
            
            # Generate random pairs of indices [Islands, NumOffspring]
            p1_idx = torch.randint(0, self.island_size, (self.n_islands, num_offspring), device=self.device)
            p2_idx = torch.randint(0, self.island_size, (self.n_islands, num_offspring), device=self.device)
            
            # Compare fitness
            f1 = torch.gather(view_fit, 1, p1_idx)
            f2 = torch.gather(view_fit, 1, p2_idx)
            
            winner_idx = torch.where(f1 < f2, p1_idx, p2_idx)
            
            # Gather Winners
            gather_idx_win_pop = winner_idx.unsqueeze(-1).expand(-1, -1, self.max_len)
            winners_pop = torch.gather(view_pop, 1, gather_idx_win_pop)
            
            gather_idx_win_c = winner_idx.unsqueeze(-1).expand(-1, -1, self.max_constants)
            winners_c = torch.gather(view_const, 1, gather_idx_win_c)
            
            # 4. Migration (Every 10 gens, Ring Topology)
            # Inject neighbor's elites into the worst slots of current offspring
            if generations % 10 == 0 and self.n_islands > 1:
                # Rotate elites: Island i gets elites from i-1 (or i+1 if we roll pos)
                migrants_pop = torch.roll(elites_pop, shifts=1, dims=0)
                migrants_c = torch.roll(elites_c, shifts=1, dims=0)
                
                # Replace last k_elite spots in WINNERS (weakest offspring? actually tournament winners are random quality)
                # But acceptable to just replace.
                if num_offspring >= k_elite_island:
                    winners_pop[:, -k_elite_island:] = migrants_pop
                    winners_c[:, -k_elite_island:] = migrants_c
            
            # D. Mutation (On Offspring Only)
            
            # 1. Safe Arity-Preserving Mutation (Dynamic Rate)
            mask = torch.rand(winners_pop.shape, device=self.device) < current_mutation_rate
            current_arities = self.token_arity[winners_pop]
            
            # Arity 0 -> Arity 0
            if len(self.arity_0_ids) > 0:
                noise_0 = self.arity_0_ids[torch.randint(0, len(self.arity_0_ids), winners_pop.shape, device=self.device)]
                winners_pop = torch.where(mask & (current_arities == 0), noise_0, winners_pop)
                
            # Arity 1 -> Arity 1
            if len(self.arity_1_ids) > 0:
                noise_1 = self.arity_1_ids[torch.randint(0, len(self.arity_1_ids), winners_pop.shape, device=self.device)]
                winners_pop = torch.where(mask & (current_arities == 1), noise_1, winners_pop)
                
            # Arity 2 -> Arity 2
            if len(self.arity_2_ids) > 0:
                noise_2 = self.arity_2_ids[torch.randint(0, len(self.arity_2_ids), winners_pop.shape, device=self.device)]
                winners_pop = torch.where(mask & (current_arities == 2), noise_2, winners_pop)
                
            # 2. Chaos Mutation (Structure changing, Low Rate: 1%)
            chaos_mask = torch.rand(winners_pop.shape, device=self.device) < current_chaos_rate
            chaos_noise = torch.randint(1, self.grammar.vocab_size, winners_pop.shape, device=self.device)
            winners_pop = torch.where(chaos_mask, chaos_noise, winners_pop)
            
            # Constant Mutation
            c_noise = torch.randn_like(winners_c) * 0.1
            winners_c = winners_c + c_noise
            
            # 5. Reconstruct Population
            # Concat [Elites, Offspring] -> [Islands, Size, L]
            next_pop_view = torch.cat([elites_pop, winners_pop], dim=1)
            next_c_view = torch.cat([elites_c, winners_c], dim=1)
            
            # Flatten to [PopSize, L]
            population = next_pop_view.view(self.pop_size, self.max_len)
            pop_constants = next_c_view.view(self.pop_size, self.max_constants)
            
        print(f"[GPU Worker] Finished. Gens: {generations}. Best RMSE: {best_rmse:.5f}")
        return best_formula_str

    def sympy_to_rpn(self, sym_expr) -> Tuple[List[int], List[float]]:
        """
        Converts a SymPy expression to RPN token IDs and a list of constants.
        """
        import sympy
        
        rpn_ids = []
        constants = []
        
        def visit(node):
            if node.is_Number:
                val = float(node)
                # Check for simple constants
                if node == sympy.pi:
                    rpn_ids.append(self.grammar.token_to_id['pi'])
                elif node == sympy.E:
                    rpn_ids.append(self.grammar.token_to_id['e'])
                elif val == 1.0 and '1' in self.grammar.token_to_id:
                     rpn_ids.append(self.grammar.token_to_id['1'])
                elif val == 2.0 and '2' in self.grammar.token_to_id:
                     rpn_ids.append(self.grammar.token_to_id['2'])
                elif val == 3.0 and '3' in self.grammar.token_to_id:
                     rpn_ids.append(self.grammar.token_to_id['3'])
                elif val == 5.0 and '5' in self.grammar.token_to_id:
                     rpn_ids.append(self.grammar.token_to_id['5'])
                else:
                    # Generic Constant -> C
                    rpn_ids.append(self.grammar.token_to_id['C'])
                    constants.append(val)
            elif node.is_Symbol:
                name = str(node)
                if name in self.grammar.token_to_id:
                    rpn_ids.append(self.grammar.token_to_id[name])
                else:
                    # Variable mismatch or unknown
                    rpn_ids.append(self.grammar.token_to_id.get('x', 0)) # Fallback
            elif isinstance(node, sympy.Add):
                # Sympy Add is n-ary. Convert to chain of binary adds.
                # A + B + C -> A B + C +
                args = node.args
                visit(args[0])
                for i in range(1, len(args)):
                    visit(args[i])
                    rpn_ids.append(self.grammar.token_to_id['+'])
            elif isinstance(node, sympy.Mul):
                # A * B * C -> A B * C *
                args = node.args
                visit(args[0])
                for i in range(1, len(args)):
                    visit(args[i])
                    rpn_ids.append(self.grammar.token_to_id['*'])
            elif isinstance(node, sympy.Pow):
                visit(node.base)
                visit(node.exp)
                rpn_ids.append(self.grammar.token_to_id['pow'])
            elif isinstance(node, sympy.sin):
                visit(node.args[0])
                rpn_ids.append(self.grammar.token_to_id['sin'])
            elif isinstance(node, sympy.cos):
                visit(node.args[0])
                rpn_ids.append(self.grammar.token_to_id['cos'])
            elif isinstance(node, sympy.tan):
                visit(node.args[0])
                rpn_ids.append(self.grammar.token_to_id['tan'])
            elif isinstance(node, sympy.exp):
                visit(node.args[0])
                rpn_ids.append(self.grammar.token_to_id['exp'])
            elif isinstance(node, sympy.log):
                visit(node.args[0])
                rpn_ids.append(self.grammar.token_to_id['log'])
            # Add other functions as needed (asin, acos, etc.)
            else:
                 # Fallback for unknown
                 # Check if it is a known function by string
                 func_name = str(node.func)
                 if func_name in self.grammar.token_to_id:
                      visit(node.args[0]) # Assumes unary
                      rpn_ids.append(self.grammar.token_to_id[func_name])
                 else:
                     # raise ValueError(f"Unknown node: {node}")
                     # Fallback to ignore? Or try to approximate?
                     # If we raise, the simplification block catches it and aborts.
                     raise ValueError(f"Unknown node: {node}")
        
        visit(sym_expr)
        return rpn_ids, constants
