
import torch
from typing import List, Tuple, Dict, Optional
from core.grammar import OPERATORS

# --- GPU GRAMMAR ENCODING (RPN / Postfix) ---
PAD_ID = 0

class GPUGrammar:
    def __init__(self, num_variables=1, use_globals=True):
        self.token_to_id = {'<PAD>': PAD_ID}
        self.id_to_token = {PAD_ID: '<PAD>'}
        self.next_id = 1
        
        # Terminals (Variables + Constants)
        self.active_variables = ['x0'] # Always support x0
        if num_variables > 1:
            self.active_variables = [f'x{i}' for i in range(num_variables)]
        elif num_variables == 1:
            self.active_variables = ['x', 'x0'] 

        self.terminals = self.active_variables + ['C', '0', '1', '2', '3', '5', '10', 'pi'] # Sync with core.grammar.CONSTANTS
        for t in self.terminals:
            self.token_to_id[t] = self.next_id
            self.id_to_token[self.next_id] = t
            self.next_id += 1
            
        # Operators
        self.operators = []
        
        # Ideally we'd pass config in, but for now we can import GpuGlobals locally or rely on caller?
        # To avoid circular imports, let's try to be data driven or import strictly what's needed.
        # Original code imported GpuGlobals.
        try:
            from .config import GpuGlobals
            use_ops = use_globals
        except ImportError:
            use_ops = False
            # Default fallback if can't import config?
            GpuGlobals = None

        if use_ops and GpuGlobals:
            if GpuGlobals.USE_OP_PLUS:  self.operators.append('+')
            if GpuGlobals.USE_OP_MINUS: self.operators.append('-')
            if GpuGlobals.USE_OP_MULT:  self.operators.append('*')
            if GpuGlobals.USE_OP_DIV:   self.operators.append('/')
            if GpuGlobals.USE_OP_POW:   
                self.operators.append('pow')
                self.operators.append('^')
            if GpuGlobals.USE_OP_MOD:   
                self.operators.append('%')
                self.operators.append('mod')
            if GpuGlobals.USE_OP_SIN:   self.operators.append('sin')
            if GpuGlobals.USE_OP_COS:   self.operators.append('cos')
            if GpuGlobals.USE_OP_LOG:   self.operators.append('log')
            if GpuGlobals.USE_OP_EXP:   self.operators.append('exp')
            if GpuGlobals.USE_OP_FACT:  self.operators.append('fact') 
            if GpuGlobals.USE_OP_GAMMA: 
                self.operators.append('gamma')
                self.operators.append('lgamma')
            if GpuGlobals.USE_OP_ASIN:  self.operators.append('asin')
            if GpuGlobals.USE_OP_ACOS:  self.operators.append('acos')
            if GpuGlobals.USE_OP_ATAN:  self.operators.append('atan')
            if GpuGlobals.USE_OP_FLOOR: self.operators.append('floor')
            if GpuGlobals.USE_OP_CEIL:  self.operators.append('ceil')
            if GpuGlobals.USE_OP_SIGN:  self.operators.append('sign')
        else:
            # Fallback set
            self.operators = ['+', '-', '*', '/', 'pow', 'sin', 'cos', 'log', 'exp']

        # Always active standard ops
        self.operators.append('sqrt')
        self.operators.append('abs')
        self.operators.append('neg')
        if 'floor' not in self.operators: self.operators.append('floor')
        if 'mod' not in self.operators: self.operators.append('mod')
        if 'pow' not in self.operators: self.operators.append('pow')

        for op in self.operators:
            # check uniqueness (C vs cos collision? 'C' is const, 'C' op is Acos?)
            # Wait, 'C' is CONSTANT token. 'C' is also ACOS in some mappings?
            # In original code: 
            # if GpuGlobals.USE_OP_ACOS:  self.operators.append('C')
            # Yes, 'C' is added to operators.
            # But 'C' is also in terminals!
            # If so, token_to_id will OVERWRITE or Duplicates?
            # self.terminals added first.
            # If 'C' is added again, it gets a NEW ID.
            # So id 'C' (const) != id 'C' (acos).
            # But converting string to ID might be ambiguous if we just use dict lookup.
            # We must distinguish context or use different key. 
            # But self.token_to_id is a flat dict.
            # Original code HAD this issue potentially? 
            # Lines 38: self.terminals = ... ['C', ...]
            # Lines 60: self.operators.append('C')
            # So 'C' key maps to later ID (Operator). 
            # Constant 'C' becomes unreachable via string 'C' lookup?
            # The parser logic:
            # Line 963: clean_tokens.append('C') -> This uses 'C' lookup.
            # If 'C' maps to acos, valid RPNs with constants get acos!
            # This seems like a BUG in original code unless 'C' op char is different.
            # In C++, acos is likely mapped to something else or context aware?
            # Let's check original code carefully.
            # Line 38: self.terminals = ... + ['C', ...]
            # Line 60: self.operators.append('C')
            # Yes, overwrites.
            # FIX: Mapping 'acos' to 'ACOS' or 'acos' token, and display char 'C'.
            # Or token_to_id should keep 'C' for constant and 'acos' for operator.
            # But infix_to_rpn uses 'C' for constant.
            # I will preserve original behavior but maybe it relies on 'C' (acos) being rarely text-parsed as 'C'?
            # Actually, `operators.append('C')` adds 'C' to list.
            # Then loop `for op in self.operators: self.token_to_id[op] = ...`
            # YES, it overwrites.
            # I should PROBABLY fix this or keep it if "it works" but it looks suspicious.
            # Wait, if infix parser produces 'C' for constant, and token_to_id['C'] is acos, 
            # then simple constants become acos operators in RPN!
            # That would be disastrous.
            # Maybe USE_OP_ACOS is False by default?
            # Assuming I should keep copy-paste but maybe warn or simple fix: use 'acos' as key.
            # Re-reading original encoding:
            # The RPN vm code has: `op_acos = self.grammar.token_to_id.get('C', -100)`
            # So it EXPECTS 'C' to be the ID for acos.
            # Constant logic: `id_C = self.grammar.token_to_id.get('C', -100)`
            # They lookup the SAME ID.
            # So Constant == Acos?
            # In `_run_vm`:
            # `mask = (token == id_C)` -> pushes constant values.
            # `op_acos = ... get('C')`
            # `mask = (token == op_acos) & valid_op` -> executes acos.
            # A constant (arity 0) pushing values... will it trigger valid_op (arity 1)?
            # valid_op checks arity/stack.
            # A constant `id_C` is likely in `arity_0_ids`?
            # `self.token_arity` is built from `self.grammar.operators`.
            # If 'C' is in operators, it has arity 1.
            # So `id_C` has arity 1.
            # When generated as constant, it pushes value... but mutation might treat it as arity 1.
            # Execution `_run_vm`:
            # `if (token == id_C): push val`
            # later `if (token == op_acos) ...`
            # Both trigger!
            # So it pushes a constant, THEN computes acos of stack top?
            # That seems very broken if ACOS is enabled.
            # I will assume this is a known dangerous collision and maybe I should map acos to 'arcc' or something internally?
            # For now, faithfully refactoring.
            pass

        for op in self.operators:
            self.token_to_id[op] = self.next_id
            self.id_to_token[self.next_id] = op
            self.next_id += 1
            
        self.vocab_size = self.next_id
        
        self.op_ids = {op: self.token_to_id[op] for op in self.operators}
        self.token_arity = {}
        for op in self.operators:
            tid = self.token_to_id[op]
            self.token_arity[op] = OPERATORS.get(op, 1) # Default 1 if missing in OPERATORS dict?

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
