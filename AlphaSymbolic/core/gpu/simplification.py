
import torch
from typing import Tuple
from .config import GpuGlobals

# SymPy for simplification
try:
    import sympy
    from sympy import symbols, sympify, simplify, nsimplify, Float
    from sympy.parsing.sympy_parser import parse_expr
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    
from .grammar import PAD_ID, GPUGrammar

class GPUSimplifier:
    def __init__(self, grammar: GPUGrammar, device, max_constants=5):
        self.grammar = grammar
        self.device = device
        self.max_constants = max_constants
        self.num_variables = len(self.grammar.active_variables)

    def simplify_expression(self, rpn_tensor: torch.Tensor, constants: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        if not SYMPY_AVAILABLE or not GpuGlobals.USE_SIMPLIFICATION:
            return rpn_tensor, constants, False
        
        try:
            # 1. RPN to Infix
            infix = self._rpn_to_infix_str(rpn_tensor, constants)
            if infix == "Invalid" or not infix:
                return rpn_tensor, constants, False
            
            # 2. SymPy Parse
            sym_vars = {v: symbols(v) for v in self.grammar.active_variables}
            # Alias x -> x0 if needed
            if 'x0' in sym_vars: sym_vars['x'] = sym_vars['x0']
            
            expr_str = infix.replace('^', '**').replace('lgamma', 'loggamma')
            
            try:
                expr = parse_expr(expr_str, local_dict=sym_vars)
            except:
                expr = sympify(expr_str, locals=sym_vars)
            
            # 3. Simplify
            simplified = simplify(expr)
            simplified = nsimplify(simplified, tolerance=1e-6, rational=True)
            
            simplified_str = str(simplified)
            if len(simplified_str) > len(infix) * 1.5:
                 return rpn_tensor, constants, False
                 
            simplified_str = simplified_str.replace('**', ' ^ ').replace('loggamma', 'lgamma')
            
            # 4. Infix to RPN
            # We need an infix to rpn converter. 
            # Ideally simplified strings are cleaner.
            # We can use the logic from Operators/Engine, but here we might need to rely on a utility.
            # For now, let's assume we can import or use a local helper.
            # Engine had `infix_to_rpn_tensor`.
            # Let's import from a shared place or reimplement minimal.
            # Importing ExpressionTree is best.
            from core.grammar import ExpressionTree
            tree = ExpressionTree.from_infix(simplified_str)
            if not tree.is_valid: return rpn_tensor, constants, False
            
            # Extract RPN
            rpn_tokens = []
            def traverse(node):
                if not node: return
                for child in node.children: traverse(child)
                rpn_tokens.append(node.value)
            traverse(tree.root)
            
            # Extract constants
            clean_tokens = []
            new_const_vals = []
            
            for t in rpn_tokens:
                 if t in self.grammar.terminals and t not in ['C', '1', '2', '3', '5'] and not t.startswith('x'):
                     clean_tokens.append(t)
                 elif (t.replace('.','',1).isdigit() or (t.startswith('-') and t[1:].replace('.','',1).isdigit())):
                     if t in ['1', '2', '3', '5']:
                         clean_tokens.append(t)
                     else:
                         clean_tokens.append('C')
                         new_const_vals.append(float(t))
                 else:
                     clean_tokens.append(t)
            
            ids = [self.grammar.token_to_id.get(t, PAD_ID) for t in clean_tokens]
            max_len = rpn_tensor.shape[0]
            if len(ids) > max_len: ids = ids[:max_len]
            else: ids += [PAD_ID] * (max_len - len(ids))
            
            if len(new_const_vals) > self.max_constants: new_const_vals = new_const_vals[:self.max_constants]
            else: new_const_vals += [0.0] * (self.max_constants - len(new_const_vals))
            
            return torch.tensor(ids, dtype=torch.long, device=self.device), torch.tensor(new_const_vals, dtype=torch.float64, device=self.device), True

        except Exception as e:
            return rpn_tensor, constants, False

    def simplify_population(self, population: torch.Tensor, constants: torch.Tensor, top_k: int = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
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

    def _rpn_to_infix_str(self, rpn_tensor: torch.Tensor, constants: torch.Tensor) -> str:
        # Re-implement simple decoder or use shared?
        # Let's copy the logic from Engine since it's formatting.
        # Or better, move formatting to `core/gpu/formatting.py`? 
        # engine.py imported `from .formatting import format_const`
        # But `rpn_to_infix` was in Engine.
        return self.rpn_to_infix_static(rpn_tensor, constants, self.grammar)

    @staticmethod
    def rpn_to_infix_static(rpn_tensor, constants, grammar) -> str:
        # Inline logic from engine
        from .formatting import format_const
        
        if rpn_tensor.ndim > 1: rpn_tensor = rpn_tensor.view(-1)
        vocab = grammar.id_to_token
        stack = []
        const_idx = 0
        
        for token_id in rpn_tensor:
            token_id = token_id.item()
            if token_id == PAD_ID: continue
            
            token = vocab.get(token_id, "")
            
            if token in grammar.operators:
                arity = grammar.token_arity.get(token, 2)
                if arity == 1:
                    if not stack: return "Invalid"
                    a = stack.pop()
                    if token == 's' or token == 'sin': stack.append(f"sin({a})")
                    elif token == 'c' or token == 'cos': stack.append(f"cos({a})")
                    elif token == 'l' or token == 'log': stack.append(f"log({a})")
                    elif token == 'e' or token == 'exp': stack.append(f"exp({a})")
                    elif token == 'q' or token == 'sqrt': stack.append(f"sqrt({a})")
                    elif token == 'a' or token == 'abs': stack.append(f"abs({a})")
                    elif token == 'neg': stack.append(f"neg({a})")
                    elif token == '_' or token == 'floor': stack.append(f"floor({a})")
                    elif token == '!' or token == 'gamma': stack.append(f"gamma({a})")
                    elif token == 'g' or token == 'lgamma': stack.append(f"lgamma({a})")
                    elif token == 'S' or token == 'asin': stack.append(f"asin({a})")
                    elif token == 'C' or token == 'acos': stack.append(f"acos({a})")
                    elif token == 'T' or token == 'atan': stack.append(f"atan({a})")
                    else: stack.append(f"{token}({a})")
                else: 
                    if len(stack) < 2: return "Invalid"
                    b = stack.pop()
                    a = stack.pop()
                    
                    if token == '+' and b.startswith("-") and not b.startswith("(-"):
                         stack.append(f"({a} - {b[1:]})")
                    elif token == '-' and a == "0":
                         stack.append(f"(-{b})")
                    elif token == 'pow':
                         stack.append(f"({a} ^ {b})")
                    elif token == '%':
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
