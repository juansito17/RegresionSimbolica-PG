import numpy as np
from scipy.special import gamma as scipy_gamma, gammaln
import math

# Supported operators and their arity (number of arguments)
# Organized by curriculum stage for progressive unlocking
OPERATORS = {
    # === STAGE 0: Pure Arithmetic ===
    '+': 2,
    '-': 2,
    '*': 2,
    '/': 2,
    
    # === STAGE 1: Powers ===
    'pow': 2,
    'sqrt': 1,
    
    # === STAGE 2: Trigonometry ===
    'sin': 1,
    'cos': 1,
    'tan': 1,
    
    # === STAGE 3: Transcendental ===
    'exp': 1,
    'log': 1,
    
    # === STAGE 4: Advanced ===
    'abs': 1,
    'neg': 1,
    'sign': 1,
    'floor': 1,
    'ceil': 1,
    'mod': 2,
    'gamma': 1,
    'lgamma': 1,  # Log-gamma function (from C++ GP engine)
}

# Operator groups for curriculum control
OPERATOR_STAGES = {
    0: ['+', '-', '*', '/'],
    1: ['+', '-', '*', '/', 'pow', 'sqrt'],
    2: ['+', '-', '*', '/', 'pow', 'sqrt', 'sin', 'cos', 'tan'],
    3: ['+', '-', '*', '/', 'pow', 'sqrt', 'sin', 'cos', 'tan', 'exp', 'log'],
    4: list(OPERATORS.keys()),  # All operators
}

# Terminal tokens
VARIABLES = ['x']
# 'C' is a placeholder for learnable constants
CONSTANTS = ['C', '0', '1', '2', '3', '5', '10', 'pi', 'e']

# Full Vocabulary
VOCABULARY = list(OPERATORS.keys()) + VARIABLES + CONSTANTS
TOKEN_TO_ID = {token: i for i, token in enumerate(VOCABULARY)}
ID_TO_TOKEN = {i: token for token, i in TOKEN_TO_ID.items()}

# Special token for start of sequence
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
PAD_TOKEN = '<PAD>'

class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children else []

    def __repr__(self):
        if not self.children:
            return str(self.value)
        return f"({self.value} " + " ".join([str(c) for c in self.children]) + ")"
    
    def to_infix(self):
        if not self.children:
            return str(self.value)
        
        op = self.value
        if len(self.children) == 1:
            return f"{op}({self.children[0].to_infix()})"
        elif len(self.children) == 2:
            if op == 'pow':
                return f"({self.children[0].to_infix()} ^ {self.children[1].to_infix()})"
            elif op == 'mod':
                return f"({self.children[0].to_infix()} % {self.children[1].to_infix()})"
            return f"({self.children[0].to_infix()} {op} {self.children[1].to_infix()})"
        return str(self.value)
    
    def count_constants(self):
        """Count the number of 'C' placeholders in the tree."""
        count = 1 if self.value == 'C' else 0
        for child in self.children:
            count += child.count_constants()
        return count
    
    def get_constant_positions(self, path=None):
        """Returns a list of paths to all 'C' nodes for optimization."""
        if path is None:
            path = []
        positions = []
        if self.value == 'C':
            positions.append(path.copy())
        for i, child in enumerate(self.children):
            positions.extend(child.get_constant_positions(path + [i]))
        return positions


import ast

class ExpressionTree:
    def __init__(self, token_list):
        """
        Parses a list of tokens in Pre-order traversal (Prefix notation)
        Example: ['+', 'x', 'sin', 'x'] -> x + sin(x)
        """
        self.tokens = token_list
        try:
            self.root, remaining = self._build_tree(token_list)
            if remaining:
                raise ValueError("Tokens remained after building tree")
            self.is_valid = True
        except Exception:
            self.root = None
            self.is_valid = False

    @classmethod
    def from_infix(cls, infix_str):
        """
        Creates an ExpressionTree from a standard infix string (e.g. "sin(x) + x^2").
        Uses Python's ast to parse.
        """
        # Replacements to make it valid python for AST
        # 1. Handle postfix factorial '!' which C++ outputs as '(... )!'
        # We convert '(... )!' to 'gamma(...)'
        # Iterate until no '!' left
        processed_str = infix_str
        while '!' in processed_str:
            idx = processed_str.find('!')
            # Helper to find matching paren backwards
            if idx > 0 and processed_str[idx-1] == ')':
                paren_count = 1
                start = idx - 2
                while start >= 0 and paren_count > 0:
                    if processed_str[start] == ')':
                        paren_count += 1
                    elif processed_str[start] == '(':
                        paren_count -= 1
                    start -= 1
                # start is now 1 char before the matching '('
                start += 1 
                # Reconstruct: ... + gamma( + ... + ) + ...
                # Content includes the parens: ( ... )
                content = processed_str[start:idx] 
                processed_str = processed_str[:start] + "gamma" + content + processed_str[idx+1:]
            else:
                # Fallback: Just remove ! if it's weirdly placed (should not happen with GP output)
                processed_str = processed_str.replace('!', '', 1)

        # 2. C++ uses ^ for power, Python uses **. AST parses ^ as BitXor.
        try:
            tree = ast.parse(processed_str, mode='eval')
            tokens = cls._ast_to_prefix(tree.body)
            return cls(tokens)
        except Exception as e:
            print(f"Error parsing infix: {e} | Original: {infix_str} | Processed: {processed_str}")
            return cls([]) # Invalid

    @staticmethod
    def _ast_to_prefix(node):
        if isinstance(node, ast.BinOp):
            # Map operators
            op_map = {
                ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/',
                ast.BitXor: 'pow', ast.Pow: 'pow', ast.Mod: 'mod'
            }
            op_type = type(node.op)
            if op_type in op_map:
                return [op_map[op_type]] + ExpressionTree._ast_to_prefix(node.left) + ExpressionTree._ast_to_prefix(node.right)
        
        elif isinstance(node, ast.UnaryOp):
            op_map = {ast.USub: 'neg', ast.UAdd: None} # Ignore unary +
            op_type = type(node.op)
            if op_type == ast.USub:
                # Check directly if it's a number to collapse "-5"
                if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
                    return [str(-node.operand.value)]
                return ['neg'] + ExpressionTree._ast_to_prefix(node.operand)
            elif op_type == ast.UAdd:
                 return ExpressionTree._ast_to_prefix(node.operand)

        elif isinstance(node, ast.Call):
            # Functions like sin(x)
            func_id = node.func.id
            if func_id in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'floor', 'ceil', 'gamma', 'lgamma']:
                tokens = [func_id]
                for arg in node.args:
                    tokens.extend(ExpressionTree._ast_to_prefix(arg))
                return tokens
        
        elif isinstance(node, ast.Name):
            return [node.id]
        
        elif isinstance(node, ast.Constant): # Python 3.8+
            return [str(node.value)]
        elif isinstance(node, ast.Num): # Older python
            return [str(node.n)]

        raise ValueError(f"Unsupported AST node: {node}")


    def _build_tree(self, tokens):
        if not tokens:
            raise ValueError("Empty token list")
        
        token = tokens[0]
        remaining = tokens[1:]
        
        if token in OPERATORS:
            arity = OPERATORS[token]
            children = []
            for _ in range(arity):
                child, remaining = self._build_tree(remaining)
                children.append(child)
            return Node(token, children), remaining
        elif token in VARIABLES or token in CONSTANTS:
            return Node(token), remaining
        else:
            # Try to parse as float literal
            try:
                float(token)
                return Node(token), remaining
            except:
                raise ValueError(f"Unknown token: {token}")

    def evaluate(self, x_values, constants=None):
        """
        Evaluates the expression tree for a given array of x values.
        constants: optional dict mapping path tuples to constant values
        Returns a numpy array of results.
        """
        # Ensure x_values is a numpy array
        if not isinstance(x_values, np.ndarray):
            x_values = np.array(x_values, dtype=np.float64)
        
        if not self.is_valid:
            return np.full_like(x_values, np.nan, dtype=np.float64)
        return self._eval_node(self.root, x_values, constants, path=[])

    def _eval_node(self, node, x, constants=None, path=None):
        val = node.value
        
        if val == 'x':
            return x.astype(np.float64)
        if val == 'pi':
            return np.full_like(x, np.pi, dtype=np.float64)
        if val == 'e':
            return np.full_like(x, np.e, dtype=np.float64)
        if val == 'C':
            # Check if we have an optimized constant for this position
            if constants is not None and tuple(path) in constants:
                return np.full_like(x, constants[tuple(path)], dtype=np.float64)
            return np.full_like(x, 1.0, dtype=np.float64)  # Default constant = 1
        
        # Check for numeric constants
        try:
            return np.full_like(x, float(val), dtype=np.float64)
        except:
            pass
            
        # Recursive evaluation
        args = []
        for i, c in enumerate(node.children):
            args.append(self._eval_node(c, x, constants, path + [i] if path is not None else None))
        
        # Operators
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            if val == '+': return args[0] + args[1]
            if val == '-': return args[0] - args[1]
            if val == '*': return args[0] * args[1]
            if val == '/': 
                return np.divide(args[0], args[1], out=np.zeros_like(x, dtype=np.float64), where=args[1]!=0)
            if val == 'pow':
                # Safe power
                return np.power(np.abs(args[0]) + 1e-10, np.clip(args[1], -10, 10))
            if val == 'mod':
                return np.mod(args[0], args[1] + 1e-10)
            if val == 'sin': return np.sin(args[0])
            if val == 'cos': return np.cos(args[0])
            if val == 'tan': return np.tan(args[0])
            if val == 'exp': 
                return np.exp(np.clip(args[0], -100, 100))
            if val == 'log': 
                return np.log(np.abs(args[0]) + 1e-10)
            if val == 'sqrt':
                return np.sqrt(np.abs(args[0]))
            if val == 'abs':
                return np.abs(args[0])
            if val == 'floor':
                return np.floor(args[0])
            if val == 'ceil':
                return np.ceil(args[0])
            if val == 'gamma':
                # Match C++ Protected Gamma/Factorial: tgamma(|x| + 1)
                # This ensures consistent evaluation for formulas from C++ engine (which uses !)
                arg = np.abs(args[0]) + 1.0
                clipped = np.clip(arg, 0.1, 50) # Clip upper bound to avoid overflow
                return scipy_gamma(clipped)
            if val == 'lgamma':
                # Protected lgamma: lgamma(|x| + 1)
                arg = np.abs(args[0]) + 1.0
                # gammaln is safe for large positive numbers, so less aggressive clipping needed for overflow,
                # but we clip for consistency and to avoid extremely large outputs if followed by exp
                clipped = np.clip(arg, 0.1, 1000) 
                return gammaln(clipped)
            if val == 'neg':
                return -args[0]
            if val == 'sign':
                return np.sign(args[0])
                
        return np.zeros_like(x, dtype=np.float64)

    def get_infix(self):
        if not self.is_valid:
            return "Invalid"
        return self.root.to_infix()
    
    def count_constants(self):
        if not self.is_valid:
            return 0
        return self.root.count_constants()
