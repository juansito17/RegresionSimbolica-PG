import numpy as np
from scipy.special import gamma as scipy_gamma
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
                # Safe gamma (clip to avoid overflow)
                clipped = np.clip(args[0], -50, 50)
                result = np.zeros_like(clipped)
                valid = clipped > 0
                result[valid] = scipy_gamma(clipped[valid])
                return result
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
