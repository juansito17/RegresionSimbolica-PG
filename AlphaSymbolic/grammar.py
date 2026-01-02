import numpy as np
import math

# Supported operators and their arity (number of arguments)
OPERATORS = {
    '+': 2,
    '-': 2,
    '*': 2,
    '/': 2,
    'sin': 1,
    'cos': 1,
    'exp': 1,
    'log': 1,
    'pow': 2
}

# Terminal tokens
VARIABLES = ['x']
CONSTANTS = ['C', '1', '2', '3', '5', 'pi']

# Full Vocabulary
VOCABULARY = list(OPERATORS.keys()) + VARIABLES + CONSTANTS
TOKEN_TO_ID = {token: i for i, token in enumerate(VOCABULARY)}
ID_TO_TOKEN = {i: token for token, i in TOKEN_TO_ID.items()}

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
            return f"({self.children[0].to_infix()} {op} {self.children[1].to_infix()})"
        return str(self.value)


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
            # Check if it is a number
            return Node(token), remaining
        else:
             # Try to parse as float literal if not in explicit constants list (optional)
            try:
                float(token)
                return Node(token), remaining
            except:
                raise ValueError(f"Unknown token: {token}")

    def evaluate(self, x_values):
        """
        Evaluates the expression tree for a given array of x values.
        Returns a numpy array of results.
        """
        if not self.is_valid:
            return np.full_like(x_values, np.nan)
        return self._eval_node(self.root, x_values)

    def _eval_node(self, node, x):
        val = node.value
        
        if val == 'x':
            return x
        if val == 'pi':
            return np.full_like(x, np.pi)
        
        # Check for numeric constants
        try:
            return np.full_like(x, float(val))
        except:
            pass
            
        # Recursive evaluation
        args = [self._eval_node(c, x) for c in node.children]
        
        # Operators
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            if val == '+': return args[0] + args[1]
            if val == '-': return args[0] - args[1]
            if val == '*': return args[0] * args[1]
            if val == '/': 
                return np.divide(args[0], args[1], out=np.zeros_like(x), where=args[1]!=0)
            if val == 'sin': return np.sin(args[0])
            if val == 'cos': return np.cos(args[0])
            if val == 'exp': 
                # Clip to avoid overflow
                return np.exp(np.clip(args[0], -100, 100))
            if val == 'log': 
                # Log of absolute value, safe log
                return np.log(np.abs(args[0]) + 1e-6)
            if val == 'pow':
                # Safe power (e.g. avoid complex numbers or overflow)
                return np.power(np.abs(args[0]), args[1]) # Simplified safe power
                
        return np.zeros_like(x)

    def get_infix(self):
        if not self.is_valid:
            return "Invalid"
        return self.root.to_infix()
