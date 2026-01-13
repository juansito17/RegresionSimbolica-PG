"""
Multi-Variable Support for AlphaSymbolic.
Extends the grammar to support multiple input variables: x1, x2, x3, etc.
"""
import numpy as np
import torch
from scipy.special import gamma as scipy_gamma

# Multi-variable operators (same as single variable)
OPERATORS = {
    '+': 2, '-': 2, '*': 2, '/': 2, 'pow': 2, 'mod': 2,
    'sin': 1, 'cos': 1, 'tan': 1, 'exp': 1, 'log': 1,
    'sqrt': 1, 'abs': 1, 'floor': 1, 'ceil': 1, 'gamma': 1, 'neg': 1,
}

# Constants
CONSTANTS = ['C', '0', '1', '2', '3', '5', '10', 'pi', 'e']


def build_vocabulary(num_variables):
    """Build vocabulary for N input variables."""
    variables = [f'x{i}' for i in range(num_variables)]
    vocab = list(OPERATORS.keys()) + variables + CONSTANTS
    token_to_id = {token: i for i, token in enumerate(vocab)}
    id_to_token = {i: token for token, i in token_to_id.items()}
    return vocab, variables, token_to_id, id_to_token


class MultiVarNode:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children else []
    
    def to_infix(self):
        if not self.children:
            return str(self.value)
        op = self.value
        if len(self.children) == 1:
            return f"{op}({self.children[0].to_infix()})"
        elif len(self.children) == 2:
            if op == 'pow':
                return f"({self.children[0].to_infix()} ^ {self.children[1].to_infix()})"
            return f"({self.children[0].to_infix()} {op} {self.children[1].to_infix()})"
        return str(self.value)
    
    def count_constants(self):
        count = 1 if self.value == 'C' else 0
        for child in self.children:
            count += child.count_constants()
        return count


class MultiVarExpressionTree:
    """Expression tree that supports multiple variables."""
    
    def __init__(self, token_list, num_variables=2):
        self.tokens = token_list
        self.num_variables = num_variables
        self.variables = [f'x{i}' for i in range(num_variables)]
        
        try:
            self.root, remaining = self._build_tree(token_list)
            if remaining:
                raise ValueError("Tokens remained")
            self.is_valid = True
        except:
            self.root = None
            self.is_valid = False
    
    def _build_tree(self, tokens):
        if not tokens:
            raise ValueError("Empty")
        
        token = tokens[0]
        remaining = tokens[1:]
        
        if token in OPERATORS:
            arity = OPERATORS[token]
            children = []
            for _ in range(arity):
                child, remaining = self._build_tree(remaining)
                children.append(child)
            return MultiVarNode(token, children), remaining
        elif token in self.variables or token in CONSTANTS:
            return MultiVarNode(token), remaining
        else:
            try:
                float(token)
                return MultiVarNode(token), remaining
            except:
                raise ValueError(f"Unknown token: {token}")
    
    def evaluate(self, x_values_dict, constants=None):
        """
        Evaluate with multiple variables.
        x_values_dict: {'x0': array, 'x1': array, ...}
        """
        if not self.is_valid:
            n = len(list(x_values_dict.values())[0]) if x_values_dict else 1
            return np.full(n, np.nan)
        return self._eval_node(self.root, x_values_dict, constants or {}, [])
    
    def _eval_node(self, node, x_dict, constants, path):
        val = node.value
        
        # Check if it's a variable
        if val in x_dict:
            return x_dict[val].astype(np.float64)
        if val == 'pi':
            n = len(list(x_dict.values())[0])
            return np.full(n, np.pi)
        if val == 'e':
            n = len(list(x_dict.values())[0])
            return np.full(n, np.e)
        if val == 'C':
            n = len(list(x_dict.values())[0])
            c_val = constants.get(tuple(path), 1.0)
            return np.full(n, c_val)
        
        try:
            n = len(list(x_dict.values())[0])
            return np.full(n, float(val))
        except:
            pass
        
        # Operators
        args = []
        for i, c in enumerate(node.children):
            args.append(self._eval_node(c, x_dict, constants, path + [i]))
        
        with np.errstate(all='ignore'):
            if val == '+': return args[0] + args[1]
            if val == '-': return args[0] - args[1]
            if val == '*': return args[0] * args[1]
            if val == '/': return np.divide(args[0], args[1], out=np.zeros_like(args[0]), where=args[1]!=0)
            if val == 'pow': return np.power(np.abs(args[0]) + 1e-10, np.clip(args[1], -10, 10))
            if val == 'mod': return np.mod(args[0], args[1] + 1e-10)
            if val == 'sin': return np.sin(args[0])
            if val == 'cos': return np.cos(args[0])
            if val == 'tan': return np.tan(args[0])
            if val == 'exp': return np.exp(np.clip(args[0], -100, 100))
            if val == 'log': return np.log(np.abs(args[0]) + 1e-10)
            if val == 'sqrt': return np.sqrt(np.abs(args[0]))
            if val == 'abs': return np.abs(args[0])
            if val == 'floor': return np.floor(args[0])
            if val == 'ceil': return np.ceil(args[0])
            if val == 'gamma': 
                clipped = np.clip(args[0], 0.1, 50)
                return scipy_gamma(clipped)
            if val == 'neg': return -args[0]
        
        return np.zeros_like(args[0]) if args else np.zeros(1)
    
    def get_infix(self):
        if not self.is_valid:
            return "Invalid"
        return self.root.to_infix()


class MultiVarDataGenerator:
    """Generate synthetic data for multi-variable regression."""
    
    def __init__(self, num_variables=2, max_depth=4):
        self.num_variables = num_variables
        self.max_depth = max_depth
        self.vocab, self.variables, self.token_to_id, _ = build_vocabulary(num_variables)
        self.operators = list(OPERATORS.keys())
        self.terminals = self.variables + CONSTANTS
    
    def generate_random_tree(self, max_depth, current_depth=0):
        import random
        
        if current_depth >= max_depth:
            return [random.choice(self.terminals)]
        
        if random.random() < 0.7:
            op = random.choice(self.operators)
            arity = OPERATORS[op]
            tokens = [op]
            for _ in range(arity):
                tokens.extend(self.generate_random_tree(max_depth, current_depth + 1))
            return tokens
        else:
            return [random.choice(self.terminals)]
    
    def generate_batch(self, batch_size, points_per_dim=10, x_range=(-5, 5)):
        """Generate batch with multi-variable data."""
        data = []
        
        while len(data) < batch_size:
            tokens = self.generate_random_tree(self.max_depth)
            tree = MultiVarExpressionTree(tokens, self.num_variables)
            
            if not tree.is_valid:
                continue
            
            # Generate input points
            x_dict = {}
            for i in range(self.num_variables):
                x_dict[f'x{i}'] = np.random.uniform(x_range[0], x_range[1], points_per_dim)
            
            y_values = tree.evaluate(x_dict)
            
            if np.any(np.isnan(y_values)) or np.any(np.isinf(y_values)):
                continue
            if np.max(np.abs(y_values)) > 1e6:
                continue
            
            data.append({
                'tokens': tokens,
                'infix': tree.get_infix(),
                'x': x_dict,
                'y': y_values
            })
        
        return data


# Quick test
if __name__ == "__main__":
    # Build vocabulary for 3 variables
    vocab, variables, t2i, i2t = build_vocabulary(3)
    print(f"Variables: {variables}")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Test expression: x0 + x1 * x2
    tokens = ['+', 'x0', '*', 'x1', 'x2']
    tree = MultiVarExpressionTree(tokens, num_variables=3)
    print(f"Formula: {tree.get_infix()}")
    
    # Evaluate
    x_dict = {
        'x0': np.array([1, 2, 3]),
        'x1': np.array([2, 3, 4]),
        'x2': np.array([3, 4, 5])
    }
    result = tree.evaluate(x_dict)
    expected = x_dict['x0'] + x_dict['x1'] * x_dict['x2']
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    print(f"Match: {np.allclose(result, expected)}")
    
    # Test data generator
    gen = MultiVarDataGenerator(num_variables=2, max_depth=3)
    batch = gen.generate_batch(3)
    for item in batch:
        print(f"\nFormula: {item['infix']}")
        print(f"Y sample: {item['y'][:3]}")
