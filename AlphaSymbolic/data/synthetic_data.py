import numpy as np
import random
from core.grammar import VOCABULARY, OPERATORS, VARIABLES, CONSTANTS, ExpressionTree
from data.augmentation import augment_formula_tokens

class DataGenerator:
    def __init__(self, max_depth=5, population_size=1000, allowed_operators=None):
        self.max_depth = max_depth
        self.population_size = population_size
        self.vocab = VOCABULARY
        # Pre-compute terminal vs operator lists
        self.terminals = VARIABLES + CONSTANTS
        if allowed_operators:
            self.operators = [op for op in allowed_operators if op in OPERATORS]
        else:
            self.operators = list(OPERATORS.keys())

    def generate_random_tree(self, max_depth, current_depth=0):
        if current_depth >= max_depth:
            # Must return a terminal
            return [random.choice(self.terminals)]
        
        # Decide if terminal or operator
        # Higher probability of operator at shallow depths
        if random.random() < 0.7: 
            op = random.choice(self.operators)
            arity = OPERATORS[op]
            tokens = [op]
            for _ in range(arity):
                tokens.extend(self.generate_random_tree(max_depth, current_depth + 1))
            return tokens
        else:
            return [random.choice(self.terminals)]

    def generate_batch(self, batch_size, point_count=10, x_range=(-10, 10)):
        """
        Generates a batch of (X, Y) pairs and their generating formulas.
        """
        data = []
        
        while len(data) < batch_size:
            # Generate random formula
            tokens = self.generate_random_tree(self.max_depth)
            tree = ExpressionTree(tokens)
            
            if not tree.is_valid:
                continue
                
            # Generate random X points
            x_values = np.random.uniform(x_range[0], x_range[1], point_count)
            # Sort X for cleaner visualization/learning
            x_values.sort()
            
            # Calculate Y
            y_values = tree.evaluate(x_values)
            
            # Check for validity (no NaNs, Infs, or extremely large values)
            if np.any(np.isnan(y_values)) or np.any(np.isinf(y_values)):
                continue
            if np.max(np.abs(y_values)) > 1e6: # Reject too large numbers
                continue
            if np.std(y_values) < 1e-6: # Reject flat lines (too simple)
                 # Optionally keep some, but mostly we want interesting curves
                 if random.random() > 0.1: continue

            data.append({
                'tokens': tokens,
                'infix': tree.get_infix(),
                'x': x_values,
                'y': y_values
            })
            
        return data

    def generate_inverse_batch(self, batch_size, point_count=10, x_range=(-5, 5)):
        """
        Inverse data generation (AlphaTensor-style):
        Generate KNOWN formulas with guaranteed solutions.
        This helps the model learn from solvable problems first.
        """
        data = []
        
        # Known formula templates with their token representations
        templates = [
            # Linear: a*x + b
            lambda a, b: (['+', '*', str(a), 'x', str(b)], f"({a}*x + {b})"),
            # Quadratic: a*x^2 + b
            lambda a, b: (['+', '*', str(a), 'pow', 'x', '2', str(b)], f"({a}*x^2 + {b})"),
            # Simple sin: sin(x)
            lambda a, b: (['sin', 'x'], "sin(x)"),
            # Scaled sin: a*sin(x)
            lambda a, b: (['*', str(a), 'sin', 'x'], f"{a}*sin(x)"),
            # Exponential: exp(x/a)
            lambda a, b: (['exp', '/', 'x', str(max(1, abs(a)))], f"exp(x/{max(1, abs(a))})"),
            # Square root: sqrt(x + a) 
            lambda a, b: (['sqrt', '+', 'x', str(abs(a)+1)], f"sqrt(x+{abs(a)+1})"),
            # Polynomial: x^2 - a
            lambda a, b: (['-', 'pow', 'x', '2', str(a)], f"(x^2 - {a})"),
            # Cosine
            lambda a, b: (['cos', 'x'], "cos(x)"),
        ]
        
        while len(data) < batch_size:
            # Random coefficients (small integers for stability)
            a = random.randint(1, 5)
            b = random.randint(-3, 3)
            
            # Pick random template
            template = random.choice(templates)
            
            try:
                tokens, formula_str = template(a, b)
                
                # Convert string numbers -> 'C'
                final_tokens = []
                for t in tokens:
                    if t in VOCABULARY:
                        final_tokens.append(t)
                    else:
                        final_tokens.append('C')
                
                # --- DATA AUGMENTATION (AlphaTensor Style) ---
                # Apply mathematical invariances (Commutativity, etc.)
                # This multiplies the effective dataset size
                if random.random() < 0.5:
                    final_tokens = augment_formula_tokens(final_tokens)
                # ---------------------------------------------
                
                tree = ExpressionTree(final_tokens)
                if not tree.is_valid:
                    continue
                
                # Generate X points (positive for sqrt/log safety)
                if 'sqrt' in final_tokens or 'log' in final_tokens:
                    x_values = np.linspace(0.5, x_range[1], point_count)
                else:
                    x_values = np.linspace(x_range[0], x_range[1], point_count)
                
                y_values = tree.evaluate(x_values)
                
                # Validity checks
                if np.any(np.isnan(y_values)) or np.any(np.isinf(y_values)):
                    continue
                if np.max(np.abs(y_values)) > 1e6:
                    continue
                
                data.append({
                    'tokens': final_tokens,
                    'infix': tree.get_infix(),
                    'x': x_values,
                    'y': y_values
                })
            except:
                continue
                
        return data

# Quick test if run directly
if __name__ == "__main__":
    gen = DataGenerator(max_depth=4)
    batch = gen.generate_batch(5)
    for item in batch:
        print(f"Formula: {item['infix']}")
        print(f"Tokens: {item['tokens']}")
        print(f"Y sample: {item['y'][:3]}...")
        print("-" * 20)
