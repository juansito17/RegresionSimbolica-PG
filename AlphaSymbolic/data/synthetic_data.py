import numpy as np
import random
from core.grammar import VOCABULARY, OPERATORS, VARIABLES, CONSTANTS, ExpressionTree

class DataGenerator:
    def __init__(self, max_depth=5, population_size=1000):
        self.max_depth = max_depth
        self.population_size = population_size
        self.vocab = VOCABULARY
        # Pre-compute terminal vs operator lists
        self.terminals = VARIABLES + CONSTANTS
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

# Quick test if run directly
if __name__ == "__main__":
    gen = DataGenerator(max_depth=4)
    batch = gen.generate_batch(5)
    for item in batch:
        print(f"Formula: {item['infix']}")
        print(f"Tokens: {item['tokens']}")
        print(f"Y sample: {item['y'][:3]}...")
        print("-" * 20)
