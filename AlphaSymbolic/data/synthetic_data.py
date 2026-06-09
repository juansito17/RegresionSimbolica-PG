import numpy as np
import random
from typing import List, Dict, Any, Optional
from AlphaSymbolic.core.grammar import ExpressionTree, OPERATORS, CONSTANTS, VARIABLES

class DataGenerator:
    """
    Generates synthetic data (X, Y) from random mathematical expression trees.
    Used for training and bootstrapping the genetic algorithm.
    """
    def __init__(self, max_depth: int = 4, num_variables: int = 1, allowed_operators: Optional[List[str]] = None):
        self.max_depth = max_depth
        self.num_variables = num_variables
        self.allowed_operators = allowed_operators if allowed_operators else list(OPERATORS.keys())
        
    def generate_batch(self, batch_size: int, point_count: int = 10, x_range: tuple = (-5, 5)) -> List[Dict[str, Any]]:
        """
        Generates a batch of synthetic problems.
        Returns a list of dictionaries with 'x', 'y', 'tokens', and 'infix'.
        """
        batch = []
        while len(batch) < batch_size:
            # Generate a random valid tree
            tree = ExpressionTree.generate_random(max_depth=self.max_depth, num_variables=self.num_variables)
            if not tree or not tree.is_valid:
                continue
            
            # Generate X values
            # For multi-variable support (num_variables > 1), we return a dict of arrays
            if self.num_variables == 1:
                x = np.random.uniform(x_range[0], x_range[1], point_count)
                x_input = x # ExpressionTree handle 1D array as x0
            else:
                x_input = np.random.uniform(x_range[0], x_range[1], (point_count, self.num_variables))
            
            # Evaluate Y values
            try:
                y = tree.evaluate(x_input)
                
                # Filter out invalid results (NaN, Inf, or extreme values)
                if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                    continue
                if np.max(np.abs(y)) > 1e6:
                    continue
                
                batch.append({
                    'x': x_input,
                    'y': y,
                    'tokens': tree.tokens,
                    'infix': tree.get_infix()
                })
            except Exception:
                continue
                
        return batch

    def generate_inverse_batch(self, batch_size: int, point_count: int = 10) -> List[Dict[str, Any]]:
        """
        Alias for generate_batch, as it generates the "inverse" (formula) for a given X,Y.
        In AlphaSymbolic terminology, 'inverse' refers to the formula itself.
        """
        return self.generate_batch(batch_size, point_count)

# Compatibility Alias
SyntheticDataGenerator = DataGenerator
