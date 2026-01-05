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

    def generate_structured_tree(self, complexity=1, input_node='x'):
        """
        Recursively builds a structured, human-like formula.
        Respects self.operators.
        """
        # Base cases
        if complexity <= 0:
            return input_node if isinstance(input_node, list) else [input_node]
            
        # Filter available structures based on allowed operators
        available_structures = []
        
        # Arithmetic needed: +, -, *
        if any(op in self.operators for op in ['+', '-', '*']):
            available_structures.append('arithmetic')
            
        # Poly needed: pow
        if 'pow' in self.operators:
            available_structures.append('poly')
            
        # Trig needed: sin, cos
        if 'sin' in self.operators or 'cos' in self.operators:
            available_structures.append('trig')
            
        # Exp/Log needed
        if 'exp' in self.operators or 'log' in self.operators:
            available_structures.append('exp_log')
            
        # Composition needs enough variety
        if len(self.operators) > 4 and complexity > 1:
             available_structures.append('composition')
        
        # Fallback if nothing allowed matches (shouldn't happen with proper init)
        if not available_structures:
            return input_node if isinstance(input_node, list) else [input_node]

        choice = random.choice(available_structures)
        
        if choice == 'poly':
            # a*x + b or a*x^2 + b
            a = str(random.randint(1, 5))
            b = str(random.randint(-5, 5))
            power = random.choice(['1', '2', '3'])
            if power == '1':
                term = ['*', a] + (input_node if isinstance(input_node, list) else [input_node])
                return ['+', ] + term + [b]
            else:
                base = input_node if isinstance(input_node, list) else [input_node]
                pow_term = ['pow'] + base + [power]
                term = ['*', a] + pow_term
                return ['+', ] + term + [b]
                
        elif choice == 'trig':
            # Filter trig ops that are allowed
            ops = [op for op in ['sin', 'cos'] if op in self.operators]
            if not ops: return input_node # Should be caught by structure check
            func = random.choice(ops)
            val = input_node if isinstance(input_node, list) else [input_node]
            return [func] + val
            
        elif choice == 'exp_log':
            ops = [op for op in ['exp', 'log'] if op in self.operators]
            if not ops: return input_node
            func = random.choice(ops)
            val = input_node if isinstance(input_node, list) else [input_node]
            return [func] + val
            
        elif choice == 'arithmetic':
            left = self.generate_structured_tree(complexity - 1, input_node)
            right = self.generate_structured_tree(complexity - 1, input_node)
            ops = [op for op in ['+', '-', '*'] if op in self.operators]
            if not ops: return input_node
            op = random.choice(ops)
            return [op] + left + right
            
        elif choice == 'composition':
            inner = self.generate_structured_tree(complexity - 1, input_node)
            outer = self.generate_structured_tree(1, inner)
            return outer
            
        return [input_node]

    def generate_inverse_batch(self, batch_size, point_count=10, x_range=(-5, 5)):
        """
        Generates complex, structured formulas using the new engine.
        """
        data = []
        attempts = 0
        
        while len(data) < batch_size and attempts < batch_size * 5:
            attempts += 1
            # Random complexity 1 to 3
            complexity = random.randint(1, 3)
            
            try:
                tokens = self.generate_structured_tree(complexity, 'x')
                
                # Convert numeric strings to 'C' placeholders if needed
                # But here we want the GROUND TRUTH tokens with numbers for checking?
                # The model predicts tokens. 'C' is for optimization.
                # If we train "End-to-End" (predict 3*x), we keep numbers.
                # If we train "Symbolic" (predict C*x), we swap.
                # The original code swapped numbers to 'C'. Let's check VOCABULARY.
                # '1','2','3' are in VOCABULARY. So we can keep small integers.
                # Large integers -> 'C'.
                
                final_tokens = []
                for t in tokens:
                    if t in self.vocab:
                        final_tokens.append(t)
                    else:
                        # If it's a number not in vocab, map to C?
                        # Or just nearest constant?
                        # For now, simplistic mapping:
                        try:
                            val = float(t)
                            if abs(val - round(val)) < 0.01 and str(int(round(val))) in self.vocab:
                                final_tokens.append(str(int(round(val))))
                            else:
                                final_tokens.append('C')
                        except:
                            final_tokens.append('C')

                # --- DATA AUGMENTATION ---
                if random.random() < 0.3:
                    final_tokens = augment_formula_tokens(final_tokens)
                # -------------------------
                
                tree = ExpressionTree(final_tokens)
                if not tree.is_valid:
                    continue
                
                # Check constraints (depth, length)
                if len(final_tokens) > 30: # Limit length
                    continue

                # Generate X points
                # Use safer range for complex funcs
                # Exp/Pow grow very fast, so we constrain X to avoid float overflow
                if 'exp' in final_tokens or 'pow' in final_tokens:
                    x_safe = np.linspace(-2, 2, point_count)
                elif 'log' in final_tokens or 'sqrt' in final_tokens:
                    x_safe = np.linspace(0.1, 5, point_count)
                else:
                    x_safe = np.linspace(x_range[0], x_range[1], point_count)
                
                y_values = tree.evaluate(x_safe)
                
                # Quality Control
                if np.any(np.isnan(y_values)) or np.any(np.isinf(y_values)):
                    continue
                if np.max(np.abs(y_values)) > 1e4: # Relaxed limit
                    continue
                if np.std(y_values) < 0.01: # Too flat
                    continue
                
                data.append({
                    'tokens': final_tokens,
                    'infix': tree.get_infix(),
                    'x': x_safe,
                    'y': y_values
                })
            except Exception:
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
