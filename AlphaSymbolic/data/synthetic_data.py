import numpy as np
import random
from core.grammar import VOCABULARY, OPERATORS, VARIABLES, CONSTANTS, ExpressionTree
from data.augmentation import augment_formula_tokens

class DataGenerator:
    def __init__(self, max_depth=5, population_size=1000, allowed_operators=None, num_variables=1):
        self.max_depth = max_depth
        self.population_size = population_size
        self.num_variables = num_variables
        self.vocab = VOCABULARY
        # Use subset of variables based on num_variables
        self.active_variables = VARIABLES[:num_variables] if num_variables > 1 else ['x']
        
        # Pre-compute terminal vs operator lists
        self.terminals = self.active_variables + CONSTANTS
        if allowed_operators:
            self.operators = [op for op in allowed_operators if op in OPERATORS]
        else:
            self.operators = list(OPERATORS.keys())

    def generate_random_tree(self, max_depth, current_depth=0):
        if current_depth >= max_depth:
            # Balanced Terminal Selection: 50% variable, 50% constant
            if random.random() < 0.5:
                return [random.choice(self.active_variables)]
            else:
                return [random.choice(CONSTANTS)]
        
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
            # Balanced Terminal Selection: 40% var, 30% C, 30% numbers
            r = random.random()
            if r < 0.4:
                return [random.choice(self.active_variables)]
            elif r < 0.7:
                return ['C']
            else:
                return [random.choice([c for c in CONSTANTS if c != 'C'])]

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
            
            # Ensure variables are present (90% of the time, check any active variable)
            if not any(v in tokens for v in self.active_variables) and random.random() < 0.9:
                continue
                
            # Generate random X points
            # If num_variables > 1, shape (point_count, num_variables)
            # If num_variables == 1, shape (point_count,) or (point_count, 1) - but maintain compat
            if self.num_variables > 1:
                x_values = np.random.uniform(x_range[0], x_range[1], (point_count, self.num_variables))
                # Sorting 2D array by first col just for consistent indexing? or keep random?
                # Maybe sort by first column for visualization
                x_values = x_values[x_values[:, 0].argsort()]
            else:
                x_values = np.random.uniform(x_range[0], x_range[1], point_count)
                # Sort X for cleaner visualization/learning
                x_values.sort()
            
            # Randomize 'C' values if present
            c_positions = tree.root.get_constant_positions()
            constant_vals = {}
            for pos in c_positions:
                # Expanded range: -20 to 20. Favor 1.0 occasionally
                val = random.uniform(-20, 20) if random.random() > 0.1 else 1.0
                constant_vals[tuple(pos)] = val
            
            # Calculate Y with randomized constants
            y_values = tree.evaluate(x_values, constants=constant_vals)
            
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
            # Randomly choose between active_variables, C and constants
            r = random.random()
            if r < 0.4: return [random.choice(self.active_variables)]
            if r < 0.7: return ['C']
            return [random.choice([c for c in CONSTANTS if c != 'C'])]
            
        # Filter available structures based on allowed operators
        available_structures = []
        
        # Arithmetic needed: +, -, *
        if any(op in self.operators for op in ['+', '-', '*']):
            available_structures.append('arithmetic')
            
        # Poly needed: pow
        if 'pow' in self.operators:
            available_structures.append('poly')
            
        # Trig needed: sin, cos, asin, acos, atan
        if any(op in self.operators for op in ['sin', 'cos', 'asin', 'acos', 'atan']):
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
            ops = [op for op in ['sin', 'cos', 'asin', 'acos', 'atan'] if op in self.operators]
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
            # Random complexity capped by max_depth
            complexity = random.randint(1, max(1, self.max_depth - 1))
            
            try:
                # Use random variable as starting seed if needed, but structured tree handles selection at leaves
                tokens = self.generate_structured_tree(complexity, random.choice(self.active_variables))
                
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
                
                # Ensure variables are present (90% of the time)
                if not any(v in final_tokens for v in self.active_variables) and random.random() < 0.9:
                    continue
                    
                # Check constraints (depth, length)
                if len(final_tokens) > 30: # Limit length
                    continue

                # Generate X points
                # Use safer range for complex funcs
                # Exp/Pow grow very fast, so we constrain X to avoid float overflow
                range_limit = x_range
                if 'exp' in final_tokens or 'pow' in final_tokens:
                    range_limit = (-2, 2)
                elif 'log' in final_tokens or 'sqrt' in final_tokens:
                    range_limit = (0.1, 5)

                if self.num_variables > 1:
                    x_safe = np.linspace(range_limit[0], range_limit[1], point_count)
                    # For multivar, linspace per column or random?
                    # Let's use random uniform for coverage in multivar space
                    x_safe = np.random.uniform(range_limit[0], range_limit[1], (point_count, self.num_variables))
                else:
                    x_safe = np.linspace(range_limit[0], range_limit[1], point_count)
                
                # Randomize 'C' values if present
                c_positions = tree.root.get_constant_positions()
                constant_vals = {}
                for pos in c_positions:
                    # Expanded range: -20 to 20
                    val = random.uniform(-20, 20) if random.random() > 0.1 else 1.0
                    constant_vals[tuple(pos)] = val
                
                y_values = tree.evaluate(x_safe, constants=constant_vals)
                
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
