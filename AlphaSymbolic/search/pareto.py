"""
Pareto Front Manager for AlphaSymbolic.
Maintains a set of non-dominated solutions (accuracy vs complexity).
"""
import numpy as np
from core.grammar import ExpressionTree

class ParetoSolution:
    def __init__(self, tokens, rmse, complexity, formula_str, constants=None):
        self.tokens = tokens
        self.rmse = rmse  # Lower is better
        self.complexity = complexity  # Lower is better (number of nodes)
        self.formula = formula_str
        self.constants = constants or {}
        
    def dominates(self, other):
        """Returns True if self dominates other (better in all objectives)."""
        # Self dominates other if:
        # - Self is at least as good in all objectives
        # - Self is strictly better in at least one objective
        at_least_as_good = (self.rmse <= other.rmse) and (self.complexity <= other.complexity)
        strictly_better = (self.rmse < other.rmse) or (self.complexity < other.complexity)
        return at_least_as_good and strictly_better
    
    def __repr__(self):
        return f"ParetoSolution(rmse={self.rmse:.4f}, complexity={self.complexity}, formula='{self.formula}')"


class ParetoFront:
    def __init__(self, max_size=50):
        self.solutions = []
        self.max_size = max_size
        
    def add(self, solution):
        """
        Attempts to add a solution to the Pareto front.
        Returns True if added, False if dominated.
        """
        # Check if new solution is dominated by any existing
        for existing in self.solutions:
            if existing.dominates(solution):
                return False  # New solution is dominated
        
        # Remove any solutions dominated by the new one
        self.solutions = [s for s in self.solutions if not solution.dominates(s)]
        
        # Add the new solution
        self.solutions.append(solution)
        
        # Enforce max size by removing worst solutions
        if len(self.solutions) > self.max_size:
            # Sort by a combined score and keep top max_size
            self.solutions.sort(key=lambda s: s.rmse + 0.01 * s.complexity)
            self.solutions = self.solutions[:self.max_size]
        
        return True
    
    def add_from_results(self, results_list):
        """
        Add multiple results from beam search or MCTS.
        results_list: list of dicts with 'tokens', 'rmse', 'constants', 'formula'
        """
        added = 0
        for r in results_list:
            tree = ExpressionTree(r['tokens'])
            complexity = len(r['tokens'])  # Simple complexity = token count
            
            sol = ParetoSolution(
                tokens=r['tokens'],
                rmse=r['rmse'],
                complexity=complexity,
                formula_str=r['formula'],
                constants=r.get('constants', {})
            )
            
            if self.add(sol):
                added += 1
        
        return added
    
    def get_best_by_rmse(self):
        """Returns the solution with lowest RMSE."""
        if not self.solutions:
            return None
        return min(self.solutions, key=lambda s: s.rmse)
    
    def get_simplest(self):
        """Returns the solution with lowest complexity."""
        if not self.solutions:
            return None
        return min(self.solutions, key=lambda s: s.complexity)
    
    def get_balanced(self, alpha=0.5):
        """
        Returns a balanced solution.
        alpha: weight for RMSE (1-alpha for complexity)
        """
        if not self.solutions:
            return None
        
        # Normalize scores
        rmse_vals = [s.rmse for s in self.solutions]
        comp_vals = [s.complexity for s in self.solutions]
        
        min_rmse, max_rmse = min(rmse_vals), max(rmse_vals) + 1e-10
        min_comp, max_comp = min(comp_vals), max(comp_vals) + 1e-10
        
        def score(s):
            norm_rmse = (s.rmse - min_rmse) / (max_rmse - min_rmse)
            norm_comp = (s.complexity - min_comp) / (max_comp - min_comp)
            return alpha * norm_rmse + (1 - alpha) * norm_comp
        
        return min(self.solutions, key=score)
    
    def summary(self):
        """Print a summary of the Pareto front."""
        print(f"\n=== Pareto Front ({len(self.solutions)} solutions) ===")
        for i, sol in enumerate(sorted(self.solutions, key=lambda s: s.rmse)[:10]):
            print(f"  {i+1}. RMSE={sol.rmse:.6f}, Nodes={sol.complexity}, Formula: {sol.formula}")


# Quick test
if __name__ == "__main__":
    front = ParetoFront()
    
    # Add some test solutions
    solutions = [
        ParetoSolution(['x'], 10.0, 1, "x"),
        ParetoSolution(['+', 'x', '1'], 5.0, 3, "(x + 1)"),
        ParetoSolution(['*', '2', 'x'], 3.0, 3, "(2 * x)"),
        ParetoSolution(['+', '*', '2', 'x', '3'], 0.5, 5, "((2 * x) + 3)"),
        ParetoSolution(['+', '*', '*', '2', 'x', 'x', '+', 'x', '1'], 0.1, 9, "complicated"),
    ]
    
    for sol in solutions:
        added = front.add(sol)
        print(f"Added {sol.formula}: {added}")
    
    front.summary()
    
    print(f"\nBest by RMSE: {front.get_best_by_rmse()}")
    print(f"Simplest: {front.get_simplest()}")
    print(f"Balanced: {front.get_balanced()}")
