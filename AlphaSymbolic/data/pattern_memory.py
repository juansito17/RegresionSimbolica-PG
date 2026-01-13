"""
Pattern Memory for AlphaSymbolic.
Stores successful formula patterns for experience replay and warm-starting searches.
"""
import json
import os
from datetime import datetime

class PatternMemory:
    DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "benchmarks", "pattern_memory.json")
    def __init__(self, path="pattern_memory.json", max_patterns=500):
        self.path = path
        self.max_patterns = max_patterns
        self.patterns = {}  # pattern_str -> PatternInfo
        self.load()
    
    def load(self):
        """Load patterns from disk."""
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    data = json.load(f)
                    self.patterns = data.get('patterns', {})
                print(f"Loaded {len(self.patterns)} patterns from {self.path}")
            except:
                self.patterns = {}
    
    def save(self):
        """Save patterns to disk."""
        with open(self.path, 'w') as f:
            json.dump({
                'patterns': self.patterns,
                'updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def record(self, tokens, rmse, formula_str):
        """
        Record a successful pattern.
        """
        # Extract structure (replace constants with 'C')
        structure = self._extract_structure(tokens)
        key = ' '.join(structure)
        
        if key not in self.patterns:
            self.patterns[key] = {
                'structure': structure,
                'best_rmse': rmse,
                'uses': 0,
                'examples': []
            }
        
        info = self.patterns[key]
        info['uses'] += 1
        
        if rmse < info['best_rmse']:
            info['best_rmse'] = rmse
        
        # Store some examples
        if len(info['examples']) < 5:
            info['examples'].append({
                'tokens': tokens,
                'formula': formula_str,
                'rmse': rmse
            })
        
        # Prune if too many patterns
        if len(self.patterns) > self.max_patterns:
            self._prune()
        
        return key
    
    def _extract_structure(self, tokens):
        """Extract structural pattern (replace numeric constants with 'C')."""
        structure = []
        for t in tokens:
            if t in ['0', '1', '2', '3', '5', '10', 'C', 'pi', 'e']:
                structure.append('C')  # Generalize constants
            else:
                structure.append(t)
        return structure
    
    def _prune(self):
        """Remove least useful patterns."""
        # Sort by (best_rmse ASC, uses DESC)
        sorted_patterns = sorted(
            self.patterns.items(),
            key=lambda kv: (kv[1]['best_rmse'], -kv[1]['uses'])
        )
        
        # Keep top max_patterns
        self.patterns = dict(sorted_patterns[:self.max_patterns])
    
    def get_best_patterns(self, n=10):
        """Get the top N patterns by RMSE."""
        sorted_patterns = sorted(
            self.patterns.items(),
            key=lambda kv: kv[1]['best_rmse']
        )
        return sorted_patterns[:n]
    
    def get_frequent_patterns(self, n=10):
        """Get the top N most frequently used patterns."""
        sorted_patterns = sorted(
            self.patterns.items(),
            key=lambda kv: -kv[1]['uses']
        )
        return sorted_patterns[:n]
    
    def suggest_starting_tokens(self):
        """Suggest good starting tokens based on pattern history."""
        if not self.patterns:
            return None
        
        # Get best pattern
        best = self.get_best_patterns(1)
        if best:
            return best[0][1]['structure']
        return None
    
    def summary(self):
        """Print a summary of stored patterns."""
        print(f"\n=== Pattern Memory ({len(self.patterns)} patterns) ===")
        
        print("\nBest by RMSE:")
        for key, info in self.get_best_patterns(5):
            print(f"  RMSE={info['best_rmse']:.4f}, Uses={info['uses']}, Pattern: {key[:50]}...")
        
        print("\nMost Frequent:")
        for key, info in self.get_frequent_patterns(5):
            print(f"  Uses={info['uses']}, RMSE={info['best_rmse']:.4f}, Pattern: {key[:50]}...")


if __name__ == "__main__":
    # Test
    memory = PatternMemory("test_memory.json")
    
    # Add some patterns
    memory.record(['+', 'x', 'C'], 5.0, "(x + C)")
    memory.record(['+', '*', 'C', 'x', 'C'], 1.0, "(C*x + C)")
    memory.record(['+', '*', 'C', 'x', 'C'], 0.5, "(2*x + 3)")  # Same structure, better RMSE
    memory.record(['pow', 'x', '2'], 2.0, "x^2")
    
    memory.summary()
    
    print(f"\nSuggested starting: {memory.suggest_starting_tokens()}")
    
    memory.save()
    print(f"\nSaved to {memory.path}")
    
    # Cleanup
    os.remove("test_memory.json")
