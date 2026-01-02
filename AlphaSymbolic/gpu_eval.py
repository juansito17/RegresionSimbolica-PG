"""
GPU Batched Formula Evaluation for AlphaSymbolic.
Evaluates multiple formulas simultaneously on GPU using PyTorch.
"""
import torch
import numpy as np
from grammar import VOCABULARY, OPERATORS, TOKEN_TO_ID, ExpressionTree

# Create operation lookup for vectorized evaluation
OP_FUNCS = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: torch.where(b != 0, a / b, torch.zeros_like(a)),
    'pow': lambda a, b: torch.pow(torch.abs(a) + 1e-8, torch.clamp(b, -10, 10)),
    'mod': lambda a, b: torch.fmod(a, b + 1e-8),
    'sin': lambda a: torch.sin(a),
    'cos': lambda a: torch.cos(a),
    'tan': lambda a: torch.tan(a),
    'exp': lambda a: torch.exp(torch.clamp(a, -100, 100)),
    'log': lambda a: torch.log(torch.abs(a) + 1e-8),
    'sqrt': lambda a: torch.sqrt(torch.abs(a)),
    'abs': lambda a: torch.abs(a),
    'floor': lambda a: torch.floor(a),
    'ceil': lambda a: torch.ceil(a),
    'gamma': lambda a: torch.lgamma(torch.clamp(a, 1e-8, 50)).exp(),  # Approximate
    'neg': lambda a: -a,
}


class GPUEvaluator:
    """Batch evaluator for formulas on GPU."""
    
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def evaluate_single(self, tokens, x_values, constants=None):
        """
        Evaluate a single formula on GPU.
        tokens: list of tokens in prefix notation
        x_values: numpy array or tensor of x values
        constants: dict mapping positions to constant values
        """
        if isinstance(x_values, np.ndarray):
            x_tensor = torch.tensor(x_values, dtype=torch.float32, device=self.device)
        else:
            x_tensor = x_values.to(self.device)
        
        try:
            result = self._eval_prefix(tokens, x_tensor, constants or {})
            return result.cpu().numpy() if isinstance(result, torch.Tensor) else result
        except Exception as e:
            return np.full(len(x_values), np.nan)
    
    def _eval_prefix(self, tokens, x, constants, idx=0, path=None):
        """Recursively evaluate prefix notation on GPU."""
        if path is None:
            path = []
        
        if idx >= len(tokens):
            return torch.zeros_like(x), idx
        
        token = tokens[idx]
        
        # Terminal nodes
        if token == 'x':
            return x, idx + 1
        if token == 'pi':
            return torch.full_like(x, np.pi), idx + 1
        if token == 'e':
            return torch.full_like(x, np.e), idx + 1
        if token == 'C':
            val = constants.get(tuple(path), 1.0)
            return torch.full_like(x, val), idx + 1
        
        # Try numeric constant
        try:
            val = float(token)
            return torch.full_like(x, val), idx + 1
        except:
            pass
        
        # Operators
        if token in OPERATORS:
            arity = OPERATORS[token]
            
            if arity == 1:
                arg, next_idx = self._eval_prefix(tokens, x, constants, idx + 1, path + [0])
                return OP_FUNCS[token](arg), next_idx
            elif arity == 2:
                arg1, mid_idx = self._eval_prefix(tokens, x, constants, idx + 1, path + [0])
                arg2, next_idx = self._eval_prefix(tokens, x, constants, mid_idx, path + [1])
                return OP_FUNCS[token](arg1, arg2), next_idx
        
        return torch.zeros_like(x), idx + 1
    
    def evaluate_batch(self, formulas, x_values, constants_list=None):
        """
        Evaluate multiple formulas on the same x values.
        formulas: list of token lists
        x_values: shared x values
        constants_list: list of constant dicts (one per formula)
        
        Returns: numpy array of shape [num_formulas, num_points]
        """
        if isinstance(x_values, np.ndarray):
            x_tensor = torch.tensor(x_values, dtype=torch.float32, device=self.device)
        else:
            x_tensor = x_values.to(self.device)
        
        results = []
        
        for i, tokens in enumerate(formulas):
            constants = constants_list[i] if constants_list else {}
            try:
                result, _ = self._eval_prefix(tokens, x_tensor, constants, 0, [])
                # Handle potential issues
                result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
                results.append(result)
            except:
                results.append(torch.full_like(x_tensor, np.nan))
        
        # Stack results
        stacked = torch.stack(results, dim=0)
        return stacked.cpu().numpy()
    
    def compute_rmse_batch(self, formulas, x_values, y_target, constants_list=None):
        """
        Compute RMSE for multiple formulas at once.
        Returns: numpy array of RMSEs [num_formulas]
        """
        y_preds = self.evaluate_batch(formulas, x_values, constants_list)
        
        # Compute RMSE for each formula
        y_target_np = np.array(y_target)
        
        rmses = []
        for y_pred in y_preds:
            if np.any(np.isnan(y_pred)):
                rmses.append(float('inf'))
            else:
                rmse = np.sqrt(np.mean((y_pred - y_target_np) ** 2))
                rmses.append(rmse)
        
        return np.array(rmses)


class BatchOptimizer:
    """Optimize constants for multiple formulas in parallel."""
    
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluator = GPUEvaluator(device)
    
    def optimize_batch(self, formulas, x_values, y_target, steps=100, lr=0.1):
        """
        Optimize constants for multiple formulas using gradient descent.
        
        Returns: list of (optimized_constants_dict, final_rmse) tuples
        """
        x_tensor = torch.tensor(x_values, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_target, dtype=torch.float32, device=self.device)
        
        results = []
        
        for tokens in formulas:
            tree = ExpressionTree(tokens)
            if not tree.is_valid:
                results.append(({}, float('inf')))
                continue
            
            # Count constants
            positions = tree.root.get_constant_positions() if tree.root else []
            n_constants = len(positions)
            
            if n_constants == 0:
                # No constants to optimize
                y_pred = self.evaluator.evaluate_single(tokens, x_values)
                rmse = np.sqrt(np.mean((y_pred - y_target) ** 2))
                results.append(({}, rmse))
                continue
            
            # Create trainable parameters
            params = torch.ones(n_constants, requires_grad=True, device=self.device)
            optimizer = torch.optim.Adam([params], lr=lr)
            
            for _ in range(steps):
                optimizer.zero_grad()
                
                # Build constants dict
                constants = {tuple(pos): params[i].item() for i, pos in enumerate(positions)}
                
                # Evaluate
                y_pred, _ = self.evaluator._eval_prefix(tokens, x_tensor, constants, 0, [])
                
                # Loss
                loss = torch.mean((y_pred - y_tensor) ** 2)
                
                if not torch.isfinite(loss):
                    break
                
                loss.backward()
                optimizer.step()
            
            # Final result
            final_constants = {tuple(pos): params[i].item() for i, pos in enumerate(positions)}
            final_rmse = np.sqrt(loss.item()) if torch.isfinite(loss) else float('inf')
            results.append((final_constants, final_rmse))
        
        return results


# Quick test
if __name__ == "__main__":
    evaluator = GPUEvaluator()
    
    # Test single evaluation
    x = np.linspace(-5, 5, 100)
    
    # Test: x^2 + 1
    tokens = ['+', 'pow', 'x', '2', '1']
    result = evaluator.evaluate_single(tokens, x)
    expected = x**2 + 1
    print(f"Single eval error: {np.max(np.abs(result - expected)):.6f}")
    
    # Test batch evaluation
    formulas = [
        ['+', 'x', '1'],          # x + 1
        ['*', '2', 'x'],          # 2 * x
        ['+', 'pow', 'x', '2', '1'],  # x^2 + 1
    ]
    
    results = evaluator.evaluate_batch(formulas, x)
    print(f"Batch results shape: {results.shape}")  # Should be [3, 100]
    
    # Test RMSE batch
    y_target = 2 * x + 3
    rmses = evaluator.compute_rmse_batch(formulas, x, y_target)
    print(f"RMSEs: {rmses}")
    
    print("\nGPU batch evaluation working!")
