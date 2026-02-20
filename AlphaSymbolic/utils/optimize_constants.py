"""
Constant Optimization Module for AlphaSymbolic.
Uses scipy.optimize to find optimal values for 'C' placeholders.
"""
import numpy as np
from scipy.optimize import minimize
from AlphaSymbolic.core.grammar import ExpressionTree

def optimize_constants(tree, x_data, y_data, method='L-BFGS-B', initial_guess=None):
    """
    Given an ExpressionTree with 'C' placeholders, find optimal constant values.
    
    Args:
        tree: ExpressionTree object
        x_data: numpy array of x values
        y_data: numpy array of target y values
        method: optimization method ('L-BFGS-B', 'SLSQP', 'Nelder-Mead')
        
    Returns:
        dict: mapping of path tuples to optimized constant values
        float: final RMSE
    """
    if not tree.is_valid:
        return {}, float('inf')
    
    # Get positions of all constants
    positions = tree.root.get_constant_positions()
    n_constants = len(positions)
    
    if n_constants == 0:
        # No constants to optimize, just evaluate
        y_pred = tree.evaluate(x_data)
        mse = np.mean((y_pred - y_data)**2)
        return {}, np.sqrt(mse)
    
    def objective(params):
        """Objective function: RMSE given constant values."""
        # Build constants dict
        constants = {tuple(pos): params[i] for i, pos in enumerate(positions)}
        
        # Evaluate
        y_pred = tree.evaluate(x_data, constants=constants)
        
        # Handle invalid predictions
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return 1e10
        
        if not np.all(np.isfinite(y_pred)):
            return 1e9
        
        # Clip huge values to prevent overflow in MSE
        y_pred = np.clip(y_pred, -1e9, 1e9)
        
        mse = np.mean((y_pred - y_data)**2)
        return mse
    
    
    # Check if initial_guess matches n_constants
    if initial_guess is not None:
        if len(initial_guess) != n_constants:
             # Fallback if mismatch
             x0 = np.ones(n_constants)
        else:
             x0 = np.array(initial_guess)
    else:
        # Initial guess: all 1s
        x0 = np.ones(n_constants)
    
    # Bounds: reasonable range for constants
    bounds = [(-1e9, 1e9)] * n_constants
    
    try:
        result = minimize(
            objective,
            x0,
            method=method,
            bounds=bounds if method in ['L-BFGS-B', 'SLSQP'] else None,
            options={'maxiter': 100, 'disp': False}
        )
        
        # Build final constants dict
        optimized_constants = {tuple(pos): result.x[i] for i, pos in enumerate(positions)}
        final_rmse = np.sqrt(result.fun) if result.fun > 0 else 0.0
        
        return optimized_constants, final_rmse
        
    except Exception as e:
        return {}, float('inf')

def convert_and_extract_constants(node, values=None):
    """
    Recursively converts numeric nodes to 'C' and extracts their values.
    Returns: list of initial values.
    """
    if values is None:
        values = []
        
    # Check if node is a number (and not a special constant like pi/e)
    try:
        val = float(node.value)
        # It is a number. Convert to C.
        node.value = 'C'
        values.append(val)
    except:
        pass
        
    for child in node.children:
        convert_and_extract_constants(child, values)
        
    return values


def substitute_constants(infix_str, constants_dict, positions):
    """
    Replace 'C' in the infix string with optimized values.
    Simple approach: replace each C with optimized value.
    """
    # For proper substitution, we'd need to track positions properly
    # This is a simplified version that replaces all C with the first constant
    result = infix_str
    for i, pos in enumerate(positions):
        if tuple(pos) in constants_dict:
            val = constants_dict[tuple(pos)]
            # Format nicely
            if abs(val - round(val)) < 1e-6:
                val_str = str(int(round(val)))
            else:
                val_str = f"{val:.4f}"
            # Replace first occurrence of C
            result = result.replace('C', val_str, 1)
    return result


# Quick test
if __name__ == "__main__":
    # Test: C * x + C should be optimized to fit y = 2*x + 3
    x_test = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    y_test = 2 * x_test + 3  # y = 2x + 3
    
    tokens = ['+', '*', 'C', 'x', 'C']  # C*x + C
    tree = ExpressionTree(tokens)
    
    print(f"Formula structure: {tree.get_infix()}")
    print(f"Target: y = 2x + 3")
    
    constants, rmse = optimize_constants(tree, x_test, y_test)
    print(f"Optimized constants: {constants}")
    print(f"Final RMSE: {rmse:.6f}")
    
    # Verify
    y_pred = tree.evaluate(x_test, constants=constants)
    print(f"Predictions: {y_pred}")
    print(f"Targets: {y_test}")
