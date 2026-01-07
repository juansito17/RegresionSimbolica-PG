
import pandas as pd
import numpy as np
import os

def load_expanded_feynman_subset(csv_path="data/benchmarks/FeynmanEquations.csv", limit=50):
    """
    Loads equations from the Feynman dataset and projects them to 1D.
    Strategies for projection:
    - Fix all variables except the first one to 1.0.
    """
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []
    
    problems = []
    
    # Filter for reasonable complexity (e.g., # variables <= 3 for now to ensure 1D projection makes sense)
    # We can be bolder, but let's start safe.
    # Actually, let's take everything and just project.
    
    count = 0
    for idx, row in df.iterrows():
        if limit is not None and count >= limit:
            break
            
        try:
            row_id = row['Filename']
            formula_raw = str(row['Formula'])
            num_vars = int(row['# variables'])
            
            # Extract variable names
            var_names = []
            for i in range(1, 11):
                v_col = f'v{i}_name'
                if v_col in row and pd.notna(row[v_col]):
                    var_names.append(row[v_col])
            
            # Projection Logic
            # We treat the first variable as 'x' and the rest as constants = 1.0
            # We need to construct a python-evaluable string where other vars are replaced by 1.0
            
            target_var = var_names[0]
            formula_1d = formula_raw
            
            # Replace other variables with "1.0"
            for other_var in var_names[1:]:
                # Simple replace might be dangerous if variable names are substrings of others
                # But Feynman dataset usually uses distinct names like m, v, theta, sigma
                # Better: use a context dict for eval, but we need a string for the model target?
                # Actually, our model needs a target string that uses 'x'.
                pass
                
            # Create a closure-like logic for evaluation
            # We will store the full formula and the fixed context
            fixed_context = {v: 1.0 for v in var_names[1:]}
            
            problems.append({
                "id": row_id,
                "name": f"Feynman {row_id}",
                "original_formula": formula_raw,
                "target_var": target_var,
                "fixed_context": fixed_context,
                "description": f"Projected 1D (varying {target_var}, others fixed to 1.0)"
            })
            count += 1
            
        except Exception as e:
            continue
            
    return problems

def evaluate_projected_formula(formula, target_var, x_val, fixed_context):
    """
    Evaluates the formula with x_val assigned to target_var, and others fixed.
    """
    # math context
    ctx = {
        'exp': np.exp, 'sin': np.sin, 'cos': np.cos, 'sqrt': np.sqrt, 'log': np.log, 
        'pi': np.pi, 'theta': 1.0, 'sigma': 1.0 # Defaults
    }
    
    # Constants from fixed_context
    ctx.update(fixed_context)
    
    # Target variable
    ctx[target_var] = x_val
    
    try:
        return eval(formula, {}, ctx)
    except Exception as e:
        return np.full_like(x_val, np.nan)
