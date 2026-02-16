import numpy as np

def normalize_batch(x_list, y_list, target_length=None):
    """Normalize X and Y values to prevent numerical instability.
    Preserves 2D structure for multivariable X: (points, vars).
    """
    normalized_x = []
    normalized_y = []
    
    for x, y in zip(x_list, y_list):
        # Handle dict input (historical DataGenerator format)
        if isinstance(x, dict):
            # Sort keys x0, x1... and convert to 2D array
            keys = sorted(x.keys(), key=lambda k: int(k[1:]) if k[1:].isdigit() else 0)
            x = np.stack([x[k] for k in keys], axis=1)
            
        # Ensure arrays
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).flatten()
        
        # Ensure X is at least 2D (points, vars)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            
        # RAW VALUES - Normalization DISABLED by user request
        # We pass the values as-is (just ensuring consistent shape/type)
        normalized_x.append(x)
        normalized_y.append(y)
    
    return normalized_x, normalized_y
