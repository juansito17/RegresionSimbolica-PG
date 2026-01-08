import numpy as np

def normalize_batch(x_list, y_list, target_length=None):
    """Normalize X and Y values to prevent numerical instability.
    Preserves 2D structure for multivariable X: (points, vars).
    """
    normalized_x = []
    normalized_y = []
    
    for x, y in zip(x_list, y_list):
        # Ensure arrays
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).flatten()
        
        # 1. Normalize X per-variable
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            
        x_norm = np.zeros_like(x)
        for i in range(x.shape[1]):
            col = x[:, i]
            c_min, c_max = col.min(), col.max()
            if c_max - c_min > 1e-6:
                x_norm[:, i] = 2 * (col - c_min) / (c_max - c_min) - 1
            else:
                x_norm[:, i] = 0
        
        # 2. Normalize Y
        y_min, y_max = y.min(), y.max()
        if y_max - y_min > 1e-6:
            y_norm = 2 * (y - y_min) / (y_max - y_min) - 1
        else:
            y_norm = np.zeros_like(y)
        
        normalized_x.append(x_norm)
        normalized_y.append(y_norm)
    
    return normalized_x, normalized_y
