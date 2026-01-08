"""
Target Pattern Detection for AlphaSymbolic.
Analyzes target Y values to detect patterns (polynomial, exponential, periodic, etc.)
and suggests initial search biases.
"""
import numpy as np
from scipy import stats
from scipy.fft import fft
from core.grammar import ExpressionTree

def detect_pattern(x_values, y_values):
    """
    Analyze (x, y) data to detect patterns.
    Returns a dict with pattern type probabilities and suggested operators.
    """
    x = np.array(x_values, dtype=np.float64)
    y = np.array(y_values, dtype=np.float64)
    
    results = {
        'type': 'unknown',
        'confidence': 0.0,
        'suggested_ops': [],
        'details': {}
    }
    
    if len(x) < 3:
        return results

    # Handle Multivariable Input (Skip 1D pattern checks)
    if x.ndim > 1 and x.shape[1] > 1:
        results['type'] = 'multivariable'
        results['confidence'] = 1.0
        results['suggested_ops'] = ['+', '-', '*', 'x', 'C']
        results['details']['multivariable'] = {'num_vars': x.shape[1]}
        return results
    
    scores = {}
    
    # 1. Check for linear pattern (y = ax + b)
    if len(x) >= 2:
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        scores['linear'] = r_value ** 2
        results['details']['linear'] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2
        }
    
    # 2. Check for quadratic pattern (y = ax^2 + bx + c)
    if len(x) >= 3:
        try:
            coeffs = np.polyfit(x, y, 2)
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            scores['quadratic'] = r2
            results['details']['quadratic'] = {
                'coefficients': coeffs.tolist(),
                'r_squared': r2
            }
        except:
            pass
    
    # 3. Check for exponential pattern (y = a * e^(bx))
    if np.all(y > 0):  # Exponential only for positive y
        try:
            log_y = np.log(y)
            slope, intercept, r_value, _, _ = stats.linregress(x, log_y)
            scores['exponential'] = r_value ** 2
            results['details']['exponential'] = {
                'a': np.exp(intercept),
                'b': slope,
                'r_squared': r_value ** 2
            }
        except:
            pass
    
    # 4. Check for periodic/sinusoidal pattern
    if len(y) >= 4:
        try:
            # Simple FFT analysis
            y_centered = y - np.mean(y)
            fft_vals = np.abs(fft(y_centered))
            
            # Check if there's a dominant frequency
            if len(fft_vals) > 1:
                max_idx = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1
                max_power = fft_vals[max_idx]
                total_power = np.sum(fft_vals[1:len(fft_vals)//2])
                
                if total_power > 0:
                    periodicity = max_power / total_power
                    scores['periodic'] = periodicity
                    results['details']['periodic'] = {
                        'dominant_freq_idx': int(max_idx),
                        'periodicity_score': periodicity
                    }
        except:
            pass
    
    # 5. Check for power law (y = a * x^b)
    if np.all(x > 0) and np.all(y > 0):
        try:
            log_x = np.log(x)
            log_y = np.log(y)
            slope, intercept, r_value, _, _ = stats.linregress(log_x, log_y)
            scores['power'] = r_value ** 2
            results['details']['power'] = {
                'a': np.exp(intercept),
                'b': slope,
                'r_squared': r_value ** 2
            }
        except:
            pass
    
    # 6. Check for factorial/gamma pattern (for integer-like x)
    if np.all(x > 0) and np.all(x == np.floor(x)):
        try:
            from scipy.special import gamma
            x_int = x.astype(int)
            y_gamma = gamma(x_int + 1)  # gamma(n+1) = n!
            
            # Simple linear fit between y and gamma
            if not np.any(np.isinf(y_gamma)):
                slope, intercept, r_value, _, _ = stats.linregress(y_gamma, y)
                scores['factorial'] = r_value ** 2
                results['details']['factorial'] = {
                    'r_squared': r_value ** 2
                }
        except:
            pass
    
    # Determine best pattern
    if scores:
        best_pattern = max(scores.items(), key=lambda x: x[1])
        results['type'] = best_pattern[0]
        results['confidence'] = best_pattern[1]
        
        # Suggest operators based on pattern
        op_suggestions = {
            'linear': ['+', '-', '*', 'x', 'C'],
            'quadratic': ['pow', '+', '*', 'x', 'C', '2'],
            'exponential': ['exp', '*', '+', 'x', 'C'],
            'periodic': ['sin', 'cos', '*', '+', 'x', 'C'],
            'power': ['pow', '*', 'x', 'C'],
            'factorial': ['gamma', '*', '+', 'x', 'C']
        }
        results['suggested_ops'] = op_suggestions.get(best_pattern[0], [])
    
    return results


def summarize_pattern(result):
    """Pretty-print pattern detection result."""
    print(f"\n=== Pattern Detection ===")
    print(f"Detected Type: {result['type']} (confidence: {result['confidence']:.2%})")
    print(f"Suggested Operators: {', '.join(result['suggested_ops'])}")
    
    if result['type'] in result['details']:
        print(f"Details: {result['details'][result['type']]}")


if __name__ == "__main__":
    # Test with different patterns
    
    # Linear: y = 2x + 3
    print("\n--- Test: Linear ---")
    x1 = np.linspace(0, 10, 20)
    y1 = 2 * x1 + 3 + np.random.normal(0, 0.1, 20)
    result1 = detect_pattern(x1, y1)
    summarize_pattern(result1)
    
    # Quadratic: y = x^2 + 1
    print("\n--- Test: Quadratic ---")
    x2 = np.linspace(-5, 5, 20)
    y2 = x2**2 + 1
    result2 = detect_pattern(x2, y2)
    summarize_pattern(result2)
    
    # Exponential: y = 2 * e^(0.5x)
    print("\n--- Test: Exponential ---")
    x3 = np.linspace(0, 5, 20)
    y3 = 2 * np.exp(0.5 * x3)
    result3 = detect_pattern(x3, y3)
    summarize_pattern(result3)
    
    # Periodic: y = sin(x)
    print("\n--- Test: Periodic ---")
    x4 = np.linspace(0, 4*np.pi, 50)
    y4 = np.sin(x4)
    result4 = detect_pattern(x4, y4)
    summarize_pattern(result4)
