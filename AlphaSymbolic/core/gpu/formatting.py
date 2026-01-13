
def format_const(val: float) -> str:
    """
    Format a constant float to string matching CPU engine rules:
    - Integer-like values -> "3"
    - Extreme values (>=1e6 or <=1e-6) -> Scientific "1.23456789e+09"
    - Normal values -> Fixed "1.23456789", trimmed trailing zeros and dot.
    """
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    if abs(val) >= 1e6 or abs(val) <= 1e-6:
        return f"{val:.8e}"
    s = f"{val:.8f}"
    s = s.rstrip('0').rstrip('.')
    return s if s else "0"
