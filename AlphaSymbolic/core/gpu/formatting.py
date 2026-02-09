
def format_const(val: float) -> str:
    """
    Format a constant float to string matching CPU engine rules:
    - Integer-like values -> "3"
    - Normal values -> Fixed "1.23456789", trimmed trailing zeros and dot.
    - NEVER use scientific notation (parsers can't handle it).
    """
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    s = f"{val:.8f}"
    s = s.rstrip('0').rstrip('.')
    return s if s else "0"
