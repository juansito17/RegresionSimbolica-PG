"""
Algebraic Simplification Module for AlphaSymbolic.
Uses SymPy for symbolic math simplification.
"""
import sympy as sp
from core.grammar import Node, ExpressionTree, OPERATORS

# SymPy symbol for x
x_sym = sp.Symbol('x')

def tree_to_sympy(node):
    """Convert an ExpressionTree Node to a SymPy expression."""
    if node is None:
        return sp.Integer(0)
    
    val = node.value
    
    # Terminals
    if val == 'x':
        return x_sym
    if val == 'pi':
        return sp.pi
    if val == 'e':
        return sp.E
    if val == 'C':
        # Keep C as symbol for now
        return sp.Symbol('C')
    
    # Try numeric
    try:
        return sp.Float(float(val))
    except:
        pass
    
    # Operators
    args = [tree_to_sympy(c) for c in node.children]
    
    if val == '+': return args[0] + args[1]
    if val == '-': return args[0] - args[1]
    if val == '*': return args[0] * args[1]
    if val == '/': return args[0] / args[1]
    if val == 'pow': return sp.Pow(args[0], args[1])
    if val == 'mod': return sp.Mod(args[0], args[1])
    if val == 'sin': return sp.sin(args[0])
    if val == 'cos': return sp.cos(args[0])
    if val == 'tan': return sp.tan(args[0])
    if val == 'exp': return sp.exp(args[0])
    if val == 'log': return sp.log(args[0])
    if val == 'sqrt': return sp.sqrt(args[0])
    if val == 'abs': return sp.Abs(args[0])
    if val == 'floor': return sp.floor(args[0])
    if val == 'ceil': return sp.ceiling(args[0])
    if val == 'gamma': return sp.gamma(args[0])
    if val == 'neg': return -args[0]
    
    return sp.Integer(0)

def sympy_to_infix(expr):
    """Convert SymPy expression back to a readable string."""
    return str(expr)

def simplify_tree(tree):
    """
    Takes an ExpressionTree and returns a simplified infix string.
    """
    if not tree.is_valid:
        return "Invalid"
    
    try:
        sympy_expr = tree_to_sympy(tree.root)
        simplified = sp.simplify(sympy_expr)
        return str(simplified)
    except Exception as e:
        # If simplification fails, return original
        return tree.get_infix()

def simplify_infix(infix_str):
    """
    Takes an infix string and returns a simplified version.
    """
    try:
        expr = sp.sympify(infix_str)
        simplified = sp.simplify(expr)
        return str(simplified)
    except:
        return infix_str

# Quick test
if __name__ == "__main__":
    from core.grammar import ExpressionTree
    
    # Test: x + 0 should simplify to x
    tokens = ['+', 'x', '0']
    tree = ExpressionTree(tokens)
    print(f"Original: {tree.get_infix()}")
    print(f"Simplified: {simplify_tree(tree)}")
    
    # Test: x * 1 should simplify to x
    tokens2 = ['*', 'x', '1']
    tree2 = ExpressionTree(tokens2)
    print(f"Original: {tree2.get_infix()}")
    print(f"Simplified: {simplify_tree(tree2)}")
    
    # Test: x - x should simplify to 0
    tokens3 = ['-', 'x', 'x']
    tree3 = ExpressionTree(tokens3)
    print(f"Original: {tree3.get_infix()}")
    print(f"Simplified: {simplify_tree(tree3)}")
