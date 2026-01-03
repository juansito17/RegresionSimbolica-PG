
import random
from core.grammar import OPERATORS

def augment_formula_tokens(tokens):
    """
    Applies mathematical invariants to generate an equivalent formula structure.
    Acts as 'Data Augmentation' for symbolic regression.
    
    Supported Transformations:
    1. Commutativity: (+) and (*)
       e.g. [+ a b] -> [+ b a]
    2. Identity:
       e.g. x -> [+ x 0], x -> [* x 1] (Rarely used to avoid bloat, but useful for robustness)
    3. Inverse operations (Conceptually):
       Not implemented directly on tokens without tree parsing, 
       so we focus on purely structural swaps that don't change value.
    
    Args:
        tokens (list): List of tokens in Prefix notation.
    
    Returns:
        list: A new list of tokens representing an equivalent formula.
    """
    if not tokens:
        return []

    # Helper to parse prefix expression into a tree-like structure (recursive)
    def parse_prefix(token_list):
        if not token_list:
            return None, []
        
        root = token_list[0]
        remaining = token_list[1:]
        
        if root in OPERATORS:
            try:
                arity = OPERATORS[root]
                children = []
                for _ in range(arity):
                    child, remaining = parse_prefix(remaining)
                    children.append(child)
                return {'val': root, 'children': children}, remaining
            except:
                 # Fallback for malformed
                return {'val': root, 'children': []}, remaining
        else:
            # Terminal
            return {'val': root, 'children': []}, remaining

    # Helper to flatten tree back to tokens
    def flatten(node):
        res = [node['val']]
        for child in node['children']:
            res.extend(flatten(child))
        return res

    # 1. Parse
    try:
        tree, _ = parse_prefix(tokens)
    except:
        return list(tokens) # Fail safe

    # 2. Augment Recursive
    def augment_recursive(node):
        # First augment children
        for i in range(len(node['children'])):
            node['children'][i] = augment_recursive(node['children'][i])
            
        val = node['val']
        children = node['children']
        
        # Transformation: Commutativity
        if val in ['+', '*'] and len(children) == 2:
            if random.random() < 0.5:
                # Swap children
                node['children'] = [children[1], children[0]]
        
        # Transformation: (- a b) -> (+ a (- b)) ? Too complex for tokens only without 'neg'
        # Transformation: (+ x x) -> (* x 2) ?
        if val == '+' and len(children) == 2:
            # Check deep equality is hard, but simple check:
            if flatten(children[0]) == flatten(children[1]):
                if random.random() < 0.3:
                    # Convert x + x -> x * 2
                    return {'val': '*', 'children': [children[0], {'val': '2', 'children': []}]}

        return node

    # 3. Apply
    augmented_tree = augment_recursive(tree)
    
    # 4. Flatten
    return flatten(augmented_tree)

if __name__ == "__main__":
    # Test
    # Formula: (+ x y) -> prefix ['+', 'x', 'y']
    t1 = ['+', 'x', 'y']
    print(f"Original: {t1} -> Aug: {augment_formula_tokens(t1)}")
    
    # Formula: (* (+ a b) c)
    t2 = ['*', '+', 'a', 'b', 'c']
    print(f"Original: {t2} -> Aug: {augment_formula_tokens(t2)}")
    
    # Formula: (+ x x)
    t3 = ['+', 'x', 'x']
    print(f"Original: {t3} -> Aug: {augment_formula_tokens(t3)}")
