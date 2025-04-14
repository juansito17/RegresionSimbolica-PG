import numpy as np

NODE_SIZE = 5  # [type, value, op, left, right]
NODE_TYPE_CONST = 0
NODE_TYPE_VAR = 1
NODE_TYPE_OP = 2
OP_MAP = {'+': 0, '-': 1, '*': 2, '/': 3, '^': 4}
OP_MAP_INV = {v: k for k, v in OP_MAP.items()}

# Convierte un árbol recursivo a array plano para GPU
def flatten_tree(tree):
    nodes = []
    def _rec(node):
        if node is None:
            return -1
        idx = len(nodes)
        if hasattr(node, 'type') and hasattr(node, 'value'):
            if node.type.name == 'CONSTANT':
                nodes.append([NODE_TYPE_CONST, node.value, 0, -1, -1])
            elif node.type.name == 'VARIABLE':
                nodes.append([NODE_TYPE_VAR, 0.0, 0, -1, -1])
            elif node.type.name == 'OPERATOR':
                left = _rec(node.left)
                right = _rec(node.right)
                op = OP_MAP.get(node.op, 0)
                nodes.append([NODE_TYPE_OP, 0.0, op, left, right])
        return idx
    _rec(tree)
    return np.array(nodes, dtype=np.float64)

# Convierte un array plano a árbol recursivo (opcional, para debug)
def unflatten_tree(nodes):
    from expression_tree import Node, NodeType
    def _rec(idx):
        if idx < 0:
            return None
        n = nodes[idx]
        ntype = int(n[0])
        if ntype == NODE_TYPE_CONST:
            node = Node(NodeType.CONSTANT)
            node.value = n[1]
            return node
        elif ntype == NODE_TYPE_VAR:
            node = Node(NodeType.VARIABLE)
            return node
        elif ntype == NODE_TYPE_OP:
            node = Node(NodeType.OPERATOR)
            node.op = OP_MAP_INV.get(int(n[2]), '+')
            node.left = _rec(int(n[3]))
            node.right = _rec(int(n[4]))
            return node
    return _rec(0)
