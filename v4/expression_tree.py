# expression_tree.py
from enum import Enum
from typing import Optional, List, Union
import math
from globals import INF

class NodeType(Enum):
    CONSTANT = 1
    VARIABLE = 2
    OPERATOR = 3

class Node:
    def __init__(self, type: NodeType = NodeType.CONSTANT):
        self.type = type
        self.value: float = 0.0  # If type == CONSTANT
        self.op: Optional[str] = None  # If type == OPERATOR: '+', '-', '*', '/', '^'
        self.left: Optional['Node'] = None
        self.right: Optional['Node'] = None

# NodePtr is just Node in Python (no need for shared_ptr)
NodePtr = Node

def evaluate_tree(node: Optional[Node], x: float) -> float:
    if node is None:
        return float('nan')
    if node.type == NodeType.CONSTANT:
        return node.value
    elif node.type == NodeType.VARIABLE:
        return x
    elif node.type == NodeType.OPERATOR:
        leftVal = evaluate_tree(node.left, x)
        rightVal = evaluate_tree(node.right, x)
        if math.isnan(leftVal) or math.isnan(rightVal):
            return float('nan')
        if node.op == '+':
            return leftVal + rightVal
        elif node.op == '-':
            return leftVal - rightVal
        elif node.op == '*':
            return leftVal * rightVal
        elif node.op == '/':
            if abs(rightVal) < 1e-9:
                return INF
            return leftVal / rightVal
        elif node.op == '^':
            if leftVal == 0.0 and rightVal == 0.0:
                return 1.0
            if leftVal < 0.0 and math.floor(rightVal) != rightVal:
                return float('nan')
            try:
                return math.pow(leftVal, rightVal)
            except (OverflowError, ValueError):
                return INF
        else:
            return float('nan')
    else:
        return float('nan')

def tree_to_string(node: Optional[Node]) -> str:
    if node is None:
        return ''
    if node.type == NodeType.CONSTANT:
        val = node.value
        if abs(val - round(val)) < 1e-6:
            return str(int(round(val)))
        else:
            return f"{val:.6f}"
    elif node.type == NodeType.VARIABLE:
        return 'x'
    elif node.type == NodeType.OPERATOR:
        return f"({tree_to_string(node.left)}{node.op}{tree_to_string(node.right)})"
    else:
        return '?'

def tree_size(node: Optional[Node]) -> int:
    if node is None:
        return 0
    if node.type != NodeType.OPERATOR:
        return 1
    return 1 + tree_size(node.left) + tree_size(node.right)

def clone_tree(node: Optional[Node]) -> Optional[Node]:
    if node is None:
        return None
    new_node = Node(node.type)
    new_node.value = node.value
    new_node.op = node.op
    new_node.left = clone_tree(node.left)
    new_node.right = clone_tree(node.right)
    return new_node

def collect_node_ptrs(node: Optional[Node], vec: List[Node]):
    if node is None:
        return
    vec.append(node)
    if node.type == NodeType.OPERATOR:
        if node.left:
            collect_node_ptrs(node.left, vec)
        if node.right:
            collect_node_ptrs(node.right, vec)
