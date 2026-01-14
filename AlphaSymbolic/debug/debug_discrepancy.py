import numpy as np
import math
from core.grammar import ExpressionTree, OPERATORS
import torch

# Formula from User (Gen 69)
FORMULA_STR = "(((log(gamma(x0)) - x0) + log((sin(abs(sin(cos(x0)))) + x0))) - cos((sqrt(x1) / ((lgamma(x0) - x0) + sin((sin(x0) + sqrt(x0)))))))"

print(f"Testing Formula: {FORMULA_STR}")

# Data (N=27)
n = 27.0
x0 = n
x1 = n % 6
x2 = n % 2
y_true_linear = 2.34907967154122528e17
y_true_log = math.log(y_true_linear)

inputs = {
    'x0': np.array([x0]), 
    'x1': np.array([x1]),
    'x2': np.array([x2])
}

# 1. Evaluate using ExpressionTree (CPU)
try:
    tree = ExpressionTree.from_infix(FORMULA_STR)
    if not tree.is_valid:
        print("ERROR: ExpressionTree failed to parse.")
    else:
        # Note: ExpressionTree evaluates using numpy operators defined in grammar.py
        result = tree.evaluate(inputs)
        y_pred_log = result[0]
        y_pred_linear = math.exp(y_pred_log)
        
        print("\n--- ExpressionTree (CPU) Evaluation ---")
        print(f"Input: x0={x0}")
        print(f"Target (Log): {y_true_log:.6f}")
        print(f"Pred   (Log): {y_pred_log:.6f}")
        print(f"Diff   (Log): {y_pred_log - y_true_log:.6f}")
        print(f"Target (Lin): {y_true_linear:.4e}")
        print(f"Pred   (Lin): {y_pred_linear:.4e}")
        
except Exception as e:
    print(f"Tree Evaluation Error: {e}")

# 2. Manual Verification of parts
t1 = math.lgamma(x0) - x0 # log(gamma(x0)) matches lgamma(x0) NOT lgamma(x0+1) !!
t2 = math.log(math.sin(abs(math.sin(math.cos(x0)))) + x0)
num = math.sqrt(x1)
denom = (math.lgamma(x0) - x0) + math.sin(math.sin(x0) + math.sqrt(x0))
t3 = math.cos(num/denom)

manual_log = t1 + t2 - t3
manual_lin = math.exp(manual_log)

print("\n--- Manual Breakdown ---")
print(f"Term 1 (lgamma(x0)-x0): {t1:.6f}")
print(f"Term 2 (log(...)):      {t2:.6f}")
print(f"Term 3 (cos(...)):      {t3:.6f}")
print(f"Manual Pred (Log):      {manual_log:.6f}")
print(f"Error (Log):            {manual_log - y_true_log:.6f}")

# 3. Hypothesis Check: Does GA mean lgamma(x0+1)?
t1_shifted = math.lgamma(x0 + 1) - x0
manual_log_shifted = t1_shifted + t2 - t3
print(f"\nHypothesis: x0 implies (x0+1) for gamma?")
print(f"Shifted Pred (Log):     {manual_log_shifted:.6f}")
print(f"Shifted Error (Log):    {manual_log_shifted - y_true_log:.6f}")

