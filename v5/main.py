# main.py
from gpu_algorithm import run_evolution
import numpy as np

# Ejemplo de datos (puedes cambiar por los de globals.py si lo prefieres)
X_VALUES_GPU = np.array([8, 9, 10], dtype=np.float64)
TARGETS_GPU = np.array([92, 352, 724], dtype=np.float64)

if __name__ == "__main__":
    print("Symbolic Regression (GPU + Advanced Features)")
    print("===============================" )
    from globals import X_VALUES, TARGETS, TOTAL_POPULATION_SIZE, GENERATIONS
    from genetic_algorithm import GeneticAlgorithm
    ga = GeneticAlgorithm(TARGETS, X_VALUES, TOTAL_POPULATION_SIZE, GENERATIONS)
    best_tree = ga.run()
    print("\nBest Solution Found (GPU + Advanced):")
    from expression_tree import tree_to_string, evaluate_tree
    print("Formula:", tree_to_string(best_tree))
    print(f"Fitness: {ga.overall_best_fitness:.6f}")
    print("Predictions vs Targets:")
    for x, y in zip(X_VALUES, TARGETS):
        pred = evaluate_tree(best_tree, x)
        print(f"  x={x}: Pred={pred:.4f}, Target={y}, Diff={abs(pred-y):.4f}")
