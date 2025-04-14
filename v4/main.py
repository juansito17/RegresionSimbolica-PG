# main.py
from genetic_algorithm import GeneticAlgorithm
from globals import TARGETS, X_VALUES, TOTAL_POPULATION_SIZE, GENERATIONS, NUM_ISLANDS
from expression_tree import tree_to_string, evaluate_tree

if __name__ == "__main__":
    print("Symbolic Regression using Genetic Programming (Island Model)")
    print("==========================================================")
    print("Target Function Points:")
    for x, y in zip(X_VALUES, TARGETS):
        print(f"  f({x}) = {y}")
    print("----------------------------------------")
    print("Parameters:")
    print(f"  Total Population: {TOTAL_POPULATION_SIZE}")
    print(f"  Generations: {GENERATIONS}")
    print(f"  Islands: {NUM_ISLANDS}")
    print("----------------------------------------")

    ga = GeneticAlgorithm(TARGETS, X_VALUES, TOTAL_POPULATION_SIZE, GENERATIONS, NUM_ISLANDS)
    best_tree = ga.run()

    if best_tree:
        print("\nBest Solution Found:")
        print("Formula:", tree_to_string(best_tree))
        print("Predictions vs Targets:")
        for x, y in zip(X_VALUES, TARGETS):
            pred = evaluate_tree(best_tree, x)
            print(f"  x={x}: Pred={pred:.6f}, Target={y}, Diff={abs(pred-y):.6f}")
    else:
        print("\nFailed to find any valid solution.")
