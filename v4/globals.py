# globals.py
# Parámetros y utilidades globales para el algoritmo genético de regresión simbólica

TARGETS = [92, 352, 724]
X_VALUES = [8, 9, 10]

# TOTAL_POPULATION_SIZE, GENERATIONS, etc.
TOTAL_POPULATION_SIZE = 1000
GENERATIONS = 500
MUTATION_RATE = 0.35
STAGNATION_LIMIT = 20
ELITE_PERCENTAGE = 0.20
NUM_ISLANDS = 7
MIGRATION_INTERVAL = 50
MIGRATION_SIZE = 30
MAX_TREE_DEPTH_INITIAL = 7
MAX_TREE_DEPTH_MUTATION = 5
COMPLEXITY_PENALTY_FACTOR = 0.1

INF = float('inf')

import random
_rng = random.Random()
def get_rng():
    return _rng
