import json
import os

source_dir = 'src'
files = [
    'AdvancedFeatures.cpp', 'AdvancedFeatures.h',
    'ExpressionTree.cpp', 'ExpressionTree.h',
    'Fitness.cpp', 'Fitness.h',
    'FitnessGPU.cu', 'FitnessGPU.cuh',
    'GeneticAlgorithm.cpp', 'GeneticAlgorithm.h',
    'GeneticOperators.cpp', 'GeneticOperators.h',
    'main.cpp'
]

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.8.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# 1. Header
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Symbolic Regression GP (CUDA Enabled)\n",
        "## Instructions\n",
        "1. Go to **Runtime -> Change runtime type** and select **T4 GPU** (or any available GPU).\n",
        "2. Run all cells to compile and execute the project.\n"
    ]
})

# 2. Check GPU
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["!nvidia-smi"]
})

# 3. Create directories
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "os.makedirs('src', exist_ok=True)\n",
        "os.makedirs('build', exist_ok=True)"
    ]
})

# 4. Create Files (One Cell for All)
files_data = {}

# Read CMakeLists.txt
try:
    content = open('CMakeLists.txt', 'r', encoding='utf-8').read()
    files_data['CMakeLists.txt'] = content
except Exception as e:
    print(f"Error reading CMakeLists.txt: {e}")

# Read Source Files
for filename in files:
    path = os.path.join(source_dir, filename)
    if os.path.exists(path):
        try:
            content = open(path, 'r', encoding='utf-8').read()
            files_data[f"src/{filename}"] = content
        except Exception as e:
             print(f"Error reading {filename}: {e}")
    else:
        print(f"Warning: {filename} not found.")

# Note: FitnessGPU.cu is read from disk above (not hardcoded) to include all latest functions

# Create the Python code for the cell
# We use json.dumps to handle escaping of strings safely
file_creation_code = [
    "import os",
    "",
    "files_to_create = " + json.dumps(files_data, indent=2),
    "",
    "for filepath, content in files_to_create.items():",
    "    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None",
    "    with open(filepath, 'w', encoding='utf-8') as f:",
    "        f.write(content)",
    "    print(f'Created: {filepath}')"
]

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["\n".join(file_creation_code)]
})

# 5. Build
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Configuration & Compilation"]
})

# Cell to define parameters and write Globals.h
globals_code = [
    "# @title Algorithm Parameters",
    "# @markdown Modify these values to tune the genetic algorithm.",
    "",
    "import os",
    "",
    "# Problem Data",
    "RAW_TARGETS_STR = \"2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884\" # @param {type:\"string\"}",
    "X_VALUES_STR = \"4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20\" # @param {type:\"string\"}",
    "USE_LOG_TRANSFORMATION = True # @param {type:\"boolean\"}",
    "",
    "# Algorithm Settings",
    "USE_GPU_ACCELERATION = True # @param {type:\"boolean\"}",
    "TOTAL_POPULATION_SIZE = 50000 # @param {type:\"integer\"}",
    "GENERATIONS = 500000 # @param {type:\"integer\"}",
    "NUM_ISLANDS = 10 # @param {type:\"integer\"}",
    "MIGRATION_INTERVAL = 100 # @param {type:\"integer\"}",
    "MIGRATION_SIZE = 50 # @param {type:\"integer\"}",
    "",
    "# Genetic Operators",
    "BASE_MUTATION_RATE = 0.30 # @param {type:\"number\"}",
    "BASE_ELITE_PERCENTAGE = 0.15 # @param {type:\"number\"}",
    "",
    "# Operator Selection",
    "USE_OP_PLUS = True # @param {type:\"boolean\"}",
    "USE_OP_MINUS = True # @param {type:\"boolean\"}",
    "USE_OP_MULT = True # @param {type:\"boolean\"}",
    "USE_OP_DIV = True # @param {type:\"boolean\"}",
    "USE_OP_POW = True # @param {type:\"boolean\"}",
    "USE_OP_MOD = True # @param {type:\"boolean\"}",
    "USE_OP_SIN = True # @param {type:\"boolean\"}",
    "USE_OP_COS = True # @param {type:\"boolean\"}",
    "USE_OP_LOG = True # @param {type:\"boolean\"}",
    "USE_OP_EXP = True # @param {type:\"boolean\"}",
    "USE_OP_FACT = True # @param {type:\"boolean\"}",
    "USE_OP_FLOOR = True # @param {type:\"boolean\"}",
    "USE_OP_GAMMA = True # @param {type:\"boolean\"}",
    "",
    "# Initial Formula Injection",
    "USE_INITIAL_FORMULA = True # @param {type:\"boolean\"}",
    "INITIAL_FORMULA_STRING = \"(g(x)-((x*909613)/1000000))+0.24423\" # @param {type:\"string\"}",
    "",
    "# Optimization Settings",
    "FORCE_INTEGER_CONSTANTS = False # @param {type:\"boolean\"}",
    "USE_SIMPLIFICATION = True # @param {type:\"boolean\"}",
    "USE_ISLAND_CATACLYSM = True # @param {type:\"boolean\"}",
    "USE_LEXICASE_SELECTION = True # @param {type:\"boolean\"}",
    "",
    "# Weighted Fitness",
    "USE_WEIGHTED_FITNESS = True # @param {type:\"boolean\"}",
    "WEIGHTED_FITNESS_EXPONENT = 0.25 # @param {type:\"number\"}",
    "",
    "# GPU Settings",
    "FORCE_CPU_MODE = False # @param {type:\"boolean\"}",
    "",
    "# Construct Globals.h content",
    "globals_content = f\"\"\"",
    "#ifndef GLOBALS_H",
    "#define GLOBALS_H",
    "",
    "#include <vector>",
    "#include <random>",
    "#include <string>",
    "#include <limits>",
    "#include <cmath>",
    "",
    "// Data",
    "const std::vector<double> RAW_TARGETS = {{ {RAW_TARGETS_STR} }};",
    "const std::vector<double> X_VALUES = {{ {X_VALUES_STR} }};",
    "const bool USE_LOG_TRANSFORMATION = {'true' if USE_LOG_TRANSFORMATION else 'false'};",
    "",
    "// GPU Settings",
    "const bool FORCE_CPU_MODE = {'true' if FORCE_CPU_MODE else 'false'};",
    "#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE",
    "const bool USE_GPU_ACCELERATION = !FORCE_CPU_MODE;",
    "#else",
    "const bool USE_GPU_ACCELERATION = false;",
    "#endif",
    "",
    "// Algorithm Parameters",
    "const int TOTAL_POPULATION_SIZE = {TOTAL_POPULATION_SIZE};",
    "const int GENERATIONS = {GENERATIONS};",
    "const int NUM_ISLANDS = {NUM_ISLANDS};",
    "const int MIN_POP_PER_ISLAND = 10;",
    "",
    "// Migration",
    "const int MIGRATION_INTERVAL = {MIGRATION_INTERVAL};",
    "const int MIGRATION_SIZE = {MIGRATION_SIZE};",
    "",
    "// Initial Formula",
    "const bool USE_INITIAL_FORMULA = {'true' if USE_INITIAL_FORMULA else 'false'};",
    "const std::string INITIAL_FORMULA_STRING = \"{INITIAL_FORMULA_STRING}\";",
    "",
    "// Genetic Operators",
    "const double BASE_MUTATION_RATE = {BASE_MUTATION_RATE};",
    "const double BASE_ELITE_PERCENTAGE = {BASE_ELITE_PERCENTAGE};",
    "const double DEFAULT_CROSSOVER_RATE = 0.85;",
    "const int DEFAULT_TOURNAMENT_SIZE = 30;",
    "const int MAX_TREE_DEPTH_MUTATION = 8;",
    "",
    "// Operator Configuration",
    "const bool USE_OP_PLUS     = {'true' if USE_OP_PLUS else 'false'};",
    "const bool USE_OP_MINUS    = {'true' if USE_OP_MINUS else 'false'};",
    "const bool USE_OP_MULT     = {'true' if USE_OP_MULT else 'false'};",
    "const bool USE_OP_DIV      = {'true' if USE_OP_DIV else 'false'};",
    "const bool USE_OP_POW      = {'true' if USE_OP_POW else 'false'};",
    "const bool USE_OP_MOD      = {'true' if USE_OP_MOD else 'false'};",
    "const bool USE_OP_SIN      = {'true' if USE_OP_SIN else 'false'};",
    "const bool USE_OP_COS      = {'true' if USE_OP_COS else 'false'};",
    "const bool USE_OP_LOG      = {'true' if USE_OP_LOG else 'false'};",
    "const bool USE_OP_EXP      = {'true' if USE_OP_EXP else 'false'};",
    "const bool USE_OP_FACT     = {'true' if USE_OP_FACT else 'false'};",
    "const bool USE_OP_FLOOR    = {'true' if USE_OP_FLOOR else 'false'};",
    "const bool USE_OP_GAMMA    = {'true' if USE_OP_GAMMA else 'false'};",
    "",
    "// Tree Generation",
    "const int MAX_TREE_DEPTH_INITIAL = 8;",
    "const double TERMINAL_VS_VARIABLE_PROB = 0.75;",
    "const double CONSTANT_MIN_VALUE = -10.0;",
    "const double CONSTANT_MAX_VALUE = 10.0;",
    "const int CONSTANT_INT_MIN_VALUE = -10;",
    "const int CONSTANT_INT_MAX_VALUE = 10;",
    "// Order: +, -, *, /, ^, %, s, c, l, e, !, _, g",
    "const std::vector<double> OPERATOR_WEIGHTS = {{",
    "    0.10 * (USE_OP_PLUS  ? 1.0 : 0.0),",
    "    0.15 * (USE_OP_MINUS ? 1.0 : 0.0),",
    "    0.10 * (USE_OP_MULT  ? 1.0 : 0.0),",
    "    0.10 * (USE_OP_DIV   ? 1.0 : 0.0),",
    "    0.05 * (USE_OP_POW   ? 1.0 : 0.0),",
    "    0.01 * (USE_OP_MOD   ? 1.0 : 0.0),",
    "    0.01 * (USE_OP_SIN   ? 1.0 : 0.0),",
    "    0.01 * (USE_OP_COS   ? 1.0 : 0.0),",
    "    0.15 * (USE_OP_LOG   ? 1.0 : 0.0),",
    "    0.02 * (USE_OP_EXP   ? 1.0 : 0.0),",
    "    0.05 * (USE_OP_FACT  ? 1.0 : 0.0),",
    "    0.05 * (USE_OP_FLOOR ? 1.0 : 0.0),",
    "    0.20 * (USE_OP_GAMMA ? 1.0 : 0.0)",
    "}};",
    "",
    "// Fitness & Other",
    "const double COMPLEXITY_PENALTY_FACTOR = 0.05;",
    "const bool USE_RMSE_FITNESS = true;",
    "const double FITNESS_ORIGINAL_POWER = 1.3;",
    "const double FITNESS_PRECISION_THRESHOLD = 0.001;",
    "const double FITNESS_PRECISION_BONUS = 0.0001;",
    "const double FITNESS_EQUALITY_TOLERANCE = 1e-9;",

    "const double EXACT_SOLUTION_THRESHOLD = 1e-8;",
    "",
    "// Weighted Fitness",
    "const bool USE_WEIGHTED_FITNESS = {'true' if USE_WEIGHTED_FITNESS else 'false'};",
    "const double WEIGHTED_FITNESS_EXPONENT = {WEIGHTED_FITNESS_EXPONENT};",
    "",
    "// Advanced Features",
    "const int STAGNATION_LIMIT_ISLAND = 50;",
    "const int GLOBAL_STAGNATION_LIMIT = 5000;",
    "const double STAGNATION_RANDOM_INJECT_PERCENT = 0.1;",
    "const int PARAM_MUTATE_INTERVAL = 50;",
    "const double PATTERN_RECORD_FITNESS_THRESHOLD = 10.0;",
    "const int PATTERN_MEM_MIN_USES = 3;",
    "const int PATTERN_INJECT_INTERVAL = 10;",
    "const double PATTERN_INJECT_PERCENT = 0.05;",

    "const size_t PARETO_MAX_FRONT_SIZE = 50;",
    "const double SIMPLIFY_NEAR_ZERO_TOLERANCE = 1e-9;",
    "const double SIMPLIFY_NEAR_ONE_TOLERANCE = 1e-9;",
    "const int LOCAL_SEARCH_ATTEMPTS = 30;",
    "const bool USE_SIMPLIFICATION = {'true' if USE_SIMPLIFICATION else 'false'};",
    "const bool USE_ISLAND_CATACLYSM = {'true' if USE_ISLAND_CATACLYSM else 'false'};",
    "const bool USE_LEXICASE_SELECTION = {'true' if USE_LEXICASE_SELECTION else 'false'};",
    "const int PROGRESS_REPORT_INTERVAL = 100;",
    "const bool FORCE_INTEGER_CONSTANTS = {'true' if FORCE_INTEGER_CONSTANTS else 'false'};",
    "",
    "#include <random>",
    "const double MUTATE_INSERT_CONST_PROB = 0.6;",
    "const int MUTATE_INSERT_CONST_INT_MIN = 1;",
    "const int MUTATE_INSERT_CONST_INT_MAX = 5;",
    "const double MUTATE_INSERT_CONST_FLOAT_MIN = 0.5;",
    "const double MUTATE_INSERT_CONST_FLOAT_MAX = 5.0;",
    "",
    "std::mt19937& get_rng();",
    "const double INF = std::numeric_limits<double>::infinity();",
    "",
    "#endif // GLOBALS_H",
    "\"\"\"",
    "",
    "with open('src/Globals.h', 'w') as f:",
    "    f.write(globals_content)",
    "print(\"Globals.h updated with new parameters.\")"
]

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["\n".join(globals_code)]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!cmake -B build -S . -DCMAKE_BUILD_TYPE=Release\n",
        "!cmake --build build -j $(nproc)"
    ]
})

# 7. Run
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Execution"]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["!./build/SymbolicRegressionGP"]
})

output_file = 'GoogleColab_Project.ipynb'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook created successfully: {output_file}")
