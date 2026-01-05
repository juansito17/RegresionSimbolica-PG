
import json
import os

# Define file paths relative to the script location (AlphaSymbolic/)
# We want to pull C++ files from ../Code/src
CPP_SOURCE_DIR = "../Code/src"
CPP_FILES = [
    'AdvancedFeatures.cpp', 'AdvancedFeatures.h',
    'ExpressionTree.cpp', 'ExpressionTree.h',
    'Fitness.cpp', 'Fitness.h',
    'FitnessGPU.cu', 'FitnessGPU.cuh',
    'GeneticAlgorithm.cpp', 'GeneticAlgorithm.h',
    'GeneticOperators.cpp', 'GeneticOperators.h',
    'main.cpp',
    'TestOperators.cpp',
    'Globals.h' # Embedding the local Globals.h directly
]

# Python files relative to script location (AlphaSymbolic/)
PYTHON_FILES = [
    "core/grammar.py",
    "core/model.py",
    "core/environment.py",
    "core/loss.py",
    "core/gp_bridge.py",
    "core/__init__.py",
    "data/synthetic_data.py",
    "data/benchmark_data.py",
    "data/augmentation.py",
    "data/__init__.py",
    "search/mcts.py",
    "search/beam_search.py",
    "search/hybrid_search.py",
    "search/pareto.py",
    "search/__init__.py",
    "ui/app_core.py",
    "ui/app_search.py",
    "ui/app_training.py",
    "ui/app_benchmark.py",
    "ui/__init__.py",
    "utils/optimize_constants.py",
    "utils/detect_pattern.py",
    "utils/benchmark_runner.py",
    "utils/benchmark_comparison.py",
    "utils/simplify.py",
    "utils/__init__.py",
    "app.py"
]

notebook = {
    "cells": [],
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# 1. Header & Google Drive Mount
install_source = [
    "# AlphaSymbolic - Unified Hybrid System\n",
    "# -------------------------------------\n",
    "# Instructions:\n",
    "# 1. Runtime -> Change runtime type -> T4 GPU\n",
    "# 2. Mount Google Drive to PERSIST models\n",
    "# 3. Run All\n",
    "\n",
    "try:\n",
    "    from google.colab import drive\n",
    "    drive.mount('/content/drive')\n",
    "    os.makedirs('/content/drive/MyDrive/AlphaSymbolic_Models', exist_ok=True)\n",
    "    print(\"✅ Google Drive mounted correctly\")\n",
    "except Exception as e:\n",
    "    print(\"⚠️ Google Drive NOT mounted. Models will be LOST after session ends.\")\n",
    "\n",
    "!nvidia-smi\n",
    "\n",
    "# Install dependencies\n",
    "!pip install gradio torch torchvision torchaudio scipy matplotlib sympy\n",
    "\n",
    "# Create Directory Structure\n",
    "import os\n",
    "os.makedirs('Code/src', exist_ok=True)\n",
    "os.makedirs('Code/build', exist_ok=True)\n",
    "os.makedirs('AlphaSymbolic', exist_ok=True)\n",
    "directories = ['core', 'data', 'search', 'ui', 'utils']\n",
    "for d in directories:\n",
    "    os.makedirs(os.path.join('AlphaSymbolic', d), exist_ok=True)\n"
]

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": install_source
})

# 2. Embed C++ Files
for filename in CPP_FILES:
    local_path = os.path.join(CPP_SOURCE_DIR, filename)
    colab_path = f"Code/src/{filename}"
    
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        cell_source = [f"%%writefile {colab_path}\n", content]
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"collapsed": True}, # Collapse large code blocks
            "outputs": [],
            "source": cell_source
        })
    else:
        print(f"Warning: C++ file {local_path} not found.")

# 3. Create CMakeLists.txt (Using local file content)
cmake_local_path = "../Code/CMakeLists.txt"
if os.path.exists(cmake_local_path):
    with open(cmake_local_path, 'r', encoding='utf-8') as f:
        cmake_content = f.read()
else:
    # Fallback if reading failed
    print("Warning: Local ../Code/CMakeLists.txt not found. Using fallback simplified version.")
    cmake_content = """
cmake_minimum_required(VERSION 3.10)
project(SymbolicRegressionGP)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA)

if(CUDA_FOUND)
    add_definitions(-DUSE_GPU_ACCELERATION_DEFINED_BY_CMAKE)
    enable_language(CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_75") # T4 is sm_75
    set(SOURCE_FILES 
        src/main.cpp 
        src/GeneticAlgorithm.cpp 
        src/ExpressionTree.cpp 
        src/GeneticOperators.cpp
        src/Fitness.cpp
        src/FitnessGPU.cu
        src/AdvancedFeatures.cpp
    )
else()
    message(WARNING "CUDA not found. Compiling for CPU only.")
    set(SOURCE_FILES 
        src/main.cpp 
        src/GeneticAlgorithm.cpp 
        src/ExpressionTree.cpp 
        src/GeneticOperators.cpp
        src/Fitness.cpp
        src/FitnessGPU.cu # Still included but ifdef'd out inside
        src/AdvancedFeatures.cpp
    )
endif()

add_executable(SymbolicRegressionGP ${SOURCE_FILES})

if(CUDA_FOUND)
    set_target_properties(SymbolicRegressionGP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(SymbolicRegressionGP ${CUDA_LIBRARIES})
else()
    target_link_libraries(SymbolicRegressionGP pthread)
endif()
"""

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["%%writefile Code/CMakeLists.txt\n", cmake_content]
})

# 4. Compile C++
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Compile C++ Engine\n",
        "%cd Code\n",
        "!cmake -B build -S . -DCMAKE_BUILD_TYPE=Release\n",
        "!cmake --build build -j $(nproc)\n",
        "import os\n",
        "if not os.path.exists('build/SymbolicRegressionGP') and not os.path.exists('build/Release/SymbolicRegressionGP'):\n",
        "    print('BUILD FAILURE? Binary not found in expected locations. Listing build dir:')\n",
        "    !ls -R build\n",
        "%cd .."
    ]
})

# 5. Embed Python Files
for rel_path in PYTHON_FILES:
    if os.path.exists(rel_path):
        with open(rel_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        colab_path = f"AlphaSymbolic/{rel_path}"
        cell_source = [f"%%writefile {colab_path}\n", content]
        
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"collapsed": True},
            "outputs": [],
            "source": cell_source
        })
    else:
        print(f"Warning: Python file {rel_path} not found.")

# 6. Run App
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Run AlphaSymbolic\n",
        "# The binaries are in ../Code/build/\n",
        "%cd AlphaSymbolic\n",
        "!python app.py\n"
    ]
})

# Save
output_filename = "AlphaSymbolic_Unified_Colab.ipynb"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Unified Notebook generated: {output_filename}")
