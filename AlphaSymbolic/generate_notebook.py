
import json
import os

# Define the files we want to embed in the notebook
files_to_embed = [
    "core/grammar.py",
    "core/model.py",
    "core/environment.py",
    "core/__init__.py",
    "data/synthetic_data.py",
    "data/benchmark_data.py",
    "data/augmentation.py",
    "data/__init__.py",
    "search/mcts.py",
    "search/beam_search.py",
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

# 1. Installation Cell
install_source = [
    "# Install dependencies\n",
    "!pip install gradio torch torchvision torchaudio scipy matplotlib sympy\n",
    "\n",
    "# Create directories\n",
    "!mkdir -p core data search ui utils\n"
]
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": install_source
})

# 2. File Embedding Cells
for rel_path in files_to_embed:
    if os.path.exists(rel_path):
        with open(rel_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        source_lines = [f"%%writefile {rel_path}\n"]
        source_lines.append(content)
        
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source_lines
        })
    else:
        print(f"Warning: File {rel_path} not found.")

# 3. Run Cell
run_source = [
    "# Run the application\n",
    "!python app.py\n"
]
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": run_source
})

# Save the notebook
with open("AlphaSymbolic_Colab_Updated.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated: AlphaSymbolic_Colab_Updated.ipynb")
