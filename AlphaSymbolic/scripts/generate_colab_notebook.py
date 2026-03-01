import json
import os
import base64

def generate_notebook():
    # 1. Configuration
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    alpha_dir = os.path.join(repo_root, 'AlphaSymbolic')
    output_file = os.path.join(alpha_dir, 'notebooks', 'AlphaSymbolic_Colab.ipynb')
    
    # Create notebooks directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Files to include (recursive discovery)
    extensions = ('.py', '.cu', '.cpp', '.h', '.cuh', '.txt')
    files_data = {}
    
    for root, dirs, files in os.walk(alpha_dir):
        # Determine relative path from repo root
        rel_root = os.path.relpath(root, repo_root).replace('\\', '/')
        
        # SKIP unnecessary directories to keep the notebook light
        skip_dirs = {
            'AlphaSymbolic/ui', 
            'AlphaSymbolic/tests', 
            'AlphaSymbolic/results', 
            'AlphaSymbolic/models', 
            'AlphaSymbolic/.gradio',
            'AlphaSymbolic/.pytest_cache',
            '.git', '__pycache__', 'build', 'dist', '.egg-info'
        }
        
        # Check if current root or any parent is in skip_dirs
        should_skip = False
        for part in rel_root.split('/'):
            if part in skip_dirs or any(rel_root.startswith(sd) for sd in skip_dirs):
                should_skip = True
                break
        if should_skip:
            continue
            
        for file in files:
            # Include all scripts except the generator itself (to save a bit of space)
            if 'AlphaSymbolic/scripts' in rel_root:
                if file == 'generate_colab_notebook.py':
                    continue

            if file.endswith(extensions):
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, repo_root) # Use relative to repo root (e.g. AlphaSymbolic/...)
                
                try:
                    with open(abs_path, 'rb') as f:
                        content_bytes = f.read()
                    # Use base64 to avoid encoding issues in JSON/Notebook cells
                    files_data[rel_path.replace('\\', '/')] = base64.b64encode(content_bytes).decode('ascii')
                except Exception as e:
                    print(f"Warning: Could not read {abs_path}: {e}")

    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.8.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # --- CELL 1: Header ---
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# AlphaSymbolic GPU Console - Google Colab\n",
            "This notebook allows you to run the **AlphaSymbolic** Genetic Algorithm using Google Colab's GPU acceleration.\n\n",
            "### Instructions\n",
            "1. Go to **Runtime -> Change runtime type** and select **T4 GPU**.\n",
            "2. Run the cells in order.\n"
        ]
    })

    # --- CELL 2: Environment Check & Deps ---
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# @title 1. Setup Environment\n",
            "!nvidia-smi\n",
            "!pip install gymnasium numpy torch==2.1.0 scipy sympy gradio\n"
        ]
    })

    # --- CELL 3: File Creation ---
    file_creation_code = [
        "import os",
        "import json",
        "import base64",
        "",
        "files_to_create = " + json.dumps(files_data, indent=2),
        "",
        "print('Extracting AlphaSymbolic library...')",
        "for filepath, b64_content in files_to_create.items():",
        "    os.makedirs(os.path.dirname(filepath), exist_ok=True)",
        "    with open(filepath, 'wb') as f:",
        "        f.write(base64.b64decode(b64_content))",
        "print('Extraction complete.')"
    ]
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"collapsible": True},
        "outputs": [],
        "source": ["\n".join(file_creation_code)]
    })

    # --- CELL 4: Compile CUDA ---
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# @title 2. Compile CUDA Kernels\n",
            "%cd AlphaSymbolic/core/gpu/cuda\n",
            "!python setup.py install\n",
            "%cd ../../../..\n"
        ]
    })

    # --- CELL 5: Configuration Form ---
    config_update_code = [
        "# @title 3. Configure & Run\n",
        "# @markdown Define los hiperparámetros. T4 Colab GPU usa 16GB VRAM (Ideal 2M a 4M de Tamaño).\n",
        "POP_SIZE = 2000000 # @param {type:'integer'}\n",
        "GENERATIONS = 1000000 # @param {type:'integer'}\n",
        "NUM_ISLANDS = 20 # @param {type:'integer'}\n",
        "USE_LOG_TRANSFORMATION = True # @param {type:'boolean'}\n",
        "COMPLEXITY_PENALTY = 0.01 # @param {type:'number'}\n",
        "MAX_FORMULA_LENGTH = 128 # @param {type:'integer'}\n",
        "NUM_VARIABLES = 3 # @param {type:'integer'}\n",
        "VAR_MOD_X1 = 6 # @param {type:'integer'}\n",
        "VAR_MOD_X2 = 2 # @param {type:'integer'}\n",
        "# @markdown --- \n",
        "# @markdown ### Operator Selection\n",
        "USE_OP_POW = True # @param {type:'boolean'}\n",
        "USE_OP_LOG = True # @param {type:'boolean'}\n",
        "USE_OP_EXP = True # @param {type:'boolean'}\n",
        "USE_OP_FACT = True # @param {type:'boolean'}\n",
        "USE_OP_GAMMA = True # @param {type:'boolean'}\n",
        "USE_OP_SQRT = True # @param {type:'boolean'}\n",
        "USE_OP_SIN = False # @param {type:'boolean'}\n",
        "USE_OP_COS = False # @param {type:'boolean'}\n",
        "\n",
        "INITIAL_FORMULA_STRING = \"((lgamma(x0)-(x0+fact(-8.19257164)))+sqrt(((x0+fact(((lgamma(3)*lgamma(((x0-1)-sqrt(2))))/((fact(x1)**(-(x2)))-2))))+sqrt(((fact(pi)+(((25**(-(x2)))+((log((-(-11.09804344)))-sqrt(x1))+10))/lgamma((x0-fact((((5**(fact((x1/5))**3))**(x1**(2-x1)))/(exp(e)-x0)))))))+(x0+fact(((6**(pi**(-(x2))))/((exp(pi)+(((e+sqrt(e))**(x1-(x0-(lgamma(x0)-(x0+fact(((lgamma(x0)-x0)/(fact(((exp(3)*(2**(-(x2))))/(((x1/3)-2)-2)))-2))))))))-x0))-x0)))))))))\" # @param {type:'string'}\n",
        "USE_INITIAL_FORMULA = False # @param {type:'boolean'}\n",
        "\n",
        "config_path = 'AlphaSymbolic/core/gpu/config.py'\n",
        "with open(config_path, 'r') as f:\n",
        "    lines = f.readlines()\n",
        "\n",
        "new_lines = []\n",
        "for line in lines:\n",
        "    s_line = line.strip()\n",
        "    if s_line.startswith('POP_SIZE ='):\n",
        "        new_lines.append(f'    POP_SIZE = {POP_SIZE}\\n')\n",
        "    elif s_line.startswith('GENERATIONS ='):\n",
        "        new_lines.append(f'    GENERATIONS = {GENERATIONS}\\n')\n",
        "    elif s_line.startswith('NUM_ISLANDS ='):\n",
        "        new_lines.append(f'    NUM_ISLANDS = {NUM_ISLANDS}\\n')\n",
        "    elif s_line.startswith('USE_LOG_TRANSFORMATION ='):\n",
        "        new_lines.append(f'    USE_LOG_TRANSFORMATION = {USE_LOG_TRANSFORMATION}\\n')\n",
        "    elif s_line.startswith('COMPLEXITY_PENALTY ='):\n",
        "        new_lines.append(f'    COMPLEXITY_PENALTY = {COMPLEXITY_PENALTY}\\n')\n",
        "    elif s_line.startswith('MAX_FORMULA_LENGTH ='):\n",
        "        new_lines.append(f'    MAX_FORMULA_LENGTH = {MAX_FORMULA_LENGTH}\\n')\n",
        "    elif s_line.startswith('VAR_MOD_X1 ='):\n",
        "        new_lines.append(f'    VAR_MOD_X1 = {VAR_MOD_X1}\\n')\n",
        "    elif s_line.startswith('VAR_MOD_X2 ='):\n",
        "        new_lines.append(f'    VAR_MOD_X2 = {VAR_MOD_X2}\\n')\n",
        "    elif s_line.startswith('USE_OP_POW ='):\n",
        "        new_lines.append(f'    USE_OP_POW = {USE_OP_POW}\\n')\n",
        "    elif s_line.startswith('USE_OP_LOG ='):\n",
        "        new_lines.append(f'    USE_OP_LOG = {USE_OP_LOG}\\n')\n",
        "    elif s_line.startswith('USE_OP_EXP ='):\n",
        "        new_lines.append(f'    USE_OP_EXP = {USE_OP_EXP}\\n')\n",
        "    elif s_line.startswith('USE_OP_FACT ='):\n",
        "        new_lines.append(f'    USE_OP_FACT = {USE_OP_FACT}\\n')\n",
        "    elif s_line.startswith('USE_OP_GAMMA ='):\n",
        "        new_lines.append(f'    USE_OP_GAMMA = {USE_OP_GAMMA}\\n')\n",
        "    elif s_line.startswith('USE_OP_SQRT ='):\n",
        "        new_lines.append(f'    USE_OP_SQRT = {USE_OP_SQRT}\\n')\n",
        "    elif s_line.startswith('USE_OP_SIN ='):\n",
        "        new_lines.append(f'    USE_OP_SIN = {USE_OP_SIN}\\n')\n",
        "    elif s_line.startswith('USE_OP_COS ='):\n",
        "        new_lines.append(f'    USE_OP_COS = {USE_OP_COS}\\n')\n",
        "    elif s_line.startswith('INITIAL_FORMULA_STRING ='):\n",
        "        new_lines.append(f'    INITIAL_FORMULA_STRING = \"{INITIAL_FORMULA_STRING}\"\\n')\n",
        "    elif s_line.startswith('USE_INITIAL_FORMULA ='):\n",
        "        new_lines.append(f'    USE_INITIAL_FORMULA = {USE_INITIAL_FORMULA}\\n')\n",
        "    else:\n",
        "        new_lines.append(line)\n",
        "\n",
        "with open(config_path, 'w') as f:\n",
        "    f.writelines(new_lines)\n",
        "\n",
        "print('Config updated.')\n",
        "\n",
        "# Update run_gpu_console.py num_variables if needed\n",
        "console_script = 'AlphaSymbolic/scripts/run_gpu_console.py'\n",
        "with open(console_script, 'r') as f:\n",
        "    script_content = f.read()\n",
        "\n",
        "import re\n",
        "script_content = re.sub(r'TensorGeneticEngine\\(num_variables=\\d+', f'TensorGeneticEngine(num_variables={NUM_VARIABLES}', script_content)\n",
        "\n",
        "with open(console_script, 'w') as f:\n",
        "    f.write(script_content)\n",
        "print('run_gpu_console.py updated.')\n",
        "\n",
        "!python AlphaSymbolic/scripts/run_gpu_console.py"
    ]

    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["\n".join(config_update_code)]
    })

    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Notebook created at: {output_file}")

if __name__ == "__main__":
    generate_notebook()
