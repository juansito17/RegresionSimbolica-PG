import os
import json

# Configuration
PROJECT_ROOT = "."
OUTPUT_FILE = "AlphaSymbolic_Colab.ipynb"
DIRS_TO_INCLUDE = ["core", "data", "search", "ui", "utils"]
MAIN_FILE = "app.py"

def create_notebook():
    cells = []
    
    # 1. Setup Cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install dependencies\n",
            "!pip install gradio torch torchvision torchaudio scipy matplotlib sympy\n",
            "\n",
            "# Create directories\n",
            "!mkdir -p core data search ui utils\n"
        ]
    })
    
    # 2. File Cells
    for folder in DIRS_TO_INCLUDE:
        if not os.path.exists(folder):
            continue
            
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".py") and "__pycache__" not in root:
                    path = os.path.join(root, file)
                    # Convert Windows path to Unix for Colab
                    unix_path = path.replace("\\", "/")
                    
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                    cells.append({
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": [
                            f"%%writefile {unix_path}\n",
                            content
                        ]
                    })
                    
    # 3. App.py Cell (Modified for Colab)
    if os.path.exists(MAIN_FILE):
        with open(MAIN_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Patch for Colab: Enable public sharing, disable local browser
        content = content.replace("share=False", "share=True")
        content = content.replace("inbrowser=True", "inbrowser=False")
        
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"%%writefile {MAIN_FILE}\n",
                content
            ]
        })
        
    # 4. Run Cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Run App\n",
            "print('Starting AlphaSymbolic on Colab GPU...')\n",
            "!python app.py"
        ]
    })
    
    # Notebook Structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    return notebook

if __name__ == "__main__":
    nb = create_notebook()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {OUTPUT_FILE} successfully!")
