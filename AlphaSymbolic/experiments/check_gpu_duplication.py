
import os
import subprocess
import tempfile
import numpy as np

def run_test():
    # Path to the compiled C++ GP executable
    # Assumes we are running from AlphaSymbolic/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    # Try Release first, then default
    binary_paths = [
        os.path.join(project_root, "Code", "build", "Release", "SymbolicRegressionGP.exe"),
        os.path.join(project_root, "Code", "build", "SymbolicRegressionGP.exe")
    ]
    
    binary_path = None
    for p in binary_paths:
        if os.path.exists(p):
            binary_path = p
            break
            
    if binary_path is None:
        print("Error: C++ GP Binary not found. Please compile the project first.")
        # Try to locate by walking if needed, but above should suffice
        return

    print(f"Using binary: {binary_path}")

    # Generate synthetic data
    # Create a simple dataset: y = x^2
    X = np.linspace(-5, 5, 20)
    Y = X**2

    # Create temp data file
    # We use .txt extension but ensure it is not blocked by gitignore if possible, 
    # but tempfile creates it in %TEMP% usually, so it should be fine.
    # However, for the C++ engine to read it, we pass the absolute path.
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as data_file:
        # Write X line
        data_file.write(" ".join(map(str, X)) + "\n")
        # Write Y line
        data_file.write(" ".join(map(str, Y)) + "\n")
        data_file_path = data_file.name

    try:
        # Run GP
        # We use a small population and few generations for the test
        # Note: Code might enforce MIN_POP_PER_ISLAND
        cmd = [binary_path, "--data", data_file_path, "--pop", "100", "--gens", "5"]
        print(f"Running command: {' '.join(cmd)}")
        print("---------------------------------------------------")
        
        # Use subprocess.call/run without capture_output to let it print to console directly
        # This avoids buffer issues and lets the user see progress
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"GP Execution failed with return code {e.returncode}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if os.path.exists(data_file_path):
            os.remove(data_file_path)

if __name__ == "__main__":
    run_test()
