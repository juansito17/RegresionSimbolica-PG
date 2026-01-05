import os
import subprocess
import tempfile
import re
import time
import sys
from typing import List, Optional

# Windows: Disable crash dialog boxes for child processes
if sys.platform == 'win32':
    try:
        import ctypes
        # SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX
        SEM_NOGPFAULTERRORBOX = 0x0002
        SEM_FAILCRITICALERRORS = 0x0001
        SEM_NOOPENFILEERRORBOX = 0x8000
        ctypes.windll.kernel32.SetErrorMode(
            SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX
        )
    except Exception:
        pass  # Silently ignore if ctypes fails

class GPEngine:
    def __init__(self, binary_path=None):
        if binary_path is None:
            # Default location: Code/build/Release/SymbolicRegressionGP.exe
            # Assuming we are in AlphaSymbolic/.. root or similar.
            # Adjust path relative to this file: alphasybolic/core/gp_bridge.py
            # So binary is at ../../Code/build/Release/SymbolicRegressionGP.exe
            # Improved Project Root Detection
            # We look for the "Code" directory by walking up from this file.
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = None
            
            # Walk up up to 5 levels
            d = current_dir
            for _ in range(5):
                if os.path.exists(os.path.join(d, "Code")):
                    project_root = d
                    break
                parent = os.path.dirname(d)
                if parent == d:
                    break
                d = parent
            
            if project_root:
                base_dir = project_root
            else:
                # Fallback
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

            # Define candidates based on OS
            is_windows = os.name == 'nt'
            search_paths = []
            
            if is_windows:
                 search_paths = [
                    os.path.join(base_dir, "Code", "build", "Release", "SymbolicRegressionGP.exe"),
                    os.path.join(base_dir, "Code", "build", "SymbolicRegressionGP.exe"),
                    # Fallbacks
                     os.path.join(base_dir, "Code", "build", "Release", "SymbolicRegressionGP"),
                 ]
            else:
                 # Linux/Mac (Colab) - Prioritize no extension
                 search_paths = [
                    os.path.join(base_dir, "Code", "build", "SymbolicRegressionGP"),
                    os.path.join(base_dir, "Code", "build", "Release", "SymbolicRegressionGP"),
                    # Fallbacks
                    os.path.join(base_dir, "Code", "build", "Release", "SymbolicRegressionGP.exe"),
                    os.path.join(base_dir, "Code", "build", "SymbolicRegressionGP.exe"),
                 ]

            self.binary_path = None
            for p in search_paths:
                if os.path.exists(p):
                    self.binary_path = p
                    break
            
            if self.binary_path is None:
                print(f"[Warning] GP Binary not found. Checked locations:")
                for p in search_paths:
                    print(f" - {p}")
                # Fallback to the most likely one for the current OS
                self.binary_path = search_paths[0]
        else:
            self.binary_path = binary_path

    def run(self, x_values: List[float], y_values: List[float], seeds: List[str] = [], timeout_sec: int = 10) -> Optional[str]:
        """
        Runs the C++ GP Engine with the given data and seeds.
        Returns the best formula found as a string, or None if failed.
        """
        if not os.path.exists(self.binary_path):
            print(f"[Error] GP Binary not found at: {self.binary_path}")
            return None

        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as seed_file, \
             tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as data_file:
            
            # Write Seeds
            for seed in seeds:
                seed_file.write(seed + "\n")
            seed_file_path = seed_file.name
            
            # Write Data
            # Line 1: x1 x2 ...
            # Line 2: y1 y2 ...
            data_file.write(" ".join(map(str, x_values)) + "\n")
            data_file.write(" ".join(map(str, y_values)) + "\n")
            data_file_path = data_file.name

        try:
            # Run Command
            cmd = [self.binary_path, "--seed", seed_file_path, "--data", data_file_path]
            print(f"Running GP Engine: {' '.join(cmd)}")
            
            # Windows-specific: Hide console window and suppress error dialogs
            startupinfo = None
            creationflags = 0
            if os.name == 'nt':  # Windows
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                # CREATE_NO_WINDOW + Don't show error dialogs
                creationflags = subprocess.CREATE_NO_WINDOW | 0x08000000  # CREATE_NO_WINDOW | SEM_NOGPFAULTERRORBOX
            
            start_time = time.time()
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout_sec,
                startupinfo=startupinfo,
                creationflags=creationflags
            )
            
            output = result.stdout
            
            # Parse Output
            # We look for the LAST occurrence of "Formula: ..."
            # Standard formats:
            # "Formula: ((x * x) + 2)"
            # "Final Formula: ..."
            
            best_formula = None
            # Look for formula lines (case-insensitive)
            # Priority: "Final Formula:" > "Formula:" > "Initial best formula:"
            for line in output.splitlines():
                line_lower = line.lower()
                if "formula:" in line_lower:
                    # Extract the part after "formula:" (case-insensitive split)
                    idx = line_lower.find("formula:")
                    if idx != -1:
                        formula_part = line[idx + len("formula:"):].strip()
                        if formula_part:
                            best_formula = formula_part
                            # Keep looking for better matches (Final Formula is best)
                            if "final formula:" in line_lower:
                                break  # Final Formula is the best, stop looking
                        
            print(f"GP Engine finished in {time.time() - start_time:.2f}s")
            
            if best_formula is None:
                print(f"[DEBUG] GP Engine Output (Stdout):\n{output}")
                print(f"[DEBUG] GP Engine Output (Stderr):\n{result.stderr}")
            
            return best_formula

        except subprocess.TimeoutExpired as e:
            print(f"GP Engine timed out after {timeout_sec}s.")
            # Recover output captured so far. 
            # Note: TimeoutExpired.stdout/stderr might be bytes even if text=True was passed to subprocess.run.
            output = e.stdout if e.stdout else ""
            if isinstance(output, bytes):
                output = output.decode('utf-8', errors='ignore')
            
            error_output = e.stderr if e.stderr else ""
            if isinstance(error_output, bytes):
                error_output = error_output.decode('utf-8', errors='ignore')

            best_formula = None
            if output:
                for line in output.splitlines():
                    line_lower = line.lower()
                    if "formula:" in line_lower:
                        idx = line_lower.find("formula:")
                        if idx != -1:
                            formula_part = line[idx + len("formula:"):].strip()
                            if formula_part:
                                best_formula = formula_part
                                if "final formula:" in line_lower:
                                    break
            
            if best_formula:
                print(f"Recovered best formula from timeout: {best_formula}")
                return best_formula
            
            if error_output:
                 print(f"GP Engine Timeout Stderr: {error_output}")
            return None

        except Exception as e:
            print(f"GP Engine failed: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"Stderr: {e.stderr}")
            return None
        finally:
            # Cleanup
            if os.path.exists(seed_file_path):
                os.unlink(seed_file_path)
            if os.path.exists(data_file_path):
                os.unlink(data_file_path)

if __name__ == "__main__":
    # Test
    engine = GPEngine()
    x = [1, 2, 3, 4]
    y = [1+2, 4+2, 9+2, 16+2] # x^2 + 2
    seeds = ["(x * x)", "(x + 2)"]
    
    print("Testing GPEngine...")
    res = engine.run(x, y, seeds)
    print(f"Result: {res}")
