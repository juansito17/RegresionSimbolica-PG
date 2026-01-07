import re
import csv
import pandas as pd
import os

def rescue_logs(log_path='rescue_logs.txt', output_path='learned_formulas_rescued.csv'):
    if not os.path.exists(log_path):
        print(f"Error: No se encuentra el archivo {log_path}")
        print("Por favor, crea este archivo y pega dentro todo el texto de tu consola/terminal.")
        return

    print(f"Leyendo logs de {log_path}...")
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    # Pattern to capture the formula
    # Matches: "Best Formula: <formula_string>"
    # We use non-greedy matching until end of line
    pattern = r"Best Formula:\s*(.+)"
    matches = re.finditer(pattern, content)
    
    data = []
    unique_set = set()
    count = 0
    
    for match in matches:
        formula = match.group(1).strip()
        
        # Basic filtering to remove noise
        if len(formula) < 2: continue
        if "Error" in formula: continue
        
        if formula not in unique_set:
            data.append({
                'formula': formula,
                'length': len(formula),
                'source': 'rescue_script'
            })
            unique_set.add(formula)
            count += 1
            
    print(f"Se encontraron {count} fórmulas únicas.")
    
    if data:
        df = pd.DataFrame(data)
        # Add a complexity metric (length for now)
        df = df.sort_values(by='length')
        
        df.to_csv(output_path, index=False)
        print(f"¡Éxito! Dataset guardado en: {os.path.abspath(output_path)}")
        print("Muestra de las 5 fórmulas más complejas recuperadas:")
        print(df.tail(5)['formula'].values)
    else:
        print("No se encontraron fórmulas válidas. Verifica el formato del archivo de logs.")

if __name__ == "__main__":
    rescue_logs()
