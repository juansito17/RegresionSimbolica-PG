
import sys
import os
import torch
import numpy as np

# Añadir el directorio raíz al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.synthetic_data import DataGenerator
from core.grammar import ExpressionTree

def test_multivariable_generation():
    print("=== Test de Generación de Datos Multivariable ===")
    num_vars = 3
    generator = DataGenerator(num_variables=num_vars)
    
    print(f"\nGenerando 10 problemas con num_variables={num_vars}...")
    problems = generator.generate_inverse_batch(batch_size=10)
    
    found_multivariable = 0
    
    for i, prob in enumerate(problems):
        formula = prob['infix']
        x = prob['x']
        y = prob['y']
        
        # Contar cuántas variables distintas aparecen en la cadena
        vars_present = []
        for v_idx in range(num_vars):
            v_name = f"x{v_idx}"
            if v_name in formula:
                vars_present.append(v_name)
        
        is_multi = len(vars_present) > 1
        if is_multi:
            found_multivariable += 1
            
        status = " [MULTIVARIABLE]" if is_multi else " [UNIVARIABLE]"
        print(f"\nProb {i+1}{status}:")
        print(f"  Fórmula: {formula}")
        print(f"  Variables detectadas: {', '.join(vars_present)}")
        print(f"  Shape de X: {x.shape}")
        print(f"  Primeros puntos de datos (N=3):")
        # x is (N, num_vars) if num_variables > 1
        for v_idx in range(num_vars):
            vals = x[:3, v_idx] if len(x.shape) > 1 else (x[:3] if v_idx == 0 else [0,0,0])
            print(f"    x{v_idx}: {vals}")
        print(f"    y: {y[:3]}")

    print("\n" + "="*50)
    print(f"RESULTADO: {found_multivariable}/10 fórmulas son multivariable.")
    print("="*50)
    
    if found_multivariable > 0:
        print("✅ ÉXITO: El generador está produciendo diversidad de variables.")
    else:
        print("❌ FALLO: El generador sigue estancado en univariable.")

if __name__ == "__main__":
    test_multivariable_generation()
