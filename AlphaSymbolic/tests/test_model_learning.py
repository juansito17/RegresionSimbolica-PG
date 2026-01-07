"""
Test para validar que el modelo está aprendiendo.
Ejecutar: python test_model_learning.py

Este script prueba el modelo en fórmulas conocidas y mide:
1. Cuántas resuelve correctamente (RMSE < 0.1)
2. Qué tan cerca están las predicciones de la respuesta

Deberías ver mejora después de más entrenamiento.
"""

import sys
import os
import numpy as np
import torch

# Agregar path del proyecto
import pathlib
PROJECT_ROOT = str(pathlib.Path(__file__).parent.absolute())
sys.path.insert(0, PROJECT_ROOT)

from core.model import AlphaSymbolicModel
from core.grammar import VOCABULARY, TOKEN_TO_ID, ExpressionTree
from search.mcts import MCTS
from search.beam_search import BeamSearch

# Fórmulas de prueba - desde simples hasta complejas
TEST_FORMULAS = [
    # Stage 0: Arithmetic
    ("x + 1", lambda x: x + 1),
    ("x * 2", lambda x: x * 2),
    ("x + x", lambda x: x + x),
    ("x - 1", lambda x: x - 1),
    
    # Stage 1: Polynomials
    ("x * x", lambda x: x * x),
    ("x^2 + 1", lambda x: x**2 + 1),
    ("x^2 + x", lambda x: x**2 + x),
    ("2*x + 3", lambda x: 2*x + 3),
    
    # Stage 2: Powers
    ("x^3", lambda x: x**3),
    ("sqrt(x)", lambda x: np.sqrt(np.abs(x) + 0.1)),
    
    # Stage 3: Trigonometry
    ("sin(x)", lambda x: np.sin(x)),
    ("cos(x)", lambda x: np.cos(x)),
]


def load_model():
    """Carga el modelo guardado usando la misma lógica que la UI."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model presets (same as app_core.py)
    MODEL_PRESETS = {
        'lite': {'d_model': 128, 'nhead': 4, 'num_encoder_layers': 3, 'num_decoder_layers': 3},
        'pro': {'d_model': 256, 'nhead': 8, 'num_encoder_layers': 6, 'num_decoder_layers': 6}
    }
    
    # Try both presets
    for preset in ['pro', 'lite']:
        filename = f"alpha_symbolic_model_{preset}.pth"
        if os.path.exists(filename):
            print(f"📂 Cargando modelo: {filename}")
            
            try:
                state_dict = torch.load(filename, map_location=device, weights_only=True)
                
                # Check for NaNs
                has_nans = False
                for k, v in state_dict.items():
                    if torch.isnan(v).any() or torch.isinf(v).any():
                        has_nans = True
                        break
                
                if has_nans:
                    print(f"⚠️ Modelo corrupto (NaNs). Saltando.")
                    continue
                
                # Create model with correct config
                config = MODEL_PRESETS[preset]
                VOCAB_SIZE = len(VOCABULARY)
                
                model = AlphaSymbolicModel(
                    vocab_size=VOCAB_SIZE + 1,  # +1 for SOS token
                    d_model=config['d_model'],
                    nhead=config['nhead'],
                    num_encoder_layers=config['num_encoder_layers'],
                    num_decoder_layers=config['num_decoder_layers']
                ).to(device)
                
                model.load_state_dict(state_dict)
                model.eval()
                print(f"   ✓ Modelo {preset.upper()} cargado")
                return model, device
                
            except Exception as e:
                print(f"⚠️ Error cargando {filename}: {e}")
                continue
    
    print("⚠️ No se encontró modelo guardado.")
    return None, device


def optimize_constants(tokens, x_data, y_data):
    """Optimiza las constantes 'C' en una fórmula para ajustar a los datos."""
    from scipy.optimize import minimize
    
    # Contar cuántas 'C' hay
    c_count = tokens.count('C')
    if c_count == 0:
        # No hay constantes para optimizar
        tree = ExpressionTree(tokens)
        if tree.is_valid:
            y_pred = tree.evaluate(x_data)
            if y_pred is not None:
                rmse = np.sqrt(np.mean((y_pred - y_data)**2))
                return tokens, rmse
        return tokens, float('inf')
    
    def objective(c_values):
        # Reemplazar cada 'C' con su valor
        new_tokens = []
        c_idx = 0
        for t in tokens:
            if t == 'C':
                new_tokens.append(str(c_values[c_idx]))
                c_idx += 1
            else:
                new_tokens.append(t)
        
        tree = ExpressionTree(new_tokens)
        if not tree.is_valid:
            return 1e10
        
        try:
            y_pred = tree.evaluate(x_data)
            if y_pred is None or len(y_pred) != len(y_data):
                return 1e10
            return np.sqrt(np.mean((y_pred - y_data)**2))
        except:
            return 1e10
    
    # Optimizar con varios puntos de inicio
    best_rmse = float('inf')
    best_c = [1.0] * c_count
    
    for init in [[1.0] * c_count, [0.0] * c_count, [2.0] * c_count, [-1.0] * c_count]:
        try:
            result = minimize(objective, init, method='Nelder-Mead', 
                            options={'maxiter': 100, 'xatol': 0.01})
            if result.fun < best_rmse:
                best_rmse = result.fun
                best_c = result.x
        except:
            continue
    
    # Crear tokens finales con constantes optimizadas
    new_tokens = []
    c_idx = 0
    for t in tokens:
        if t == 'C':
            new_tokens.append(f"{best_c[c_idx]:.4f}")
            c_idx += 1
        else:
            new_tokens.append(t)
    
    return new_tokens, best_rmse


def evaluate_formula(model, device, x_data, y_data):
    """Usa Beam Search para generar una fórmula, optimiza constantes, y evalúa RMSE."""
    try:
        searcher = BeamSearch(model, device, beam_width=10, max_len=30)
        results = searcher.search(x_data, y_data)
        
        if results:
            best = results[0]
            original_tokens = best['tokens']
            
            # Optimizar constantes 'C' para ajustar a los datos
            optimized_tokens, rmse = optimize_constants(original_tokens, x_data, y_data)
            
            tree = ExpressionTree(optimized_tokens)
            formula_str = tree.to_infix() if tree.is_valid else str(optimized_tokens)
            
            return {
                'formula': formula_str,
                'tokens': original_tokens,
                'rmse': rmse,
                'success': rmse < 0.1
            }
        
        return {'formula': 'Error', 'tokens': [], 'rmse': float('inf'), 'success': False}
    
    except Exception as e:
        return {'formula': f'Exception: {e}', 'tokens': [], 'rmse': float('inf'), 'success': False}


def run_test():
    """Ejecuta el test completo."""
    print("=" * 60)
    print("🧪 TEST DE APRENDIZAJE DEL MODELO")
    print("=" * 60)
    
    model, device = load_model()
    print(f"🖥️ Dispositivo: {device}")
    print()
    
    # Generar datos de prueba
    x_test = np.linspace(0.1, 5, 20).astype(np.float32)
    
    results = []
    
    print("Probando fórmulas...")
    print("-" * 60)
    
    for name, func in TEST_FORMULAS:
        try:
            y_test = func(x_test).astype(np.float32)
            
            # Normalizar
            y_min, y_max = y_test.min(), y_test.max()
            if y_max - y_min > 1e-6:
                y_norm = (y_test - y_min) / (y_max - y_min)
            else:
                y_norm = y_test
            
            result = evaluate_formula(model, device, x_test, y_test)
            result['target'] = name
            results.append(result)
            
            status = "✅" if result['success'] else "❌"
            print(f"{status} {name:15} → {result['formula']:25} RMSE: {result['rmse']:.4f}")
            if not result['success']:
                print(f"   Tokens: {result['tokens']}")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"❌ {name:15} → Error: {e}")
            sys.stdout.flush()
            results.append({'target': name, 'success': False, 'rmse': float('inf')})
    
    print("-" * 60)
    
    # Resumen
    total = len(results)
    successes = sum(1 for r in results if r.get('success', False))
    avg_rmse = np.mean([r['rmse'] for r in results if r['rmse'] < float('inf')])
    
    print()
    print("=" * 60)
    print("📊 RESUMEN")
    print("=" * 60)
    print(f"   Fórmulas testeadas: {total}")
    print(f"   Resueltas (RMSE < 0.1): {successes}/{total} ({100*successes/total:.1f}%)")
    print(f"   RMSE promedio: {avg_rmse:.4f}")
    print()
    
    # Interpretación
    if successes == 0:
        print("⚠️ RESULTADO: El modelo NO resuelve ninguna fórmula.")
        print("   → Necesita más pre-entrenamiento o hay un problema.")
    elif successes < 4:
        print("📈 RESULTADO: El modelo está empezando a aprender.")
        print("   → Continúa con más Self-Play.")
    elif successes < 8:
        print("🔄 RESULTADO: El modelo está aprendiendo bien.")
        print("   → Buen progreso, sigue entrenando.")
    else:
        print("🎉 RESULTADO: El modelo está muy bien entrenado!")
        print("   → Listo para usar o benchmark.")
    
    print()
    return successes, total


if __name__ == "__main__":
    run_test()
