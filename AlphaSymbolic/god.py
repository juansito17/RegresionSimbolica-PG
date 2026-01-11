import math

# --- 1. DATOS REALES (Ground Truth) ---
# Soluciones conocidas del problema N-Reinas (A000170)
soluciones_reales = {
    1: 1, 2: 0, 3: 0, 4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352,
    10: 724, 11: 2680, 12: 14200, 13: 73712, 14: 365596, 15: 2279184,
    16: 14772512, 17: 95815104, 18: 666090624, 19: 4968057848,
    20: 39029188884, 21: 314666222712, 22: 2691008701644,
    23: 24233937684440, 24: 227514171973736, 25: 2207893435808352,
    26: 22317699616364044,
    27: 234907967154122528
}

def evaluar_formula_maestra(n):
    """
    Implementa la fórmula descubierta por la IA:
    Residuo = -x0 + (x0**(0.1363593*x0) + 302.125542968737)**log(x0)
    """
    x0 = float(n)
    
    # --- A. EVALUACIÓN ROBUSTA (Para evitar Overflow) ---
    try:
        # Intentamos cálculo directo primero
        term_potencia = x0**(0.1363593 * x0)
        base = term_potencia + 302.125542968737
        exponente = math.log(x0)
        
        # La fórmula de la IA es el resultado directo
        return -x0 + (base ** exponente)
        
    except (OverflowError, ValueError):
        # Fallback usando logaritmos si los números son demasiado grandes
        # log(base ** exponente) = exponente * log(base)
        # log(x0**(a*x0) + c) ~ a*x0*log(x0) para x0 grande
        log_base = math.log(x0**(0.1363593 * x0) + 302.125542968737)
        log_total = exponente * log_base
        return math.exp(log_total) # Si esto desborda, no hay nada que hacer en float64

# --- 2. EJECUCIÓN Y TABLA DE ERRORES ---
print(f"{'N':<4} | {'Predicción IA':<25} | {'Valor Real':<25} | {'Error %':<12}")
print("-" * 75)

errores = []

# Evaluamos desde N=4 porque N=2 y N=3 son 0 (error infinito)
# Y tu fórmula fue optimizada para la asintótica (N > 8)
for n in range(4, 28):
    real = soluciones_reales[n]
    pred = evaluar_formula_maestra(n)
    
    diff = abs(real - pred)
    error_porc = (diff / real) * 100
    errores.append(error_porc)
    
    # Formato visual para detectar el 0.00026%
    if n >= 20:
        # Mostrar con mucha precisión para los números grandes
        print(f"{n:<4} | {pred:<25.4e} | {real:<25.4e} | {error_porc:.6f}%")
    else:
        # Formato normal para los pequeños
        print(f"{n:<4} | {pred:<25.1f} | {real:<25.1f} | {error_porc:.4f}%")

print("-" * 75)
promedio_asintotico = sum(errores[-5:]) / 5
print(f"Error Promedio (N=23 a 27): {promedio_asintotico:.6f}%")