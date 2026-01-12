import math

# --- 1. LA FÓRMULA GENÉTICA (Traducida a Python) ---
def formula_genetica_discreta(n):
    # Variables del contexto (según tus logs)
    x0 = float(n)
    x1 = float(n % 6)  # El log mostraba x1=5.0 para N=23 (23 mod 6 = 5)
    
    # PROTECCIÓN: Evitar división por cero si n=0 (aunque no evaluamos n=0)
    if x0 == 0: return 0
    
    # --- LA ECUACIÓN RAW ---
    # Formula: lgamma(x0 + 1) - (abs(floor((x0 * 0.1038) ^ (x1 / 5)) ^ (x0 / (x0 * 0.585))) + x0)
    
    # Desglose paso a paso para evitar errores de sintaxis
    term_factorial = math.lgamma(x0 + 1)
    
    # Parte interna del Floor
    # Nota: En Python '^' es bitwise, usamos '**' para potencia
    base_floor = (x0 * 0.10380637) ** (x1 / 5.0)
    
    # El Floor
    piso = math.floor(base_floor)
    
    # El Exponente Constante (x0 se cancela con x0, queda 1/0.585...)
    exponente_constante = x0 / (x0 * 0.58525872)
    
    # El término de sustracción complejo
    sustraccion = abs(piso ** exponente_constante) + x0
    
    # Resultado final en escala logarítmica
    log_pred = term_factorial - sustraccion
    
    return math.exp(log_pred)

# --- 2. DATOS REALES (A000170) ---
soluciones_reales = {
    1: 1, 2: 0, 3: 0, 4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352,
    10: 724, 11: 2680, 12: 14200, 13: 73712, 14: 365596, 15: 2279184,
    16: 14772512, 17: 95815104, 18: 666090624, 19: 4968057848,
    20: 39029188884, 21: 314666222712, 22: 2691008701644,
    23: 24233937684440,       # VALOR CORRECTO (El que tu dataset tenía mal)
    24: 227514171973736, 25: 2207893435808352,
    26: 22317699616364044, 27: 234907967154122528
}

# --- 3. EVALUACIÓN ---
print(f"{'N':<3} | {'Real':<22} | {'Predicción IA':<22} | {'Error %':<10}")
print("-" * 65)

errores = []

for n in range(1, 28):
    real = soluciones_reales[n]
    pred = formula_genetica_discreta(n)
    
    # Cálculo de Error
    if real == 0:
        texto_error = "N/A (Div0)" # No se puede calcular error % de 0
    else:
        diff = abs(real - pred)
        error = (diff / real) * 100
        texto_error = f"{error:.4f}%"
        errores.append(error)
    
    # Marcador visual para errores bajos
    marca = "⭐" if (real > 0 and error < 1.0) else ""
    
    print(f"{n:<3} | {real:<22} | {pred:<22.4f} | {texto_error:<10} {marca}")

print("-" * 65)
if errores:
    # Promedio ignorando los N pequeños que suelen ser ruidosos
    avg_total = sum(errores) / len(errores)
    # Promedio de la "Zona Estable" (N > 15)
    errores_estables = errores[12:] # Desde N=16 en adelante (indices ajustados)
    if errores_estables:
        avg_estable = sum(errores_estables) / len(errores_estables)
        print(f"Error Promedio Global: {avg_total:.4f}%")
        print(f"Error Promedio (N>15): {avg_estable:.4f}%")