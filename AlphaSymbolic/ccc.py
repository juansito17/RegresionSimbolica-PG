import math

# --- 1. DATOS REALES (OEIS A000170) ---
soluciones_reales = {
    1: 1, 2: 0, 3: 0, 4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724,
    11: 2680, 12: 14200, 13: 73712, 14: 365596, 15: 2279184,
    16: 14772512, 17: 95815104, 18: 666090624, 19: 4968057848,
    20: 39029188884, 21: 314666222712, 22: 2691008701644,
    23: 24233937684440, 24: 227514171973736, 25: 2207893435808352,
    26: 22317699616364044, 27: 234907967154122528
}

def formula_simkin_2022(n):
    """
    Aproximación basada en el paper de Michael Simkin (Harvard, 2022).
    Q(n) ~ (0.143 * n)^n
    """
    if n == 0: return 0
    # Usamos logaritmos para evitar desbordamiento en números gigantes si fuera necesario
    # pero para N=27 Python aguanta potencias directas con floats.
    return (0.143 * n) ** n

print(f"{'N':<3} | {'Real (Exacto)':<24} | {'Simkin (2022)':<24} | {'Error %':<10}")
print("-" * 75)

for n in range(1, 28):
    real = soluciones_reales[n]
    estimado = formula_simkin_2022(n)
    
    # Cálculo de error
    if real == 0:
        texto_error = "N/A (Div0)"
    else:
        diff = abs(real - estimado)
        error_pct = (diff / real) * 100
        texto_error = f"{error_pct:.2f}%"
    
    # Visualización rápida de calidad
    estado = "🏆" if error_pct < 1.0 and real > 0 else "❌"
    
    print(f"{n:<3} | {real:<24} | {estimado:<24.2e} | {texto_error:<10} {estado}")

print("-" * 75)
print("CONCLUSIÓN: La fórmula de Simkin es un 'Límite Inferior'.")
print("Observa cómo SIEMPRE da menos que el valor real (subestima sistemáticamente).")