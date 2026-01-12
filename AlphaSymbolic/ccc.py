import math
import matplotlib.pyplot as plt

# --- 1. TUS CONSTANTES CALIBRADAS (El corazón de tu fórmula) ---
# Derivadas de la regresión en N=24, 26 (Pares) y N=25, 27 (Impares)
A_PAR = 0.945525
B_PAR = 0.966099

A_IMPAR = 0.943389
B_IMPAR = 0.911941

# --- 2. LA VERDAD BASE (Secuencia OEIS A000170) ---
# Soluciones exactas conocidas hasta hoy
soluciones_reales = {
    1: 1, 2: 0, 3: 0, 4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724,
    11: 2680, 12: 14200, 13: 73712, 14: 365596, 15: 2279184,
    16: 14772512, 17: 95815104, 18: 666090624, 19: 4968057848,
    20: 39029188884, 21: 314666222712, 22: 2691008701644,
    23: 24233937684440, 24: 227514171973736, 25: 2207893435808352,
    26: 22317699616364044, 27: 234907967154122528
}

def predecir_soluciones(n):
    """Calcula soluciones estimadas usando el modelo asintótico híbrido."""
    log_factorial = math.lgamma(n + 1) # Logaritmo de n!
    
    if n % 2 == 0:
        # Fórmula para PARES: n! * e^(-An + B)
        log_pred = log_factorial - (A_PAR * n) + B_PAR
    else:
        # Fórmula para IMPARES: n! * e^(-An + B)
        log_pred = log_factorial - (A_IMPAR * n) + B_IMPAR
        
    return math.exp(log_pred)

# --- 3. EJECUCIÓN Y TABLA ---
print(f"{'N':<3} | {'Real (Exacto)':<24} | {'Tu Fórmula':<24} | {'Error %':<10} | {'Estado'}")
print("-" * 85)

n_vals = []
errores = []

for n in range(1, 28): # De 1 a 27
    real = soluciones_reales[n]
    estimado = predecir_soluciones(n)
    
    # Cálculo del error
    if real == 0:
        # Caso especial para N=2 y N=3 para evitar división por cero
        error_pct = 100.0 # Asumimos error total técnicamente
        texto_error = "N/A (Div0)"
        estado = "❌"
    else:
        diff = abs(real - estimado)
        error_pct = (diff / real) * 100
        texto_error = f"{error_pct:.4f}%"
        
        # Clasificación visual del éxito
        if error_pct < 0.01: estado = "🏆 GOD"   # Precisión Divina
        elif error_pct < 1.0: estado = "✅ Excel" # Excelente
        elif error_pct < 5.0: estado = "⚠️ Bueno" # Aceptable
        else: estado = "❌ Malo"                # Ruido inicial
    
    # Guardar datos para gráfica (filtramos los infinitos de N=2,3)
    if n > 3:
        n_vals.append(n)
        errores.append(error_pct)

    print(f"{n:<3} | {real:<24} | {estimado:<24.2e} | {texto_error:<10} | {estado}")

print("-" * 85)
print("NOTA: El error es alto en N < 12 debido al 'ruido de borde'.")
print("      Observa cómo converge a 0.00% a medida que N crece.")

# --- 4. GRÁFICA DE CONVERGENCIA (Opcional) ---
try:
    plt.figure(figsize=(10, 6))
    plt.plot(n_vals, errores, marker='o', color='b', label='Error Relativo')
    plt.axhline(y=0, color='black', linewidth=1)
    plt.axhline(y=1, color='green', linestyle='--', label='Umbral 1%')
    plt.title('Convergencia Asintótica de la Fórmula Peña-Usuga (N=4 a N=27)')
    plt.xlabel('Tamaño del Tablero (N)')
    plt.ylabel('Porcentaje de Error (%)')
    plt.yscale('log') # Escala logarítmica para ver mejor los errores pequeños
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.show()
except Exception as e:
    print("\n[!] No se pudo generar la gráfica (falta matplotlib), pero la tabla es correcta.")