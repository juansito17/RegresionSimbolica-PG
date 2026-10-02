# Perfilado del motor GPU y mejoras de convergencia — 2026-10-01

Hardware: NVIDIA GeForce RTX 3050 Laptop (GA107, 16 SM, 4 GiB, CC 8.6).
Software: Python 3.11, PyTorch 2.5.1+cu121, CUDA Toolkit 12.6, Nsight Systems
2024.5.1. Línea base: commit `b24272a`.

## 1. ¿Cuánto tiempo se iba en Python?

Medido con Nsight Systems (marcadores NVTX por generación y por etapa) sobre
`vlad1` (2 variables, 128 puntos):

| | Población 1 M | Población 50 k |
|---|---:|---:|
| Tiempo por generación | 70,5 ms | 19,5 ms |
| GPU ocupada | 84 % | 80 % |
| GPU ociosa esperando al host | 16 % | 20 % |
| … de ello, SymPy dentro del bucle | 12 % | 9 % |
| … resto del bucle Python y lanzamientos | 4 % | 11 % |

Reparto del tiempo de pared en corridas de 6 s según el problema:

| Componente | 50 k | 1 M |
|---|---:|---:|
| SymPy en el bucle (`_post_simplify_formula` en cada nuevo mejor) | 0–43 % | 0–20 % |
| PSO fusionado (500 ind. × 30 partículas × 40 pasos, ~35–40 ms fijos) | 24–52 % | 5–13 % |

Conclusiones:

- El bucle de Python en sí cuesta poco; migrarlo entero a C++ ganaría como
  máximo un ~10 %. El tiempo «de Python» relevante era SymPy.
- A 1 M el 64 % del tiempo de GPU es la evaluación, y el 37 % de la población
  son clones exactos (tokens y constantes) que se reevaluaban cada generación.
- Entre el 64 % y el 78 % de la población son duplicados estructurales.
- El PSO era un coste fijo por llamada que dominaba con poblaciones pequeñas.

## 2. Cambios

### Linear scaling (`USE_LINEAR_SCALING`)

El fitness de búsqueda es el RMSE del ajuste por mínimos cuadrados `a + b·f(x)`
(Keijzer, 2003). Por defecto se activa de forma adaptativa (ver §3). Se calcula dentro del kernel fusionado: medias, momentos y
co-momento con Welford, combinados entre hilos con la fórmula de Chan, y una
segunda pasada que suma los residuos (la identidad de una pasada
`SSE = M2y − Cfy²/M2f` cancela catastróficamente cerca de un ajuste exacto).
Las predicciones de la primera pasada se guardan en memoria compartida cuando
caben. Una predicción numéricamente constante (varianza < 1e-10·media²) recibe
`b = 0`.

Error frente a `numpy.linalg.lstsq` en float64: ≤ 1,2e-6 · std(y), también con
predicciones de media 2e4 y desviación 0,3.

La fórmula devuelta lleva `a` y `b` explícitos (`_materialize_scaling`):
`best_global_rpn/consts`, los callbacks y la cadena final describen `a + b·f`,
y los evaluadores fuera de la corrida (p. ej. `validate_strict`) no escalan.
El PSO fusionado y el evaluador clásico usan el mismo objetivo.

### Levenberg–Marquardt nativo (`CONSTANT_OPTIMIZER = 'lm'`)

`cuda/lm_kernels.cu`: un warp por individuo. El jacobiano respecto a las
constantes se calcula con números duales (`rpn_run_program_dual` en
`eval_core.cuh`), con las derivadas exactas de la misma semántica
estricta/protegida del evaluador (incluidas `pow` con base negativa,
`log`/`sqrt` protegidos y gamma vía digamma). Cada lane escribe la fila del
jacobiano de su muestra en memoria compartida y los lanes reducen JᵀJ y Jᵀr
repartiéndose las entradas. Un lane resuelve `(JᵀJ + λ·diag) δ = −Jᵀr` con
Cholesky en doble precisión. Con linear scaling `a` y `b` se optimizan junto
con las constantes y tras cada paso aceptado se reproyectan a su óptimo exacto.

Coste: optimizar 4000 individuos lleva ~2,4 ms (el PSO tardaba ~40 ms con
500). En el bucle se aplica a los `LM_K_NORMAL` mejores individuos
estructuralmente distintos que tienen al menos una constante libre (antes el
top-K estaba lleno de clones del mejor).

### Reutilización de fitness (`USE_FITNESS_REUSE`)

El kernel fusionado recibe el índice del padre de cada hijo (lo devuelve el
orquestador). Un hijo idéntico bit a bit a su padre (tokens y constantes)
copia su fitness y no se evalúa. Cualquier cambio posterior (migración,
deduplicación, inyecciones, perturbación de constantes, reparación) rompe la
igualdad y la fila se evalúa: la comprobación es por contenido, no por
contabilidad de índices modificados.

### SymPy acotado dentro del bucle

La limpieza simbólica de un nuevo mejor voluminoso sigue existiendo (actúa de
regularizador: quitarla por completo produjo fórmulas inválidas en test), pero
se intenta una sola vez por estructura y su tiempo total se limita a
`SYMPY_INLOOP_BUDGET_FRACTION` (5 %) del tiempo transcurrido más 0,25 s.

### Deduplicación (`DEDUP_REPLACEMENT`)

Los duplicados estructurales pueden sustituirse por fórmulas aleatorias
(`'random'`, comportamiento anterior) o por una mutación de subárbol del propio
duplicado (`'mutate'`).

### Fallos corregidos por el camino

- **Migración con fitness obsoleto**: los migrantes se elegían con el fitness
  de la generación anterior indexando la población nueva (la migración era en
  la práctica aleatoria). Ahora la migración se hace tras la evaluación y
  arrastra el fitness de los migrantes.
- **Constantes desalineadas al cargar fórmulas**: en `load_population_from_strings`
  un marcador `C` no reservaba hueco, de modo que en `C*sin(x0) + 2.5` el 2.5
  acababa en el slot de la primera `C`.

## 3. Resultados

Protocolo de `benchmark_convergence`: 15 problemas × 3 semillas, 15 s de pared
por corrida, 128 puntos de entrenamiento y 512 de test. «Resuelta» = RMSE de
entrenamiento < 1e-6 (umbral absoluto).

| Configuración | Población | Resueltas | Pared total | NRMSE test (media geom.) | gen/s (mediana) |
|---|---|---:|---:|---:|---:|
| Línea base (`b24272a`) | 1 M | 30/45 | 250,2 s | 6,88e-6 | 15,1 |
| LM, sin linear scaling | 1 M | 33/45 | 209,6 s | 2,41e-6 | 15,9 |
| LM + linear scaling desde el inicio | 1 M | 31/45 | 245,7 s | 2,67e-6 | 10,5 |
| **LM + linear scaling adaptativo (por defecto)** | 1 M | **33/45** | **208,2 s** | **1,68e-6** | 15,9 |
| Línea base (`b24272a`) | 50 k | 22/45 | 366,9 s | 2,78e-5 | 61,1 |
| LM, sin linear scaling | 50 k | 36/45 | 166,9 s | 2,30e-6 | 85,3 |
| **LM + linear scaling adaptativo (por defecto)** | 50 k | **35/45** | **172,5 s** | **1,32e-6** | 84,5 |

Por problema (mediana de 3 semillas; resueltas · NRMSE de test):

| Problema | Base 1 M | Nueva 1 M | Base 50 k | Nueva 50 k |
|---|---|---|---|---|
| Nguyen-5 | 3/3 · 1,7e-7 | 3/3 · 1,8e-7 | 0/3 · 5,0e-5 | 3/3 · 4,1e-6 |
| Nguyen-7 | 0/3 · inf | 0/3 · 2,5e-6 | 0/3 · 5,8e-5 | 1/3 · 1,3e-6 |
| Gaussiana de Feynman | 3/3 · 1,4e-7 | 3/3 · 1,3e-7 | 1/3 · 1,1e-4 | 3/3 · 1,5e-7 |
| Nguyen-12 | 3/3 · 7,4e-8 (2,5 s) | 3/3 · 8,0e-8 (1,7 s) | 2/3 · 7,2e-7 (8,9 s) | 3/3 · 7,0e-8 (0,7 s) |
| Keijzer-11 | 3/3 (2,5 s) | 3/3 (0,5 s) | 2/3 (6,0 s) | 3/3 (3,7 s) |
| Vladislavleva-1 | 0/3 · 1,4e-2 | 0/3 · 2,8e-4 | 0/3 · 3,3e-2 | 1/3 · 1,9e-4 |
| Pagie-1 | 0/3 · 3,3e-2 | 0/3 · 3,2e-3 | 0/3 · 4,7e-2 | 1/3 · 2,3e-3 |
| Gaussiana de Feynman 3 var. | 0/3 · 2,3e-3 | 3/3 · 3,0e-5 | 0/3 · 5,1e-2 | 2/3 · 2,1e-5 |
| Friedman-1 (ruido) | 0/3 · 1,1e-1 | 0/3 · 1,2e-1 | 0/3 · 1,9e-1 | 0/3 · 1,1e-1 |

El resto (Nguyen-1, 3, 6, 8, 10, Coulomb) se resuelve 3/3 en < 0,4 s en todas
las configuraciones.

Rendimiento del bucle (vlad1, 1 M, 8 s): configuración antigua 13,8 gen/s;
nueva 13,0 gen/s (12,0 sin reutilización de fitness: la reutilización aporta
~8 %; el linear scaling de dos pasadas y el LM cuestan ~5 %, a cambio de un
mejor RMSE ~50× menor en esa corrida). A 50 k la mediana sube de 61 a 85 gen/s
porque el LM sustituye al PSO, que era un coste fijo de ~40 ms por llamada.

### Linear scaling: por qué adaptativo

Activado desde el inicio, el linear scaling reduce mucho el error donde la
búsqueda se estanca lejos de la solución (Vladislavleva-1 y Pagie-1, 8–16×
frente a LM solo), pero rompe recuperaciones exactas sin constantes: en
Nguyen-3 (3/3 → 0/3) la población se llena de aproximaciones con constantes
ajustadas (p. ej. `0.83·2.719^x·x·(x²+1.2) − …`, error 5e-6) antes de dar con
`x⁵+x⁴+x³+x²+x`. El modo adaptativo empieza sin escalar y lo activa al 20 %
del presupuesto (o tras 40 generaciones de estancamiento global), salvo que el
mejor RMSE de entrenamiento ya sea < 1e-4 · std(y). Conserva las recuperaciones
exactas y obtiene la mejor media geométrica en ambas poblaciones.

### Deduplicación: ablación

9 problemas difíciles × 3 semillas, 50 k, 15 s (la referencia son las mismas
corridas de la configuración por defecto):

| Variante | Resueltas | NRMSE test (media geom.) |
|---|---:|---:|
| `'random'`, cada 100 generaciones (por defecto) | 17/27 | 7,5e-6 |
| `'mutate'`, cada 100 | 17/27 | 1,6e-5 |
| `'random'`, cada 25 | 15/27 | 2,1e-5 |
| `'mutate'`, cada 25 | 16/27 | 1,6e-5 |

Ninguna variante mejora la actual, que se mantiene; `'mutate'` queda como
opción. Deduplicar más a menudo resta generaciones (59,9 frente a 71,3 gen/s
de mediana) y sustituye demasiada población a la vez.

### Límites y observaciones

- Friedman-1 (ruido σ = 1): con LM el RMSE de entrenamiento baja a 0,85–0,98,
  por debajo del ruido; parte del ruido se ajusta. Con 3 semillas la diferencia
  con la línea base no es concluyente.
- El umbral de «resuelta» es absoluto: en la Gaussiana de Feynman de 3
  variables (y ≈ 0,05) una aproximación con RMSE < 1e-6 cuenta como resuelta
  aunque su NRMSE de test sea ~2e-5.
- La línea base a 1 M devolvió 3 fórmulas con errores de dominio en puntos de
  test (NRMSE `inf`: Nguyen-7 ×2, Friedman-1); la configuración por defecto
  nueva, ninguna en 90 corridas. Una fórmula sigue validándose solo en los
  puntos de entrenamiento, así que el riesgo existe.
- Con 3 semillas, diferencias de una corrida resuelta entre configuraciones no
  son estadísticamente significativas.

## 4. Pruebas

`tests/gpu/test_linear_scaling_lm_reuse.py` (26 pruebas): RMSE escalado frente
a mínimos cuadrados en los modos warp y bloque, detección de soluciones exactas
y coeficientes, reutilización de fitness (un ulp de diferencia fuerza la
evaluación), recuperación de constantes no lineales y derivadas por operador
del LM, LM nunca empeora y coincide con el evaluador, fórmula materializada
equivalente, estado restaurado tras `run()`, presupuesto de SymPy, máscara de
duplicados, deduplicación por mutación, migración con fitness, alineación de
constantes al cargar fórmulas y la regla del modo adaptativo. Suite completa:
265 pruebas pasan, 4 omitidas (las mismas que antes).

## Reproducción

```powershell
python -m warpsymbolic.cli.benchmark_convergence --seeds 3 --budget 15 `
  --output benchmarks/convergence.jsonl
```

Compilar la extensión (desde un *Developer PowerShell* de Visual Studio 2022):

```powershell
Push-Location src/warpsymbolic/gpu/cuda
python setup.py build_ext --inplace
Pop-Location
```
