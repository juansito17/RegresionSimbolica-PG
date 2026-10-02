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
constantes se calcula en modo inverso sobre una cinta de valores por muestra
(`rpn_tape_forward`/`rpn_tape_backward` en `eval_core.cuh`): una pasada hacia
delante y otra hacia atrás dan la fila entera, tenga la fórmula las constantes
que tenga; los hijos de cada nodo se precalculan una vez por individuo en
memoria compartida. Programas de más de 128 tokens usan una pasada de números
duales por constante (`rpn_run_program_dual`). Las derivadas son exactas para la
misma semántica estricta/protegida del evaluador (incluidas `pow` con base
negativa, `log`/`sqrt` protegidos y gamma vía digamma). Cada lane escribe la fila del
jacobiano de su muestra en memoria compartida y los lanes reducen JᵀJ y Jᵀr
repartiéndose las entradas. Un lane resuelve `(JᵀJ + λ·diag) δ = −Jᵀr` con
Cholesky en doble precisión. Con linear scaling `a` y `b` se optimizan junto
con las constantes y tras cada paso aceptado se reproyectan a su óptimo exacto.

En el bucle se aplica cada `LM_INTERVAL` generaciones a los `LM_K_NORMAL`
(4096) mejores individuos estructuralmente distintos que tienen al menos una
constante libre: la máscara de duplicados se calcula sobre toda la población
(el 70 % son clones; filtrar solo un top-K dejaba 369 estructuras distintas de
4096 candidatos) y el filtro de constantes solo sobre un grupo de 2·K.
El coste por individuo depende de la longitud de la fórmula y del número de
constantes: ~17–32 evaluaciones equivalentes en los problemas de 128 puntos,
frente a 1200 del PSO.

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

| Configuración | Población | Resueltas | Pared total | NRMSE test (media geom.) |
|---|---|---:|---:|---:|
| Línea base (`b24272a`) | 1 M | 30/45 | 250,2 s | 6,88e-6 |
| LM, sin linear scaling | 1 M | 33/45 | 209,6 s | 2,41e-6 |
| LM + linear scaling desde el inicio | 1 M | 31/45 | 245,7 s | 2,67e-6 |
| **Versión final (LM + linear scaling adaptativo)** | 1 M | **33/45** | **211,3 s** | **1,72e-6** |
| Línea base (`b24272a`) | 50 k | 22/45 | 366,9 s | 2,78e-5 |
| LM, sin linear scaling | 50 k | 36/45 | 166,9 s | 2,30e-6 |
| **Versión final (LM + linear scaling adaptativo)** | 50 k | **35/45** | **171,8 s** | **1,44e-6** |

Las filas intermedias son de compilaciones previas del LM (modo directo); la
versión final usa el jacobiano en modo inverso y se volvió a medir completa.

Por problema (mediana de 3 semillas; resueltas · NRMSE de test):

| Problema | Base 1 M | Final 1 M | Base 50 k | Final 50 k |
|---|---|---|---|---|
| Nguyen-3 | 3/3 (0,4 s) | 3/3 (0,4 s) | 2/3 (0,3 s) | 3/3 (0,1 s) |
| Nguyen-5 | 3/3 · 1,7e-7 | 3/3 · 1,8e-7 | 0/3 · 5,0e-5 | 3/3 · 4,9e-6 |
| Nguyen-7 | 0/3 · inf | 0/3 · 2,0e-6 | 0/3 · 5,8e-5 | 1/3 · 1,8e-6 |
| Gaussiana de Feynman | 3/3 · 1,4e-7 | 3/3 · 1,3e-7 | 1/3 · 1,1e-4 | 3/3 · 1,4e-7 |
| Nguyen-12 | 3/3 (2,5 s) | 3/3 (1,5 s) | 2/3 (8,9 s) | 3/3 (0,5 s) |
| Keijzer-11 | 3/3 (2,5 s) | 3/3 (0,5 s) | 2/3 (6,0 s) | 3/3 (1,8 s) |
| Vladislavleva-1 | 0/3 · 1,4e-2 | 0/3 · 3,4e-3 | 0/3 · 3,3e-2 | 1/3 · 1,9e-4 |
| Pagie-1 | 0/3 · 3,3e-2 | 0/3 · 3,5e-3 | 0/3 · 4,7e-2 | 1/3 · 9,6e-4 |
| Gaussiana de Feynman 3 var. | 0/3 · 2,3e-3 | 3/3 · 3,0e-5 | 0/3 · 5,1e-2 | 2/3 · 6,3e-5 |
| Friedman-1 (ruido) | 0/3 · 1,1e-1 | 0/3 · 2,2e-1 | 0/3 · 1,9e-1 | 0/3 · 1,2e-1 |

El resto (Nguyen-1, 3, 6, 8, 10, Coulomb) se resuelve 3/3 en < 0,4 s en todas
las configuraciones.

### Rendimiento (candidatos·generación por segundo)

El rendimiento depende mucho del tamaño del dataset. Protocolo de la auditoría
del 2026-07-26 (A000170: 17 puntos, 3 variables, objetivo en log, `fact`/`gamma`;
1 M de población, 120 generaciones, semillas 4200+, se descarta la primera):

| Código / configuración | Candidatos·gen/s | Mejor RMSE (mediana) |
|---|---:|---:|
| 2026-07-26 (auditoría) | 25,7 M/s | 0,032 |
| Actual con PSO (`CONSTANT_OPTIMIZER='pso'`, sin LS ni reutilización) | 33,9 M/s | 0,021 |
| Primera versión del LM (jacobiano en modo directo) | 21,6 M/s | 0,0033 |
| **Versión final (LM en modo inverso)** | **22,4 M/s** | **0,0029** |
| Final con `LM_K_NORMAL = 2048` | ~29 M/s | ~0,004 |

Con 17 puntos evaluar es tan barato que el LM (4096 individuos cada 2
generaciones, fórmulas de ~40 tokens y ~8 constantes) pesa: es la causa
del ~35 % menos de candidatos/s frente al PSO, a cambio de un RMSE 7× menor a
igual número de generaciones. El jacobiano en modo inverso redujo cada llamada
de 27,5 a 14 ms en este caso; el resto del coste es memoria local por hilo (la
cinta de valores no cabe en L1 con ~20 warps por SM). `LM_K_NORMAL` es el
mando para cambiar velocidad por calidad.

Con 128 puntos (vlad1, 1 M, 8 s) la configuración antigua daba 13,8 gen/s y
la nueva ~13 gen/s (la reutilización de fitness aporta ~8 %). A 50 k la
mediana del benchmark de convergencia sube de 61 a 68 gen/s porque el LM
sustituye al PSO, que era un coste fijo de ~40 ms por llamada.

Evaluaciones por generación (vlad1/pagie1, 128 puntos): a 1 M, ~1,14 M antes
(1,01 M de evaluación + ~130 k del PSO sobre 125 individuos) frente a ~0,80 M
ahora (~0,76 M reales tras reutilizar ~25 % + ~35 k equivalentes del LM sobre
2048 individuos); a 50 k, ~185 k antes (el PSO era el 70 %) frente a ~95 k.

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

- Friedman-1 (ruido σ = 1) es el único problema ruidoso y el resultado es
  mixto: el RMSE de entrenamiento baja a 0,83–1,0 (línea base 0,93–1,13), es
  decir, se ajusta parte del ruido. NRMSE de test por semilla: a 1 M
  0,22/0,23/0,10 frente a inf/0,11/0,085 (peor); a 50 k 0,14/0,11/0,12 frente
  a 0,19/0,25/0,08 (mejor). Para datos ruidosos conviene validar con un
  conjunto de reserva (el estimador sklearn ya lo hace).
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
