# Optimización y corrección del motor GPU — 2026-10-01

Hardware: NVIDIA GeForce RTX 3050 Laptop (GA107, 16 SM, 4 GiB GDDR6, CC 8.6).
Software: Python 3.11, PyTorch 2.5.1+cu121, CUDA Toolkit 12.6.
Línea base: commit `9522bb3` con su extensión compilada original.

## Resultado

Mismo protocolo, mismas semillas y el mismo presupuesto de pared para la
línea base y la versión nueva (15 problemas × 3 semillas, 15 s por corrida,
población 1 M, 128 puntos de entrenamiento y 512 de test):

| Métrica | Línea base | Nueva | Cambio |
|---|---:|---:|---:|
| Corridas resueltas (RMSE de entrenamiento < 1e-6) | 20/45 | 30/45 | +50 % |
| Tiempo total de pared | 402,9 s | 250,3 s | −38 % |
| Media geométrica del NRMSE de test | 1,50e-4 | 5,54e-6 | 27× menor |

| Problema | Base: resueltas / NRMSE test | Nueva: resueltas / NRMSE test |
|---|---|---|
| Nguyen-3 (x⁵+…+x) | 1/3 · 1,2e-3 | 3/3 · 9,0e-8 (0,4 s) |
| Gaussiana de Feynman | 1/3 · 9,5e-4 | 3/3 · 1,4e-7 (0,3 s) |
| Nguyen-12 | 1/3 · 5,9e-2 | 3/3 · 7,4e-8 (2,6 s) |
| Keijzer-11 | 0/3 · 2,3e-1 | 3/3 · 1,0e-7 (3,2 s) |
| Vladislavleva-1 | 0/3 · 1,2e-1 | 0/3 · 2,0e-2 |
| Pagie-1 | 0/3 · 1,2e-1 | 0/3 · 3,1e-2 |
| Gaussiana de Feynman 3 variables | 0/3 · 3,2e-1 | 0/3 · 2,5e-3 |
| Friedman-1 (5 variables, ruido) | 0/3 · 2,6e-1 | 0/3 · 1,8e-1 |

Throughput del bucle evolutivo completo (1 variable, 128 puntos salvo que se
indique):

| Configuración | Base gen/s | Nueva gen/s | Memoria pico (base → nueva) |
|---|---:|---:|---|
| 100 k | 33,8 | 99,7 | 50 → 56 MiB |
| 1 M | 13,1 | 29,5 | 488 → 563 MiB |
| 2 M | 2,5 | 14,7 | 3,6 GiB (desborda) → 1,1 GiB |
| 4 M | 0,81 | 7,3 | 4,3 GiB (desborda) → 2,2 GiB |
| 6 M | — | 4,7 | → 3,3 GiB |
| 1 M, 5 variables | 9,1 | 29,3 | 3,2 GiB → 566 MiB |
| 1 M, 1024 puntos | 2,2 | 5,8 | 487 → 563 MiB |

Evaluador aislado (1 M individuos): 1×256 puntos en modo estricto pasa de
87 ms a 23,5 ms; con 5 variables de 641 ms (ruta clásica) a 19 ms; con
fórmulas de ~31 tokens de 218 ms a 55 ms (≈145 G nodos·punto/s).

## Bugs corregidos

### Correctitud matemática

- **`pow` con base negativa era siempre inválido.** `--use_fast_math`
  convertía `powf` en `exp2(y·log2(x))`, que da NaN para `x<0`; `x^2`, `x^3`
  o `x^12` con `x<0` contaban como error de dominio. Era la causa principal
  de los fallos en problemas polinómicos. Además, `sin`/`cos` perdían
  precisión con argumentos grandes (error relativo 1e-3 en `sin(12345)`).
  Ahora se compila sin las intrínsecas aproximadas (se conservan
  flush-to-zero, división/raíz aproximadas y FMA).
- **Constantes `e` y `pi` truncadas** (`2.718281828`): ahora se usa el valor
  exacto en todos los evaluadores y en la sanitización de SymPy.
- **Pila del evaluador clásico**: un programa más profundo que la pila
  descartaba valores en silencio; ahora es inválido. La pila pasa de 32 a 64.
- **PSO con una tercera semántica**: el PSO fusionado tenía sus propias
  reglas (`%` sin corrección de signo, umbrales distintos para `/`, `log`,
  `exp`, `pow`, `fact`). Ahora comparte operadores y modo estricto con el
  evaluador.

### Lógica evolutiva

- **SBX mezclaba constantes no alineadas**: el 97,5 % de las copias sin cruce
  salía con constantes alteradas. Ahora las copias conservan sus constantes;
  SBX solo mezcla padres con idéntica estructura, y los hijos del cruce
  heredan las constantes de sus propios segmentos.
- **La mutación estructural truncaba fórmulas** (5,9 % de hijos inválidos):
  ahora los injertos que no caben se omiten, y las constantes del sufijo se
  re-mapean (las del subárbol injertado reciben valores nuevos).
- **Hoist desalineaba constantes**: al elevar un subárbol, sus `C` seguían
  leyendo los slots antiguos; ahora se desplazan.
- **La mutación puntual podía crear o borrar `C`**, desplazando todas las
  constantes posteriores. Ahora `C` nunca muta por puntos.
- **Fitness obsoleto tras inyecciones**: los individuos aleatorios inyectados
  en el estancamiento, los del frente de Pareto y el reseed de ALPS heredaban
  el fitness del individuo reemplazado; ahora se evalúan.
- **Inyecciones perdidas**: Pareto y residual boosting escribían en
  `pop_buffer_A/B`, que no es la población viva en la ruta C++.
- **Lexicase leía fuera del tensor**: el caso se sorteaba sobre el dataset
  completo pero indexaba la submuestra de 128 columnas. Además, el mejor RMSE
  se medía en la submuestra (optimista); ahora se mide en todos los datos.
- **ALPS**: `pop_constants[idx].uniform_()` no hacía nada y las edades nunca
  se heredaban; ahora la edad sigue el linaje que devuelve el orquestador.
- **Generador aleatorio**: el 75 % de las fórmulas con una variable no tenía
  ninguna variable y el 37 % tenía 2 tokens. Ahora cada fórmula se genera con
  una longitud objetivo en `[INIT_MIN_LENGTH, INIT_MAX_LENGTH]` y los
  terminales siguen pesos configurables (variables 50 %, `C` 30 %, literales
  20 %).
- **Deduplicación**: tabla fija de 2²⁰ huecos (se saturaba con millones de
  individuos) y sin verificación de igualdad; ahora la tabla escala con la
  población y las coincidencias de hash se confirman comparando filas.
- **Perturbación de constantes**: recortaba a ±25 valores ya fuera de rango.
- **Resultado final**: la simplificación final podía aceptar una fórmula un
  20 % peor (evaluada en modo protegido) y reportar el RMSE anterior; la
  validación con NumPy solo conocía `x0`. Ahora se valida con la semántica
  de búsqueda, no se acepta nada peor y funciona con varias variables.
- Otros: `RPN_CUDA_AVAILABLE` no definido en el motor (fitness sharing CUDA
  inalcanzable), `selection_metric` usado antes de asignarse,
  `PSO_PARTICLES` ignorado, `get_arity_ids` devolvía PAD como relleno,
  la ruta rápida se desactivaba para siempre sin aviso.

## Optimizaciones

- **Evaluador decodificado** (`cuda/eval_core.cuh`): cada programa se
  decodifica una vez por individuo con un warp (opcode, variable, constante
  resuelta, validación de pila por prefix-scan); la evaluación por punto es un
  `switch` denso con el tope de pila en registro. Sin límite de 4 variables ni
  de 1024 puntos; los programas inválidos no se ejecutan; en modo estricto un
  error termina el individuo para todo el warp.
- **Sin tope de 1,5 M** para la ruta rápida: por encima, la ruta clásica
  materializaba matrices `[1M, D]` y desbordaba la VRAM.
- **PSO fusionado con un warp por partícula**: antes cada hilo recorría todos
  los puntos en serie (20 hilos por bloque).
- **Rastreador del mejor** con reducción sobre toda la GPU (antes 1 bloque).
- **Longitudes y presencia de variables** en un solo kernel por fila (antes
  cuatro reducciones de PyTorch, ~14 % del tiempo por generación).
- **Menos sincronizaciones CPU↔GPU**: mutación estructural y hoist con
  máscaras en vez de `nonzero`, `masked_fill_` en lugar de indexado booleano.

## Ajuste de parámetros

Ablación pareada sobre los 6 problemas no resueltos (6 semillas, 15 s):
`BASE_MUTATION_RATE` 0,22 → 0,10 e `INIT_MAX_LENGTH` 24 → 12 reducen la media
geométrica del NRMSE de test de ~2,1e-3 a ~8,2e-4. Más PSO
(`PSO_K_NORMAL=2000`, `PSO_INTERVAL=2`) empeoró. `BEST_SYNC_INTERVAL=1`
detecta soluciones exactas hasta 9 generaciones antes sin coste medible.
Con presupuesto de 15 s, una población de 2 M no mejoró de forma consistente
a 1 M (8/36 frente a 7/36 resueltas, NRMSE similar), así que 1 M sigue siendo
el valor por defecto.

## Pruebas

`tests/gpu/test_evolution_semantics_regressions.py` cubre `pow` con base
negativa, precisión de `sin`, `e` exacto, conservación de constantes en las
copias, mutación estructural sin truncamiento, índices de lexicase,
generador aleatorio, perturbación y deduplicación. Siete de sus nueve pruebas
fallan con la línea base. Suite completa: 239 pruebas pasan, 4 omitidas.

## Reproducción

```powershell
python -m warpsymbolic.cli.benchmark_convergence --seeds 3 --budget 15 `
  --output benchmarks/convergence.jsonl
```

Compilación de la extensión en Windows (desde un *Developer PowerShell* de
Visual Studio 2022, o tras `vcvars64.bat`):

```powershell
Push-Location src/warpsymbolic/gpu/cuda
python setup.py build_ext --inplace
Pop-Location
```

## Límites conocidos

- Fórmulas de hasta 256 tokens en la ruta rápida y profundidad de pila 64.
- Constantes acotadas a `CONSTANT_MIN_VALUE..CONSTANT_MAX_VALUE` (±25) por el
  PSO.
- La deduplicación es estructural: dos individuos con la misma estructura y
  constantes distintas cuentan como duplicados.
- Lexicase sigue usando el evaluador clásico para la matriz de errores
  `[B, casos]`; es adecuado para las poblaciones del modo adaptativo
  (~50 k), no para millones.
- Con 3–6 semillas por problema las diferencias pequeñas entre
  configuraciones no son estadísticamente concluyentes.
