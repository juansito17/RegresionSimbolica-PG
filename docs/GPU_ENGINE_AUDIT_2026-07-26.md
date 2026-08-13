# Auditoría del motor GPU, kernels CUDA y uso web

Fecha: 2026-07-26  
Commit de partida: `c62d68d`  
Hardware: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB, compute capability 8.6  
Software: Python 3.11.0, PyTorch 2.5.1+cu121, driver 610.74, CUDA Toolkit 12.6

## Veredicto

El motor es rápido y ya resulta útil como plataforma experimental de regresión
simbólica GPU. Después de esta auditoría es más seguro, medible y reutilizable:
se corrigieron rutas que podían producir resultados matemáticamente incorrectos,
se aisló el estado entre ejecuciones, se eliminó una sincronización web costosa,
se hizo efectiva la distribución configurada de operadores y se añadió un
benchmark científico con holdout.

No hay evidencia suficiente para llamarlo SOTA. En el benchmark interno
controlado mejora la mediana de rendimiento un 2,62% y la mediana final de RMSE
un 22,81%, pero una de cinco semillas todavía converge mal. En el holdout
científico recupera exactamente dos familias fáciles, funciona muy bien en una
gaussiana de Feynman y muestra debilidad clara en Friedman-1 y Nguyen-5. La
comparación completa con sistemas externos y cientos de problemas aún es un
requisito, no un resultado asumido.

## Alcance auditado

- Motor tensorial e islas evolutivas en `src/warpsymbolic/gpu`.
- Evaluador RPN, generación, evolución, PSO y simplificación CUDA nativa.
- Evaluador y fitness CUDA del motor C++ en `legacy/cpp_engine/src`.
- Search híbrido, Beam/MCTS, caché de motores y memoria de patrones.
- Pantallas Gradio de evolución en vivo y benchmark.
- Semántica de fórmulas entre parser, Python, GPU y presentación.
- Medición de rendimiento, convergencia, validez, diversidad y VRAM.
- Holdout científico en Nguyen, Feynman, Friedman y OEIS A000170.

## Protocolo de rendimiento

La prueba antes/después usa:

- población: 1.000.000;
- 20 islas;
- 120 generaciones medidas y 2 de calentamiento;
- semillas 4200–4204;
- cinco repeticiones;
- dos segundos de enfriamiento entre repeticiones;
- misma RTX 3050 Laptop y mismo perfil explícito de operadores A000170.

`candidatos-generación/s` se calcula como
`población × generaciones completadas / tiempo de pared`. Incluye el trabajo
del ciclo evolutivo y no multiplica por el número de filas del dataset. Por eso
no debe compararse directamente con GPops/s ni con “evaluaciones por punto” de
otros trabajos.

La primera repetición se excluye solamente de las métricas “hot” de throughput
y RMSE. Las métricas de estructura y tiempo a umbral usan las cinco semillas.

### Resultado controlado

| Métrica | Antes | Después | Cambio |
|---|---:|---:|---:|
| Throughput hot, mediana | 25,755 M/s | 26,430 M/s | +2,62% |
| Throughput hot, media | 25,787 M/s | 28,063 M/s | +8,83% |
| Throughput hot, media recortada | 25,755 M/s | 26,430 M/s | +2,62% |
| RMSE final, mediana de 5 | 0,034856 | 0,026904 | −22,81% |
| RMSE final hot, mediana | 0,032263 | 0,032074 | −0,59% |
| Mejor RMSE | 0,025021 | 0,024664 | −1,43% |
| Tiempo mediano a RMSE < 0,10 | 1,505 s | 1,457 s | −3,20% |
| Tiempo mediano a RMSE < 0,05 | 3,586 s | 2,738 s | −23,65% |
| Aciertos de RMSE < 0,05 | 5/5 | 4/5 | peor varianza |
| Individuos inválidos, mediana | 33,83% | 32,89% | −2,76% |
| Individuos inválidos, media | 34,16% | 27,70% | −18,93% |
| Diversidad estructural, mediana | 0,7576 | 0,7610 | +0,45% |
| Longitud media muestreada, mediana | 35,18 | 30,73 | −12,64% |
| Pico de memoria asignada, mediana | 628,88 MiB | 628,73 MiB | esencialmente igual |

La media de throughput posterior está influida por una corrida de 33,62 M/s.
La mediana/recortada de +2,62% es la estimación prudente. Una corrida posterior
terminó con RMSE 0,1122; por eso tampoco se debe describir el cambio como una
mejora uniforme de convergencia.

Datos crudos:

- [línea base](../benchmarks/audit_baseline_20260726.jsonl);
- [resultado final](../benchmarks/audit_post_final_20260726.jsonl).

## Hallazgo web aislado

La app establecía `CUDA_LAUNCH_BLOCKING=1`, lo que convertía operaciones CUDA
asíncronas en una secuencia de esperas globales. Un A/B sobre el código base,
cambiando únicamente esa condición, usó tres repeticiones de 40 generaciones:

| Ruta | Throughput hot |
|---|---:|
| Web con `CUDA_LAUNCH_BLOCKING=1` | 30,422 M/s |
| Ejecución asíncrona | 34,271 M/s |
| Mejora | +12,65% |

Las curvas y RMSE fueron idénticas con las semillas pareadas. La app ya no
fuerza ese modo de depuración.

Datos crudos:

- [web bloqueante](../benchmarks/audit_web_blocking_baseline_20260726.jsonl);
- [control asíncrono](../benchmarks/audit_web_async_baseline_20260726.jsonl).

## Validación científica fuera de muestra

El nuevo `benchmark_scientific.py` separa train/test, registra semilla,
hardware, configuración, tiempo, fórmula y métricas estrictas. No usa plantillas,
Sniper ni una fórmula inicial. La prueba aquí registrada usó 100.000 individuos,
20 islas, hasta 150 generaciones, 30 s por corrida y semillas 6000–6002.

| Problema | Resultado de test |
|---|---|
| Nguyen-1 | 3/3 recuperaciones exactas; NRMSE mediana ≈ 1,0×10⁻¹⁶ |
| Nguyen-5 | 1/3 exacta; NRMSE mediana 0,1103 |
| Feynman I.6.2, gaussiana | NRMSE mediana 1,66×10⁻⁶; dos corridas hallan `377*exp(-x0**2/2)/945` |
| Producto de Feynman | 3/3 exactas con `x0*x1` |
| Friedman-1, cinco variables | NRMSE mediana 0,4469; mejor 0,3788 |
| A000170, train n=8…24/test n=25…27 | 2/3 fórmulas válidas; NRMSE mediana válida 0,1277; mejor 0,0817 |

Los resultados completos están en
[scientific_post_20260726.jsonl](../benchmarks/scientific_post_20260726.jsonl).

Friedman-1 es especialmente importante: usa cinco variables, por lo que activa
el evaluador clásico y no el fused limitado a cuatro. El resultado muestra que
la velocidad en A000170 no implica capacidad multivariable general.

### Comparación externa pareada con PySR

También se ejecutó PySR con los mismos seis problemas, datasets, semillas,
holdouts y timeout solicitado de 30 s. Los tiempos reales se conservaron: PySR
tuvo una primera ejecución de 104,37 s por inicialización/compilación y ambos
sistemas pueden sobrepasar el timeout dentro de una fase no interrumpible. Por
eso esta comparación es informativa, no una certificación SOTA de presupuesto
estrictamente igual.

| Problema | AlphaSymbolic: NRMSE mediana (válidas) | PySR: NRMSE mediana (válidas) | Lectura |
|---|---:|---:|---|
| Nguyen-1 | 1,02×10⁻¹⁶ (3/3; 3 exactas) | 6,41×10⁻¹⁷ (3/3; 3 exactas) | empate práctico |
| Nguyen-5 | 0,1103 (3/3; 1 exacta) | 0,1349 (3/3; 0 exactas) | AlphaSymbolic |
| Feynman gaussiana | 1,66×10⁻⁶ (3/3) | 4,56×10⁻⁷ (2/3) | PySR entre válidas; una corrida no finita |
| Producto Feynman | 0 (3/3 exactas) | 1,79×10⁻⁸ (3/3; 1 exacta) | AlphaSymbolic |
| Friedman-1 | 0,4469 (3/3) | 0,4051 (3/3) | PySR |
| A000170 | 0,1277 (2/3) | 0,4266 (3/3) | AlphaSymbolic en mediana válida; PySR obtuvo el mejor run (0,00957) |

En las 18 parejas, AlphaSymbolic obtuvo menor NRMSE en 7, PySR en 9 y 2 no
fueron comparables por predicciones inválidas. AlphaSymbolic consumió 182,99 s
agregados frente a 690,10 s de PySR (3,77× menos); la mediana por run fue
1,47 s frente a 34,04 s (23,1×). Estos tiempos incluyen early-stop y overhead
real, no igualan trabajo interno ni tuning.

Los 18 resultados de PySR están en
[scientific_pysr_20260726.jsonl](../benchmarks/scientific_pysr_20260726.jsonl).
El harness ahora persiste cada run al terminar, para no perder una campaña larga
si el proceso se interrumpe.

## N-Reinas: qué resuelve y qué no

El caso A000170 ajusta una secuencia con `n`, `n mod 6` y `n mod 2`; no coloca
reinas, no enumera soluciones y no demuestra una identidad exacta. Los valores
de test 25–27 se tomaron de la [tabla oficial OEIS
A000170](https://oeis.org/A000170/list).

La teoría actual ofrece priors que una futura pista combinatoria podría usar:

- Simkin caracteriza el crecimiento asintótico como
  `Q(n)=((1±o(1)) n e^-α)^n`, con α≈1,942
  ([The number of n-queens configurations](https://arxiv.org/abs/2107.13460)).
- Nielsen demuestra que `Q(n)` es divisible por cuatro para `n≥6`
  ([The n-queens solution count Q(n) is divisible by 4](https://arxiv.org/abs/2601.05856)).

Esos hechos deben entrar como restricciones verificables o features de una
pista separada. Inyectarlos sin declarar el prior invalidaría una comparación
de descubrimiento “desde cero”.

## Correcciones realizadas

### Corrección matemática y kernels

- El evaluador fused ahora solo se usa con `variables≤4`,
  `longitud≤256` y `muestras≤1024`; fuera de esos límites cae al evaluador
  clásico. El launcher C++ valida además tamaños no vacíos, shapes, dtype,
  dispositivo y contigüidad, incluso si se llama directamente a la extensión.
- El PSO fused rechaza FP64 y usa la ruta multi-kernel compatible; el cálculo
  de memoria compartida usa el tamaño real del escalar.
- El fitness C++ aplica realmente los pesos solicitados y se añadieron pruebas
  de operadores/fitness.
- `gamma` y `lgamma` tienen flags independientes.
- `^` se normaliza a `**` sin corromper expresiones que ya contienen `**`.
- El ensemble consume el RMSE registrado por cada engine, no un valor
  inexistente o reconstruido.

### Simplificación segura

- Se eliminaron reglas no válidas bajo semántica estricta, entre ellas
  cancelaciones de `exp/log`, inversas trigonométricas, `0/x`, autocancelación,
  módulo y consolidaciones que ignoraban dominios o NaN.
- Una fórmula con slots de constantes `C` no pasa por la simplificación
  simbólica que perdería la correspondencia posicional.
- Simplificación CUDA, library learning y L-BFGS nativo quedan desactivados por
  defecto hasta contar con equivalencia semántica y pruebas de colisiones.

### Convergencia y estado de ejecución

- Los pesos de operadores ahora son categóricos por operador real; antes el
  vector configurado no gobernaba correctamente el muestreo CUDA.
- Los cambios de operadores hechos desde la UI después del import conservan su
  distribución correspondiente.
- El perfil genérico activa aritmética, trigonometría común, `log`, `exp`,
  `sqrt` y `abs`; no activa factorial/gamma ni transforma `Y` implícitamente.
- Los runners A000170 optan explícitamente por `fact/gamma/lgamma`, target
  logarítmico y un perfil uniforme dentro de cada aridad, elegido con una
  ablación pareada.
- Cada `run()` reinicia best, métricas, cancelación, memoria y cachés
  dependientes de la corrida. La fórmula final se construye desde una copia de
  constantes, evitando mutar el campeón almacenado.
- Se registran curvas, tiempo a umbral, inválidos, diversidad, longitud, VRAM,
  generaciones efectivas y modo de evaluación.

### Web, concurrencia y search

- Stop solicita cancelación y espera la finalización del worker antes de
  declarar la tarea detenida.
- La UI restaura globals y libera buffers en `finally`, también ante error.
- Un lock de proceso evita que benchmark, evolución live e híbrido muten la
  configuración global o usen la misma GPU simultáneamente.
- La caché híbrida es LRU acotada a un solo engine y su clave incluye shape,
  precisión y portfolio de operadores; existe limpieza explícita.
- La pestaña benchmark ya no presenta Beam/MCTS ficticios. Ejecuta el GP real y
  una regresión polinómica de grado cinco claramente etiquetada, con streams
  train/test independientes y fallos excluidos correctamente de la media.
- Los entrypoints MCTS consumen el contrato actual (`dict["tokens"]`) y manejan
  la ausencia de candidato. `PatternMemory` es opcional y ya no importa una
  implementación inexistente/incompatible.
- Un smoke test en navegador cargó la app real, abrió GPU Evolution y Benchmark,
  ejecutó el baseline polinomial (10/10 runs válidos) y no encontró errores ni
  warnings de consola. Ese test detectó y permitió corregir un estado atascado:
  un input vacío ahora restaura `INICIAR` y deshabilita `DETENER`.
- La UI genérica ya no activa factorial/gamma/lgamma por defecto; el usuario
  puede habilitarlos para N-Reinas u otro dominio combinatorio.

## Qué hacen los sistemas actuales y qué falta aquí

| Referencia | Idea relevante | Implicación para este motor |
|---|---|---|
| [EvoGP](https://arxiv.org/abs/2501.17168) | Árboles tensorizados, kernels CUDA propios y paralelismo adaptativo intra/inter-individuo; publica >10¹¹ GPops/s y comparativas amplias | Adoptar una métrica interoperable y paralelismo adaptativo por shape; nuestro conteo interno no prueba superioridad |
| [Parallel Symbolic Enumeration](https://www.nature.com/articles/s43588-025-00904-8) | Reutiliza subárboles comunes, evalúa cientos de millones de candidatos y reporta >200 problemas | La mayor oportunidad arquitectónica es representar/evaluar DAGs o bancos de subexpresiones compartidas |
| [Beagle GPU](https://arxiv.org/abs/2603.12292) | Compara GPU GP con StackGP y PySR bajo el mismo presupuesto de pared en Feynman | Repetir exactamente ese criterio de presupuesto y recuperación simbólica |
| [SRBench 2025](https://arxiv.org/abs/2505.03977) | Ningún algoritmo domina todos los datasets; exige recursos/tuning estandarizados y mide exactitud, complejidad y energía | “SOTA” debe declararse por pista, no globalmente |
| [SRBench++](https://pubmed.ncbi.nlm.nih.gov/40761553/) | Extrapolación, ruido, selección de variables, mínimos locales e interpretación experta | El holdout añadido es solo el comienzo de estas pistas |
| [Frentes Pareto absolutos, ICML 2025](https://proceedings.mlr.press/v267/fong25b.html) | Límites exactos error–longitud en 34 datasets | Mantener un archivo Pareto real y comparar distancia al frente, no solo un campeón RMSE |
| [Fast Symbolic Regression Benchmarking](https://arxiv.org/abs/2508.14481) | Expresiones equivalentes curadas y early-stop reducen coste y cambian recovery | Añadir equivalencia aceptable y callbacks de recuperación exacta al harness |

## Riesgos pendientes

1. **Varianza de convergencia.** Una de cinco semillas controladas empeoró mucho;
   el portfolio necesita adaptación por problema y más réplicas.
2. **Más de cuatro variables.** El fallback es correcto, pero pierde la ruta
   fused y Friedman-1 muestra calidad insuficiente.
3. **Semántica dispersa.** Aún existen implementaciones de operadores en Python,
   varios kernels y C++; faltan property tests masivos de equivalencia y dominio.
4. **Configuración global.** El lock evita carreras, pero el diseño correcto es
   una configuración inmutable por ejecución.
5. **Features experimentales.** Pareto, lexicase, ALPS, library learning,
   simplificador y L-BFGS no deben activarse por llamarse “avanzadas”; necesitan
   ablation y pruebas de corrección.
6. **Límites de stack.** Los kernels usan stacks fijos y el motor C++ conserva
   buffers de 64 posiciones; se deben validar/rechazar programas incompatibles
   en todos los launchers.
7. **Toolchain.** La extensión se compiló con CUDA 12.6 mientras PyTorch enlaza
   CUDA 12.1. Funciona en esta máquina, pero la matriz soportada debe fijarse.
8. **Benchmark externo incompleto.** Se necesita una matriz publicada contra
   PySR, Operon, EvoGP, Beagle y PSE, con versiones y presupuestos pareados.

## Ruta razonable hacia competitividad SOTA

1. Convertir el harness a SRBench/SRBench++ y ejecutar 20–100 semillas por
   problema, con recuperación simbólica, Pareto, holdout, ruido, energía y
   presupuesto de pared.
2. Hacer la configuración inmutable por run y separar RNG por engine/isla para
   reproducibilidad determinista.
3. Extender o reemplazar el fused para `V>4` y seleccionar kernels por
   `(B,L,D,V,dtype)` con autotuning persistente.
4. Introducir evaluación compartida de subárboles/DAG y deduplicación exacta,
   inspirada en PSE, con límites de VRAM.
5. Implementar un archivo Pareto correcto y selección epsilon-lexicase; aceptar
   cada feature solo si gana una ablación preregistrada.
6. Aprender el portfolio de operadores mediante bandits por isla, usando mejora
   marginal por milisegundo y conservando una isla de exploración sin prior.
7. Añadir restricciones científicas declarativas: unidades/dimensiones,
   simetrías, monotonicidad, conservación y dominios.
8. Separar N-Reinas en tres pistas: regresión ciega, regresión con priors
   matemáticos declarados y solver/contador real. No mezclar sus resultados.

## Reproducción

Benchmark de rendimiento:

```powershell
python -m warpsymbolic.cli.benchmark_gpu_console `
  --pop-size 1000000 --islands 20 --generations 120 `
  --warmup-generations 2 --repeats 5 --discard-first 1 `
  --cooldown-sec 2 --seed 4200 --timeout-sec 120 `
  --output benchmarks/audit_post_final_20260726.jsonl
```

Holdout científico:

```powershell
python -m warpsymbolic.cli.benchmark_scientific `
  --suite all --methods alphasymbolic --seeds 6000,6001,6002 `
  --train-points 128 --test-points 512 --pop-size 100000 `
  --islands 20 --generations 150 --timeout-sec 30 `
  --output benchmarks/scientific_post_20260726.jsonl
```

Comparador PySR pareado:

```powershell
python -m warpsymbolic.cli.benchmark_scientific `
  --suite all --methods pysr --seeds 6000,6001,6002 `
  --train-points 128 --test-points 512 --pop-size 100000 `
  --islands 20 --generations 150 --timeout-sec 30 `
  --pysr-iterations 100 `
  --output benchmarks/scientific_pysr_20260726.jsonl
```

Pruebas ejecutadas:

```text
python -m pytest tests -q
47 passed

python -m pytest tests/gpu -q
94 passed, 4 skipped

legacy/cpp_engine/scripts/run_tests.bat
7 native C++/CUDA test suites passed
```

Los cuatro skips GPU son pruebas legacy no aplicables a la gramática/configuración
actual, no fallos de ejecución. La extensión CUDA fue recompilada y el target
CMake Release también compiló correctamente.
