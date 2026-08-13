# Estado verificable de AlphaSymbolic Adaptive — 2026-08-13

## Conclusión

El modo `adaptive` está implementado como candidato universal, pero todavía no ha superado los gates para ser el valor predeterminado ni para afirmar SOTA. No existe aún un puesto SRBench válido para esta versión: el ranking local anterior corresponde a otra configuración y no se reutiliza como evidencia del buscador nuevo.

## Smoke tests ejecutados

Configuración reducida: una semilla, un segundo máximo, 64 filas de entrenamiento y 96 de test. El guard de timeout omitió el arranque frío del motor evolutivo; por tanto estas cifras validan el pipeline disperso, la selección, la exportación y los artefactos, no la búsqueda profunda de 60 s.

| Medición | Resultado |
|---|---:|
| Casos de desarrollo | 6 |
| Recuperaciones numéricas | 2/6 |
| R² mediano | 0.999529 |
| RMSE mediano | 0.036821 |
| Complejidad mediana | 67.5 nodos |
| Tiempo mediano | 0.0232 s |
| N‑reinas log-RMSE, n=25…27 | 0.451372 |
| N‑reinas error relativo medio | 32.86 % |
| N‑reinas exactitud redondeada | 0 % |
| N‑reinas tiempo (1 semilla) | 0.1213 s |

Los fallos visibles son importantes: el caso periódico extrapola mal sin la fase evolutiva; el racional aún se aproxima con un polinomio; y N‑reinas sólo supera por margen pequeño al baseline polinómico logarítmico de grado 3. No se reinterpretan como éxitos.

La última auditoría profunda del motor anterior observó alrededor de 33 % de expresiones inválidas. El modo adaptativo activa precondiciones, reparación, NSGA‑II y epsilon‑lexicase, pero todavía falta una corrida profunda comparable que demuestre la meta inferior a 10 % y que descarte una regresión CUDA mayor al 5 %.

## Artefactos

- `adaptive_dev_smoke_20260813.jsonl`: filas completas de desarrollo.
- `adaptive_nqueens_clean_smoke_20260813.jsonl`: evaluación limpia y baselines entrenables.
- `adaptive_config_candidate_20260813.json`: configuración candidata de 60 s y población 50.000, hash `b85eb9a326cc256e6c8dd0b579640e4c045308558a97efce244c0d7ca3921c4e`.

## Reproducción

```powershell
python AlphaSymbolic/scripts/benchmark_scientific.py --suite development --mode adaptive --seeds 1 --max-time 1 --train-samples 64 --test-samples 96 --output benchmarks/adaptive_dev_smoke_20260813.jsonl
python AlphaSymbolic/scripts/benchmark_scientific.py --suite nqueens --mode adaptive --seeds 1 --max-time 1 --output benchmarks/adaptive_nqueens_clean_smoke_20260813.jsonl
python AlphaSymbolic/scripts/freeze_adaptive_config.py benchmarks/adaptive_config_candidate_20260813.json
```

La promoción exige después ejecutar desarrollo adaptativo contra legado, N‑reinas a 30 semillas, 24×30 con un único hash y SRBench upstream. `check_release_gates.py` falla de forma cerrada si falta evidencia o si cualquiera de las seis métricas no queda primera.
