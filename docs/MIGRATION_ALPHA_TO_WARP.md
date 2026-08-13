# Migración de AlphaSymbolic a WarpSymbolic

## Imports

Usa los imports nuevos:

```python
from warpsymbolic import WarpSymbolicRegressor
from warpsymbolic.gpu import TensorGeneticEngine
```

Durante la transición todavía funcionan:

```python
from AlphaSymbolic.sklearn import AlphaSymbolicRegressor
```

Ese alias emite `DeprecationWarning` y se eliminará en la siguiente versión
mayor. Las rutas internas como `AlphaSymbolic.core.gpu.engine` ya no son API
pública.

## Comandos

| Antes | Ahora |
|---|---|
| `alphasymbolic-srbench` | `warp-symbolic-srbench` |
| `python -m AlphaSymbolic.scripts.benchmark_srbench` | `python -m warpsymbolic.cli.benchmark_srbench` |
| `python -m AlphaSymbolic.app` | `python -m AlphaSymbolic.app` (UI legacy; no reemplaza al núcleo GPU) |

El comando antiguo de SRBench sigue delegando al nuevo durante la transición.

## Política de dispositivo

`WarpSymbolicRegressor` usa CUDA por defecto. Si no existe una GPU CUDA,
`fit()` lanza `GpuUnavailableError`. Para pruebas explícitas de CPU:

```python
WarpSymbolicRegressor(device="cpu").fit(X, y)
```

Ese modo no representa el backend de producción ni sus métricas de rendimiento.

## Resultados históricos

Los JSONL y reportes que contienen `AlphaSymbolic` no se reescriben: sus
identificadores forman parte de la procedencia de la corrida. Los benchmarks
nuevos deben registrar `WarpSymbolic`.
