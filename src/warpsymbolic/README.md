# warpsymbolic

Paquete principal de WarpSymbolic: regresión simbólica evolutiva con GPU como
camino de producción.

## Contenido

- `api/`: estimador compatible con scikit-learn y funciones públicas.
- `symbolic/`: gramática, parsing y simplificación de fórmulas.
- `gpu/`: motor genético tensorial, evaluación, selección, optimización y
  extensión CUDA nativa.
- `cli/`: consola GPU, benchmarks y runners operativos.

## Uso público

```python
from warpsymbolic import WarpSymbolicRegressor
from warpsymbolic.gpu import TensorGeneticEngine
```

El ajuste de producción requiere CUDA y no realiza fallback silencioso a CPU.
El modo CPU se reserva para parsing, validación, tests y debugging explícito.

Consulta la documentación general en [`README.md`](../../README.md) y la
arquitectura en [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).
