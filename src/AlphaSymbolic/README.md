# AlphaSymbolic

Namespace legacy para la aplicación Gradio y los experimentos que no forman
parte del camino GPU oficial de WarpSymbolic.

## Contenido

- `app.py` y `ui/`: aplicación Gradio y componentes de interfaz.
- `experimental/`: redes neuronales, Beam Search, MCTS y búsqueda híbrida.
- `data/`: utilidades y datos auxiliares.
- `benchmarking/`: comparaciones y herramientas históricas.
- `scripts/`: wrappers de compatibilidad para integraciones antiguas.

La evolución GPU principal vive en [`warpsymbolic`](../warpsymbolic/), no en
este paquete. Las nuevas integraciones deben importar desde:

```python
from warpsymbolic import WarpSymbolicRegressor
from warpsymbolic.gpu import TensorGeneticEngine
```

Para ejecutar la interfaz legacy desde la raíz del repositorio:

```powershell
python -m AlphaSymbolic.app
```

Consulta la guía de migración en
[`docs/MIGRATION_ALPHA_TO_WARP.md`](../../docs/MIGRATION_ALPHA_TO_WARP.md).
