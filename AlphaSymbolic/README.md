# AlphaSymbolic

> Capa Python/Gradio para regresion simbolica neuro-evolutiva con PyTorch y GPU.

AlphaSymbolic combina varios enfoques:

- Modelo neuronal Transformer para generar candidatos.
- Beam Search, MCTS y busqueda hibrida.
- Motor genetico tensorial en GPU (`TensorGeneticEngine`).
- Extension CUDA nativa para evaluacion, mutacion, simplificacion y optimizacion de constantes.
- Interfaz web en Gradio para busqueda, entrenamiento, benchmark y monitoreo.

---

## Instalacion

Desde esta carpeta:

```powershell
pip install -r requirements.txt
```

Dependencias principales:

- `torch`
- `numpy`
- `scipy`
- `sympy`
- `gradio`
- `gymnasium`
- `pandas`

Para GPU necesitas un driver NVIDIA funcional. Si PyTorch no detecta CUDA, instala la rueda CUDA adecuada desde la documentacion oficial de PyTorch.

---

## Ejecutar la App

```powershell
cd AlphaSymbolic
python app.py
```

Por defecto la app corre en modo local. Para URL publica de Gradio o logs detallados:

```powershell
python app.py --share
python app.py --verbose
```

Tambien puedes activar logs detallados con:

```powershell
$env:ALPHASYMBOLIC_VERBOSE="1"
python app.py
```

La app crea una interfaz Gradio con:

- `Buscar Formula`: entrada de datos X/Y, carga CSV y busqueda por Beam Search, MCTS o Alpha-GP Hybrid.
- `Entrenar Modelo`: entrenamiento basico, curriculum, self-play, feedback loop y memoria.
- `GPU Evolution`: ejecucion/monitoreo del motor evolutivo GPU.
- `Benchmark`: problemas clasicos de regresion simbolica.
- `Informacion`: resumen del dispositivo y operadores.

---

## Scripts Utiles

Ejecutalos desde la raiz del repositorio para conservar imports limpios:

```powershell
python AlphaSymbolic\scripts\run_gpu_console.py
```

Ejecuta una busqueda GPU tipo consola usando la configuracion de `core/gpu/config.py`.

```powershell
python AlphaSymbolic\scripts\run_gpu_console.py --verbose
```

```powershell
python AlphaSymbolic\scripts\run_gpu_benchmark.py
```

Ejecuta benchmarks sinteticos definidos en el propio script.

```powershell
python AlphaSymbolic\scripts\run_gpu_benchmark.py --verbose --timeout 10 --pop-size 1000
```

```powershell
python AlphaSymbolic\scripts\profile_gpu_engine.py
```

Perfila rutas del motor GPU.

```powershell
python AlphaSymbolic\scripts\infinite_search.py
```

Lanza una busqueda continua con memoria de patrones y mutacion estructural.

---

## Configuracion

El archivo principal del motor GPU es:

```text
AlphaSymbolic/core/gpu/config.py
```

Parametros frecuentes:

| Parametro | Descripcion |
|-----------|-------------|
| `USE_FLOAT32` | Usa `float32` para ganar velocidad en GPUs de consumo. |
| `FORCE_CPU_MODE` | Fuerza CPU aunque CUDA este disponible. |
| `USE_CUDA_ORCHESTRATOR` | Activa el orquestador CUDA nativo. |
| `USE_LOG_TRANSFORMATION` | Transforma `Y` con log. Importante para N-Queens; puede ser incorrecto para datos simples. |
| `POP_SIZE` | Tamano global de poblacion. |
| `NUM_ISLANDS` | Cantidad de islas evolutivas. |
| `MAX_FORMULA_LENGTH` | Longitud maxima de formula. |
| `MAX_CONSTANTS` | Constantes disponibles por individuo. |
| `USE_INITIAL_FORMULA` | Inyecta una formula inicial si esta activo. |
| `INITIAL_FORMULA_STRING` | Formula semilla. |

La configuracion actual esta optimizada para una busqueda sobre N-Queens/OEIS A000170 con variables derivadas de `n`, `n % 6` y `n % 2`.

---

## Extension CUDA Nativa

La extension vive en:

```text
AlphaSymbolic/core/gpu/cuda/
```

Archivos clave:

- `setup.py`
- `bindings.cpp`
- `rpn_kernels.cu`
- `pso_kernels.cu`
- `fused_pso_kernels.cu`
- `simplify_kernels.cu`
- `lbfgs_kernels.cu`
- `best_tracker_kernels.cu`

Si ya existe `rpn_cuda_native.cp311-win_amd64.pyd`, fue compilada para Python 3.11 en Windows.

Para recompilar manualmente:

```powershell
cd AlphaSymbolic\core\gpu\cuda
python setup.py build_ext --inplace
```

Requisitos para recompilar en Windows:

- Visual Studio 2022 con C++.
- CUDA Toolkit compatible.
- PyTorch instalado.

---

## Smoke Test Rapido

Desde la raiz, en PowerShell:

```powershell
@'
import os, sys
sys.path.insert(0, os.getcwd())
from AlphaSymbolic.core.gpu import TensorGeneticEngine
engine = TensorGeneticEngine(num_variables=1, pop_size=256, n_islands=2, max_len=16, max_constants=4)
print(engine.device, engine.pop_size, engine.n_islands)
'@ | python -
```

---

## Tests de UI

Instala dependencias de desarrollo:

```powershell
pip install -r AlphaSymbolic\requirements-dev.txt
python -m playwright install chromium
```

Ejecuta la suite oficial de UI/E2E:

```powershell
python -m pytest AlphaSymbolic\tests AlphaSymbolic\tests\ui
python -m pytest AlphaSymbolic\tests\e2e --browser chromium
```

La carpeta `tests/` puede contener pruebas legacy locales ignoradas por git. La coleccion oficial queda acotada por `AlphaSymbolic/tests/conftest.py`.

## Estructura

```text
AlphaSymbolic/
|-- app.py
|-- requirements.txt
|-- pyproject.toml
|-- core/
|   |-- model.py
|   |-- grammar.py
|   |-- gpu_engine.py
|   `-- gpu/
|       |-- engine.py
|       |-- config.py
|       |-- operators.py
|       |-- evaluation.py
|       |-- optimization.py
|       |-- gpu_simplifier.py
|       `-- cuda/
|-- search/
|-- scripts/
|-- ui/
`-- utils/
```
