# AlphaSymbolic

> Capa legacy Python/Gradio para regresión simbólica neuro-evolutiva con PyTorch y GPU.

Este documento describe la capa experimental que conserva `AlphaSymbolic`. La
documentación vigente del proyecto está en [`README.md`](../../README.md) y el
núcleo GPU oficial vive en `src/warpsymbolic`.

AlphaSymbolic combina varios enfoques:

- Modelo neuronal Transformer para generar candidatos.
- Beam Search, MCTS y busqueda hibrida.
- Motor genetico tensorial en GPU (`TensorGeneticEngine`).
- Extension CUDA nativa para evaluacion, mutacion, simplificacion y optimizacion de constantes.
- Interfaz web en Gradio para busqueda, entrenamiento, benchmark y monitoreo.

---

## Instalacion

Desde la raíz del repositorio:

```powershell
python -m pip install -e ".[dev]"
```

La aplicación legacy requiere además las dependencias de Gradio y PyTorch del
entorno de ejecución. El archivo `pyproject.toml` raíz es la única fuente de
configuración del paquete.

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
python -m AlphaSymbolic.app
```

Por defecto la app corre en modo local. Para URL publica de Gradio o logs detallados:

```powershell
python -m AlphaSymbolic.app --share
python -m AlphaSymbolic.app --verbose
```

Tambien puedes activar logs detallados con:

```powershell
$env:ALPHASYMBOLIC_VERBOSE="1"
python -m AlphaSymbolic.app
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
python -m warpsymbolic.cli.run_gpu_console
```

Ejecuta una búsqueda GPU tipo consola usando la configuración de
`src/warpsymbolic/gpu/config.py`.

```powershell
python -m warpsymbolic.cli.run_gpu_console --verbose
```

```powershell
python -m warpsymbolic.cli.run_gpu_benchmark
```

Ejecuta benchmarks sinteticos definidos en el propio script.

```powershell
python -m warpsymbolic.cli.run_gpu_benchmark --verbose --timeout 10 --pop-size 1000
```

```powershell
python -m warpsymbolic.cli.profile_gpu_engine
```

Perfila rutas del motor GPU.

```powershell
python -m warpsymbolic.cli.benchmark_scientific --suite all --methods alphasymbolic
```

Ejecuta Nguyen, Feynman, Friedman y N-Reinas con semillas registradas, datos
train/test independientes y métricas estrictas fuera de muestra. PySR es un
comparador opcional mediante `--methods alphasymbolic,pysr`.

```powershell
python -m warpsymbolic.cli.infinite_search
```

Lanza una busqueda continua con memoria de patrones y mutacion estructural.

---

## Configuracion

El archivo principal del motor GPU es:

```text
src/warpsymbolic/gpu/config.py
```

Parametros frecuentes:

| Parametro | Descripcion |
|-----------|-------------|
| `USE_FLOAT32` | Usa `float32` para ganar velocidad en GPUs de consumo. |
| `FORCE_CPU_MODE` | Fuerza CPU aunque CUDA este disponible. |
| `USE_CUDA_ORCHESTRATOR` | Activa el orquestador CUDA nativo. |
| `USE_LOG_TRANSFORMATION` | Transforma `Y` con log. Su valor seguro por defecto es `False`; N-Reinas lo activa explícitamente. |
| `POP_SIZE` | Tamano global de poblacion. |
| `NUM_ISLANDS` | Cantidad de islas evolutivas. |
| `MAX_FORMULA_LENGTH` | Longitud maxima de formula. |
| `MAX_CONSTANTS` | Constantes disponibles por individuo. |
| `USE_INITIAL_FORMULA` | Inyecta una formula inicial si esta activo. |
| `INITIAL_FORMULA_STRING` | Formula semilla. |

La configuración genérica usa operadores científicos comunes y no transforma
el objetivo implícitamente. Los scripts de N-Reinas seleccionan de forma
explícita su perfil combinatorio (`fact`, `gamma`, `lgamma`) y el objetivo
logarítmico.

El informe reproducible de rendimiento, convergencia, corrección CUDA y
limitaciones está en
[`docs/GPU_ENGINE_AUDIT_2026-07-26.md`](../GPU_ENGINE_AUDIT_2026-07-26.md).

---

## Extension CUDA Nativa

La extension vive en:

```text
src/warpsymbolic/gpu/cuda/
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
Push-Location src/warpsymbolic/gpu/cuda
python setup.py build_ext --build-temp ../../../../.local/build/cuda/temp --build-lib ../../../../.local/build/python
Pop-Location
```

Los artefactos de compilación se guardan en `.local/`, fuera del código fuente.

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
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from warpsymbolic.gpu import TensorGeneticEngine
engine = TensorGeneticEngine(num_variables=1, pop_size=256, n_islands=2, max_len=16, max_constants=4)
print(engine.device, engine.pop_size, engine.n_islands)
'@ | python -
```

---

## Tests de UI

Instala dependencias de desarrollo desde la raíz:

```powershell
python -m pip install -e ".[dev]"
python -m playwright install chromium
```

Ejecuta la suite oficial de UI/E2E:

```powershell
python -m pytest tests/integration tests/e2e
python -m pytest tests/e2e --browser chromium
```

La suite oficial vive en `tests/`. Las pruebas específicas de CUDA se ejecutan
con el marcador `gpu` y requieren una instalación compatible con CUDA.

## Estructura

```text
src/
|-- warpsymbolic/
|   |-- api/
|   |-- symbolic/
|   |-- gpu/
|   `-- cli/
`-- AlphaSymbolic/
    |-- app.py
    |-- ui/
    |-- experimental/
    |-- data/
    |-- benchmarking/
    `-- scripts/
```
