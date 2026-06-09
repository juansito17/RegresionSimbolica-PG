# Formula Genetica

> Regresion simbolica con programacion genetica, busqueda neuro-simbolica y aceleracion GPU.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=cplusplus)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![OpenMP](https://img.shields.io/badge/OpenMP-Parallel-00ADD8.svg)](https://www.openmp.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?logo=python)](https://www.python.org/)

Este repositorio contiene dos implementaciones complementarias para descubrir formulas matematicas a partir de datos numericos:

- `AlphaSymbolic/`: aplicacion Python/Gradio con PyTorch, busqueda hibrida, motor genetico tensorial en GPU y extension CUDA nativa.
- `Code/`: motor C++17 con CUDA/OpenMP, modelo de islas, operadores geneticos, fitness y suite de pruebas.

El frente mas activo actualmente es `AlphaSymbolic`, especialmente el motor GPU y los scripts de busqueda continua.

---

## Estado Actual

- Rama principal de trabajo: `develop`.
- Python probado localmente: `3.11`.
- PyTorch con CUDA detectado: `torch 2.5.1+cu121`.
- Extension CUDA Python existente: `AlphaSymbolic/core/gpu/cuda/rpn_cuda_native.cp311-win_amd64.pyd`.
- CMake detecta CUDA y genera build GPU para el motor C++.
- Tests C++ disponibles en el target `TestOperators`.

---

## Estructura

```text
Algoritmo-Genetico-de-Formulas/
|-- AlphaSymbolic/
|   |-- app.py                         # App Gradio
|   |-- core/                          # Modelo, gramatica y motor GPU
|   |-- core/gpu/config.py             # Configuracion principal del motor GPU Python
|   |-- core/gpu/cuda/                 # Extension CUDA nativa
|   |-- scripts/run_gpu_console.py     # Busqueda GPU desde consola
|   |-- scripts/run_gpu_benchmark.py   # Benchmarks sinteticos
|   |-- scripts/infinite_search.py     # Busqueda continua
|   |-- search/                        # Beam Search, MCTS e hibridos
|   |-- ui/                            # Componentes Gradio
|   `-- utils/
|-- Code/
|   |-- CMakeLists.txt
|   |-- run.bat
|   |-- scripts/run_tests.bat
|   `-- src/
|       |-- Globals.h                  # Configuracion C++
|       |-- main.cpp                   # Entry point C++
|       |-- GeneticAlgorithm.*
|       |-- ExpressionTree.*
|       |-- Fitness.*
|       |-- *GPU.cu / *GPU.cuh
|       `-- TestOperators.cpp
|-- LICENSE
`-- README.md
```

---

## Requisitos

### Python / AlphaSymbolic

- Python 3.9 o superior.
- PyTorch.
- NumPy, SciPy, SymPy, Gradio y Pandas.
- Opcional pero recomendado: GPU NVIDIA con driver actualizado.
- Para recompilar la extension nativa: CUDA Toolkit y Visual Studio 2022 en Windows.

Instalacion basica:

```powershell
cd AlphaSymbolic
pip install -r requirements.txt
```

Si necesitas instalar PyTorch con CUDA manualmente, usa la rueda apropiada desde la documentacion oficial de PyTorch para tu version de CUDA/driver.

### C++ / Code

- CMake 3.18 o superior.
- Compilador C++17.
- OpenMP.
- Opcional: CUDA Toolkit para build GPU.
- En Windows, Visual Studio 2022 funciona con el `run.bat` incluido.

---

## Uso Rapido

### 1. Abrir la app web

```powershell
cd AlphaSymbolic
python app.py
```

La app lanza Gradio y abre el navegador. Si no se abre automaticamente, revisa la URL que imprime la consola, normalmente `http://127.0.0.1:7860`.

### 2. Ejecutar busqueda GPU desde consola

Desde la raiz del repositorio:

```powershell
python AlphaSymbolic\scripts\run_gpu_console.py
```

Este script usa `AlphaSymbolic/core/gpu/config.py` y, por defecto, esta orientado al problema N-Queens/OEIS A000170.

### 3. Ejecutar benchmark GPU

```powershell
python AlphaSymbolic\scripts\run_gpu_benchmark.py
```

### 4. Compilar el motor C++

```powershell
cd Code
cmake -S . -B build -G "Visual Studio 17 2022"
cmake --build build --config Debug
```

El ejecutable queda en:

```text
Code/build/Debug/SymbolicRegressionGP.exe
```

### 5. Ejecutar pruebas C++

```powershell
cd Code
cmake --build build --target TestOperators --config Debug
.\build\Debug\TestOperators.exe
```

Tambien puedes usar:

```powershell
cd Code
.\scripts\run_tests.bat
```

---

## Configuracion Importante

### AlphaSymbolic

El archivo principal es:

```text
AlphaSymbolic/core/gpu/config.py
```

Parametros clave:

| Parametro | Uso |
|-----------|-----|
| `USE_LOG_TRANSFORMATION` | Aplica log a `Y`. Esta activo por defecto para el caso N-Queens. Desactivalo para datos lineales/simples. |
| `POP_SIZE` | Tamano de poblacion GPU. Valores grandes consumen mucha VRAM. |
| `NUM_ISLANDS` | Numero de islas evolutivas. |
| `MAX_FORMULA_LENGTH` | Longitud maxima de formulas RPN. |
| `MAX_CONSTANTS` | Numero maximo de constantes optimizables. |
| `USE_CUDA_ORCHESTRATOR` | Usa el orquestador CUDA cuando esta disponible. |
| `PROBLEM_X_START`, `PROBLEM_X_END`, `PROBLEM_Y_FULL` | Dataset actual de trabajo. |

Para probar datos personalizados desde la app, usa la pestana de busqueda y desactiva transformaciones que no apliquen a tu problema.

### C++

El archivo principal es:

```text
Code/src/Globals.h
```

El motor C++ tambien esta configurado actualmente para un caso N-Queens multivariable. `X_VALUES` es `vector<vector<double>>`, no un vector univariable simple.

Tambien puedes pasar datos por archivo al ejecutable:

```powershell
.\build\Debug\SymbolicRegressionGP.exe --data datos.txt --seed semillas.txt
```

Formato esperado de `datos.txt`:

```text
x0_1 x0_2 x0_3 ...
x1_1 x1_2 x1_3 ...
y_1  y_2  y_3  ...
```

La ultima linea es `Y`; las anteriores son variables de entrada.

---

## Operadores

El sistema soporta operadores aritmeticos y funciones avanzadas segun la configuracion activa:

- Aritmeticos: `+`, `-`, `*`, `/`, potencia.
- Funciones: `sin`, `cos`, `log`, `exp`, `sqrt`, `lgamma/fact` y otras, segun flags.
- Optimizacion de constantes con PSO/L-BFGS en la parte GPU Python.
- Simplificacion simbolica y validacion estricta de dominio en varios caminos del motor.

---

## Notas de Mantenimiento

- `Code/build/`, caches, modelos y resultados estan ignorados por git.
- La extension `rpn_cuda_native.cp311-win_amd64.pyd` esta ignorada, pero puede existir localmente como artefacto compilado.
- Si cambias Python de version, recompila la extension CUDA porque el sufijo `cp311` solo corresponde a Python 3.11.
- Los README antiguos mencionaban scripts como `run_benchmark_feynman.py`, `rescue_data.py` y `build_extension.bat`; esos nombres no existen actualmente en el arbol del repo.

---

## Licencia

Este proyecto esta licenciado bajo Apache 2.0. Consulta [LICENSE](LICENSE).

Copyright 2026 Juan Manuel Pena Usuga.
