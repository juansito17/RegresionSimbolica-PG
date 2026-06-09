# Formula Genetica - Motor C++/CUDA

> Implementacion C++17 del motor genetico para regresion simbolica.

Este modulo contiene el nucleo evolutivo clasico del proyecto:

- Arboles de expresion.
- Operadores geneticos.
- Modelo de islas.
- Fitness CPU/OpenMP y GPU/CUDA.
- Optimizacion de constantes.
- Suite de pruebas para operadores, parsing, simplificacion y fitness.

---

## Dependencias

| Dependencia | Version / nota |
|-------------|----------------|
| CMake | 3.18 o superior |
| C++ | Compilador C++17 |
| OpenMP | Requerido para CPU paralelo |
| CUDA Toolkit | Opcional, requerido para GPU |
| Visual Studio | 2022 recomendado en Windows |

---

## Compilar en Windows

Desde `Code/`:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022"
cmake --build build --config Debug
```

Ejecutable:

```text
Code/build/Debug/SymbolicRegressionGP.exe
```

Para Release:

```powershell
cmake --build build --config Release
```

---

## Script Rapido

```powershell
cd Code
.\run.bat
```

`run.bat` configura Visual Studio, recrea `build/`, compila en `Debug` y ejecuta el programa.

Si tu Visual Studio esta instalado en otra ruta, ajusta `VS_ENV_SCRIPT` dentro del archivo.

---

## Ejecutar

```powershell
cd Code
.\build\Debug\SymbolicRegressionGP.exe
```

Con archivo de datos y semillas:

```powershell
.\build\Debug\SymbolicRegressionGP.exe --data datos.txt --seed semillas.txt
```

Formato de `datos.txt`:

```text
x0_1 x0_2 x0_3 ...
x1_1 x1_2 x1_3 ...
y_1  y_2  y_3  ...
```

La ultima linea es la variable objetivo `Y`; las anteriores son variables de entrada. El programa transpone internamente las variables a muestras multivariable.

Formato de `semillas.txt`:

```text
x0 * 2
log(x0 + 1)
sqrt(x0) + x1
```

Una formula por linea.

---

## Tests

Compilar y ejecutar tests:

```powershell
cd Code
cmake --build build --target TestOperators --config Debug
.\build\Debug\TestOperators.exe
```

O usar el script:

```powershell
cd Code
.\scripts\run_tests.bat
```

La suite cubre:

- Operadores binarios.
- Operadores unarios.
- Casos borde.
- Simplificacion.
- Parsing complejo.
- Calculo de fitness.
- Utilidades de arbol.

---

## Configuracion

Archivo principal:

```text
Code/src/Globals.h
```

Parametros importantes:

| Parametro | Descripcion |
|-----------|-------------|
| `RAW_TARGETS` | Datos objetivo base. |
| `X_VALUES` | Entradas multivariable (`vector<vector<double>>`). |
| `USE_LOG_TRANSFORMATION` | Aplica `log` a targets positivos. |
| `FORCE_CPU_MODE` | Fuerza CPU aunque CMake detecte CUDA. |
| `TOTAL_POPULATION_SIZE` | Tamano global de poblacion. |
| `GENERATIONS` | Maximo de generaciones. |
| `NUM_ISLANDS` | Numero de islas. |
| `USE_INITIAL_FORMULA` | Inyecta formula inicial. |
| `INITIAL_FORMULA_STRING` | Formula semilla. |
| `USE_OP_*` | Flags para habilitar/deshabilitar operadores. |
| `OPERATOR_WEIGHTS` | Pesos de seleccion de operadores. |
| `USE_SIMPLIFICATION` | Activa simplificacion algebraica. |
| `USE_LEXICASE_SELECTION` | Usa seleccion lexicase. |

La configuracion actual no es el ejemplo simple `x -> y`; esta orientada a datos N-Queens multivariable.

---

## Build CPU-only

La deteccion CUDA ocurre automaticamente en `CMakeLists.txt`. Para forzar CPU en runtime, cambia:

```cpp
const bool FORCE_CPU_MODE = true;
```

en `src/Globals.h`.

Si necesitas impedir completamente que CMake compile archivos CUDA, configura el entorno sin `nvcc` disponible o ajusta temporalmente la logica de deteccion en `CMakeLists.txt`.

---

## Estructura

```text
Code/
|-- CMakeLists.txt
|-- run.bat
|-- scripts/
|   |-- run_tests.bat
|   `-- generate_colab_notebook.py
`-- src/
    |-- main.cpp
    |-- Globals.h
    |-- ExpressionTree.cpp / .h
    |-- Fitness.cpp / .h
    |-- FitnessGPU.cu / .cuh
    |-- GeneticAlgorithm.cpp / .h
    |-- GeneticOperators.cpp / .h
    |-- GradientOptimizer.cpp / .h
    |-- GradientOptimizerGPU.cu / .cuh
    |-- AdvancedFeatures.cpp / .h
    `-- TestOperators.cpp
```
