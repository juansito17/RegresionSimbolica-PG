# 🧬 Fórmula Genética - Implementación CPU/GPU

> Módulo de ejecución con soporte para **CUDA** y **OpenMP**

Esta carpeta contiene el código fuente y los archivos de compilación para el sistema de Regresión Simbólica.

---

## 📋 Resumen

| Componente | Descripción |
|------------|-------------|
| **Lenguaje** | C++17 |
| **Paralelismo** | CUDA (GPU) + OpenMP (CPU) |
| **Build System** | CMake ≥ 3.18 |
| **Compiladores** | MSVC, g++, clang++, nvcc |

---

## 📦 Dependencias

### Obligatorias

| Dependencia | Versión Mínima | Notas |
|-------------|----------------|-------|
| Compilador C++ | C++17 | MSVC 2019+, GCC 8+, Clang 10+ |
| CMake | 3.18 | Para mejor soporte de CUDA |
| OpenMP | 4.0+ | Incluido en la mayoría de compiladores |

### Opcionales (GPU)

| Dependencia | Versión Mínima | Notas |
|-------------|----------------|-------|
| CUDA Toolkit | 11.0 | Para aceleración GPU |
| GPU NVIDIA | CC 5.0+ | Maxwell o superior |

---

## 🔨 Compilación

### Windows (Recomendado)

El script `run.bat` automatiza la configuración, compilación y ejecución:

```batch
cd Code
.\run.bat
```

### Windows (Manual)

```batch
cd Code
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

### Linux/macOS

```bash
cd Code
mkdir -p build && cd build

# Con GPU (si CUDA está instalado)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compilar
cmake --build . -j$(nproc)
```

### Compilación Solo CPU (Sin CUDA)

1. Edita `CMakeLists.txt` y comenta la línea 55:

   ```cmake
   # target_compile_definitions(SymbolicRegressionGP PUBLIC "USE_GPU_ACCELERATION_DEFINED_BY_CMAKE")
   ```

2. Reconfigura y compila:

   ```bash
   cd build
   cmake ..
   cmake --build .
   ```

---

## ▶️ Ejecución

### Desde el directorio `build`

```bash
# Linux/macOS
./SymbolicRegressionGP

# Windows
.\SymbolicRegressionGP.exe
```

### Usando el script (Windows)

```batch
.\run.bat
```

---

## 🗂️ Estructura de Archivos

```
Code/
├── src/
│   ├── main.cpp              # Punto de entrada
│   ├── GeneticAlgorithm.cpp  # Algoritmo genético principal
│   ├── GeneticAlgorithm.h
│   ├── GeneticOperators.cpp  # Operadores: selección, cruce, mutación
│   ├── GeneticOperators.h
│   ├── ExpressionTree.cpp    # Representación de árboles
│   ├── ExpressionTree.h
│   ├── Fitness.cpp           # Evaluación de fitness (CPU)
│   ├── Fitness.h
│   ├── FitnessGPU.cu         # Evaluación de fitness (GPU/CUDA)
│   ├── FitnessGPU.cuh
│   ├── AdvancedFeatures.cpp  # Pareto, patrones, simplificación
│   ├── AdvancedFeatures.h
│   └── Globals.h             # ⚙️ Parámetros configurables
├── tests/                    # Tests unitarios
├── build/                    # Archivos de compilación (generado)
├── CMakeLists.txt            # Configuración de CMake
├── run.bat                   # Script de ejecución automática
└── README.md                 # Este archivo
```

---

## ⚙️ Configuración Rápida

Edita `src/Globals.h` para ajustar:

### Datos de Entrada

```cpp
const std::vector<double> TARGETS = {92, 352, 724};  // Valores Y
const std::vector<double> X_VALUES = {8, 9, 10};     // Valores X
```

### Parámetros del Algoritmo

```cpp
const int TOTAL_POPULATION_SIZE = 50000;  // Tamaño de población
const int GENERATIONS = 500000;           // Generaciones máximas
const int NUM_ISLANDS = 10;               // Islas paralelas
```

### Fórmula Inicial (Opcional)

```cpp
const bool USE_INITIAL_FORMULA = true;
const std::string INITIAL_FORMULA_STRING = "x^2 + 5*x";
```

---

## 🔗 Documentación Adicional

Para más detalles sobre la arquitectura, características avanzadas y guía de uso completa, consulta el [README principal](../README.md).