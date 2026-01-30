# 🧬 Fórmula Genética

> **Regresión Simbólica con Programación Genética y Aceleración GPU**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O2iCFqhXckKg4XF1ZCvpXO_gqt4fmkEI?usp=sharing)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=cplusplus)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![OpenMP](https://img.shields.io/badge/OpenMP-Parallel-00ADD8.svg)](https://www.openmp.org/)

Un sistema de **Programación Genética** de alto rendimiento diseñado para descubrir fórmulas matemáticas a partir de datos numéricos. Combina técnicas evolutivas avanzadas con aceleración por GPU (CUDA) y CPU multi-hilo (OpenMP).

---

## 📑 Tabla de Contenidos

- [🎯 Características Principales](#-características-principales)
- [🏗️ Arquitectura](#️-arquitectura)
- [⚡ Requisitos](#-requisitos)
- [🚀 Instalación y Compilación](#-instalación-y-compilación)
- [💻 Uso](#-uso)
- [⚙️ Configuración](#️-configuración)
- [📊 Ejemplo de Salida](#-ejemplo-de-salida)
- [🔧 Estructura del Proyecto](#-estructura-del-proyecto)
- [👤 Autor](#-autor)
- [📄 Licencia](#-licencia)

---

## 🎯 Características Principales

### Algoritmo Evolutivo
| Característica | Descripción |
|----------------|-------------|
| **Modelo de Islas** | Múltiples poblaciones evolucionan en paralelo con migración periódica |
| **Selección por Torneo** | Con presión de parsimonia para favorecer soluciones simples |
| **Cruce de Subárboles** | Intercambio de ramas entre árboles de expresión |
| **Mutación Múltiple** | Cambio de constantes, operadores, inserción/deleción de nodos |
| **Parámetros Adaptativos** | Tasas de mutación y cruce que evolucionan durante la ejecución |

### Rendimiento
| Característica | Descripción |
|----------------|-------------|
| **Aceleración GPU (CUDA)** | Evaluación masivamente paralela del fitness en GPU NVIDIA |
| **Paralelismo CPU (OpenMP)** | Fallback multi-hilo para sistemas sin GPU |
| **Compilación Condicional** | Soporte automático GPU/CPU según disponibilidad |

### Optimización Inteligente
| Característica | Descripción |
|----------------|-------------|
| **Simplificación Algebraica** | Plegado de constantes e identidades matemáticas automático |
| **Optimización Pareto** | Balance entre precisión y complejidad de la fórmula |
| **Memoria de Patrones** | Reutilización de sub-estructuras exitosas |
| **Búsqueda Local** | Refinamiento de las mejores soluciones encontradas |
| **Detección de Estancamiento** | Inyección de diversidad o terminación temprana |

### Flexibilidad
| Característica | Descripción |
|----------------|-------------|
| **Parser de Fórmulas** | Conversión de texto a árbol de expresión |
| **Inyección de Fórmula Inicial** | Punto de partida opcional para la evolución |
| **Función de Fitness Configurable** | RMSE o error potencial con penalización por complejidad |

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                    ALGORITMO GENÉTICO                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐       ┌─────────┐       │
│  │ Isla 0  │  │ Isla 1  │  │ Isla 2  │  ...  │ Isla N  │       │
│  │         │  │         │  │         │       │         │       │
│  │ Pop[k]  │  │ Pop[k]  │  │ Pop[k]  │       │ Pop[k]  │       │
│  └────┬────┘  └────┬────┘  └────┬────┘       └────┬────┘       │
│       │            │            │                  │            │
│       └────────────┴─────┬──────┴──────────────────┘            │
│                          │                                      │
│                    ┌─────▼─────┐                                │
│                    │ Migración │ (cada N generaciones)          │
│                    └───────────┘                                │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    EVALUACIÓN DE FITNESS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐    ┌──────────────────────┐          │
│  │     GPU (CUDA)       │ OR │    CPU (OpenMP)      │          │
│  │  Evaluación Batch    │    │  Evaluación Paralela │          │
│  └──────────────────────┘    └──────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Árbol de Expresión

Las fórmulas se representan como árboles binarios:

```
        [+]
       /   \
     [*]   [5]
    /   \
  [x]   [2]

  Representa: (x * 2) + 5
```

**Operadores soportados:** `+`, `-`, `*`, `/`, `^` (potencia)

---

## ⚡ Requisitos

### Obligatorios
- **Compilador C++17** (MSVC, g++, clang++)
- **CMake** ≥ 3.18
- **OpenMP** (incluido en la mayoría de compiladores)

### Opcionales (para aceleración GPU)
- **NVIDIA GPU** con Compute Capability ≥ 5.0
- **CUDA Toolkit** ≥ 11.0
- **Driver NVIDIA** actualizado

---

## 🚀 Instalación y Compilación

### Opción 1: Con GPU (CUDA)

```bash
# Clonar el repositorio
git clone https://github.com/juansito17/RegresionSimbolica-PG.git
cd Algoritmo-Genetico-de-Formulas/Code

# Crear directorio de compilación
mkdir build && cd build

# Configurar con CMake (detecta CUDA automáticamente)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compilar
cmake --build . --config Release
```

### Opción 2: Solo CPU (sin CUDA)

Para compilar sin aceleración GPU, comenta o elimina la línea 55 en `CMakeLists.txt`:

```cmake
# target_compile_definitions(SymbolicRegressionGP PUBLIC "USE_GPU_ACCELERATION_DEFINED_BY_CMAKE")
```

Luego sigue los mismos pasos de compilación.

### Windows (Script Rápido)

```batch
cd Code
.\run.bat
```

---

## 💻 Uso

### Ejecución Básica

```bash
# Desde el directorio build
./SymbolicRegressionGP        # Linux/macOS
.\SymbolicRegressionGP.exe    # Windows
```

### Configuración de Datos

Edita `Code/src/Globals.h` para definir tus datos de entrada:

```cpp
// Valores objetivo (Y)
const std::vector<double> TARGETS = {92, 352, 724};

// Valores de entrada (X)
const std::vector<double> X_VALUES = {8, 9, 10};
```

### Inyección de Fórmula Inicial (Opcional)

Si tienes una aproximación inicial de la fórmula:

```cpp
const bool USE_INITIAL_FORMULA = true;
const std::string INITIAL_FORMULA_STRING = "x^2 + 5*x - 3";
```

---

## ⚙️ Configuración

Los parámetros principales se encuentran en `Code/src/Globals.h`:

### Parámetros del Algoritmo

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `TOTAL_POPULATION_SIZE` | 50,000 | Tamaño total de la población |
| `GENERATIONS` | 500,000 | Número máximo de generaciones |
| `NUM_ISLANDS` | 10 | Número de islas paralelas |
| `MIGRATION_INTERVAL` | 100 | Generaciones entre migraciones |
| `MIGRATION_SIZE` | 50 | Individuos intercambiados |

### Parámetros de Evolución

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `BASE_MUTATION_RATE` | 0.30 | Tasa de mutación base |
| `DEFAULT_CROSSOVER_RATE` | 0.85 | Tasa de cruce |
| `DEFAULT_TOURNAMENT_SIZE` | 30 | Tamaño del torneo de selección |
| `BASE_ELITE_PERCENTAGE` | 0.15 | Porcentaje de élite preservada |

### Parámetros de Fitness

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `USE_RMSE_FITNESS` | `true` | Usar RMSE como métrica |
| `COMPLEXITY_PENALTY_FACTOR` | 0.005 | Penalización por complejidad |
| `EXACT_SOLUTION_THRESHOLD` | 1e-8 | Umbral de solución exacta |

---

## 📊 Ejemplo de Salida

```
Info: Running with 10 islands, 5000 individuals per island.
Evaluating initial population (simplifying all)...
Initial best fitness: 1.23456789e+02
Initial best formula size: 5
Initial best formula: ((x * x) + (x * 3))
----------------------------------------
Starting Genetic Algorithm...

========================================
New Global Best Found (Gen 127, Island 3)
Fitness: 0.00000142
Size: 7
Formula: ((x ^ 2) + ((x * 5) - 3))
Predictions vs Targets:
  x=   8.0000: Pred=      92.0000, Target=      92.0000, Diff=      0.0000
  x=   9.0000: Pred=     352.0001, Target=     352.0000, Diff=      0.0001
  x=  10.0000: Pred=     724.0000, Target=     724.0000, Diff=      0.0000
========================================

--- Generation 200/500000 (Elapsed: 12.45s) ---
Overall Best Fitness: 1.42000000e-06
Best Formula Size: 7
(Last improvement at gen: 127)
```

---

## 🔧 Estructura del Proyecto

```
Algoritmo-Genetico-de-Formulas/
├── AlphaSymbolic/            # Unified Neuro-Symbolic implementation
├── Code/                     # Core C++ (GP) implementation
│   ├── src/                  # Source files
│   ├── notebooks/            # Notebooks (Colab)
│   ├── scripts/              # Helper scripts
│   └── ...
├── LICENSE                   # Apache 2.0
└── README.md                 # This file
```

---

## 👤 Autor

**Juan Manuel Peña Usuga**

- 🎓 Estudiante de Ingeniería Informática (Quinto Semestre)
- 🏛️ Politécnico Colombiano Jaime Isaza Cadavid
- 📅 Última actualización: Enero 2026

---

## 📄 Licencia

Este proyecto está licenciado bajo la **Licencia Apache 2.0** - ver el archivo [LICENSE](LICENSE) para más detalles.

```
Copyright 2026 Juan Manuel Peña Usuga

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

---

<div align="center">

**⭐ Si este proyecto te resulta útil, considera darle una estrella ⭐**

</div>
