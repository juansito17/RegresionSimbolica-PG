# Fórmula Genética - v2 (CPU / OpenMP)

Esta versión implementa Regresión Simbólica usando Programación Genética con un modelo de islas, ejecutándose en CPU y utilizando OpenMP para evaluación paralela de aptitud.

## Características

*   Programación Genética para Regresión Simbólica
*   Modelo de Islas para evolución paralela y migración
*   Paralelización OpenMP para evaluación de aptitud
*   Parámetros de evolución adaptativos
*   Optimización Pareto para soluciones multi-objetivo (precisión vs. complejidad)
*   Memoria de patrones para aprender sub-estructuras exitosas
*   Restricciones de dominio para guiar la evolución

## Dependencias

*   Compilador compatible con C++17 (ej., g++)
*   CMake (versión 3.10 o superior recomendada)
*   Sistema de construcción Make o Ninja (Ninja se usa en el ejemplo)
*   Soporte OpenMP en tu compilador

## Construcción

1.  **Navega al directorio `v2`:**
    ```bash
    cd v2
    ```
2.  **Crea un directorio de construcción:**
    ```bash
    mkdir build
    cd build
    ```
3.  **Configura usando CMake:**
    *   Usando Ninja (si está instalado):
        ```bash
        cmake .. -G "Ninja"
        ```
    *   Usando Make:
        ```bash
        cmake ..
        ```
4.  **Compila el proyecto:**
    *   Usando Ninja:
        ```bash
        ninja
        ```
    *   Usando Make:
        ```bash
        make
        ```
    Esto creará un ejecutable (ej., `SymbolicRegressionGP.exe` en Windows) en el directorio `build`.

## Ejecución

Ejecuta el programa compilado desde el directorio `build`:

```bash
./SymbolicRegressionGP
```

El programa mostrará el progreso del algoritmo genético, incluyendo valores de aptitud y la mejor fórmula encontrada. Los datos objetivo (X_VALUES, TARGETS) pueden modificarse en `src/Globals.h`.

## Estructura del Código

*   `src/`: Contiene el código fuente C++.
    *   `main.cpp`: Punto de entrada.
    *   `Globals.h`: Constantes y parámetros globales.
    *   `ExpressionTree.h/.cpp`: Define la estructura de árbol para fórmulas.
    *   `GeneticOperators.h/.cpp`: Implementa selección, cruce, mutación.
    *   `Fitness.h/.cpp`: Lógica de evaluación de aptitud.
    *   `GeneticAlgorithm.h/.cpp`: Orquestación principal del AG (modelo de islas).
    *   `AdvancedFeatures.h/.cpp`: Optimización Pareto, memoria de patrones, etc.
*   `CMakeLists.txt`: Archivo de configuración de construcción.
