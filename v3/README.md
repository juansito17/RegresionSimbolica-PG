# Fórmula Genética - v3 (GPU / CUDA)

Esta versión implementa Regresión Simbólica usando Programación Genética con un modelo de islas, acelerado mediante CUDA para ejecución en GPUs NVIDIA.

## Características

*   Programación Genética para Regresión Simbólica
*   Modelo de Islas para evolución paralela y migración
*   Aceleración CUDA para evaluación masivamente paralela de aptitud
*   Parámetros evolutivos adaptativos
*   Optimización Pareto para soluciones multi-objetivo (precisión vs. complejidad)
*   Memoria de patrones para aprender sub-estructuras exitosas
*   Restricciones de dominio para guiar la evolución

## Dependencias

*   Compilador compatible con C++17 (ej., g++)
*   CMake (versión 3.18 o superior recomendada para soporte CUDA)
*   Sistema de construcción Make o Ninja (Ninja se usa en el ejemplo)
*   NVIDIA CUDA Toolkit (compatible con tu versión de driver)
*   GPU NVIDIA con Capacidad de Cómputo 3.5 o superior (revisa `fitness_kernels.cu` para requisitos específicos)

## Compilación

1.  **Navega al directorio `v3`:**
    ```bash
    cd v3
    ```
2.  **Crea un directorio de construcción:**
    ```bash
    mkdir build
    cd build
    ```
3.  **Configura usando CMake:**
    *   Asegúrate que CMake encuentre tu instalación CUDA (usualmente lo hace automáticamente).
    *   Usando Ninja (si está instalado):
        ```bash
        cmake .. -G "Ninja"
        ```
    *   Usando Make:
        ```bash
        cmake ..
        ```
    *   *Solución de problemas:* Si CMake no encuentra CUDA, podrías necesitar establecer la variable de entorno `CUDAToolkit_ROOT` o la variable CMake.
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

El programa mostrará el progreso del algoritmo genético, aprovechando la GPU para los cálculos de aptitud. Los datos objetivo (X_VALUES, TARGETS) pueden modificarse en `src/Globals.h`.

## Estructura del Código

*   `src/`: Contiene el código host en C++.
    *   `main.cpp`: Punto de entrada.
    *   `Globals.h`: Constantes y parámetros globales.
    *   `ExpressionTree.h/.cpp`: Define la estructura del árbol y lógica de aplanamiento.
    *   `GeneticOperators.h/.cpp`: Implementa selección, cruce, mutación.
    *   `Fitness.h/.cpp`: Configuración de aptitud del lado host y llamadas a CUDA.
    *   `GeneticAlgorithm.h/.cpp`: Orquestación principal del AG (modelo de islas).
    *   `AdvancedFeatures.h/.cpp`: Optimización Pareto, memoria de patrones, etc.
*   `cuda/`: Contiene código kernel CUDA.
    *   `fitness_kernels.cu`: Kernel CUDA para evaluar aptitud del árbol.
    *   `FitnessCuda.h`: Wrapper estilo C para lanzar el kernel desde C++.
*   `include/`: Encabezados compartidos (como definiciones CUDA).
    *   `cuda_defs.h`: Constantes CUDA y macros/funciones auxiliares.
*   `CMakeLists.txt`: Archivo de configuración de construcción.
