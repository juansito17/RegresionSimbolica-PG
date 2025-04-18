# Fórmula Genética - Versión CPU / OpenMP

Esta carpeta (`Code`) contiene la implementación para Regresión Simbólica usando Programación Genética con un modelo de islas, ejecutándose en CPU y utilizando OpenMP para evaluación paralela de aptitud.

## Características

* Programación Genética para Regresión Simbólica.
* Modelo de Islas para evolución paralela y migración.
* Paralelización OpenMP para evaluación de aptitud.
* Parámetros de evolución adaptativos.
* Optimización Pareto para soluciones multi-objetivo (precisión vs. complejidad).
* Memoria de patrones para aprender sub-estructuras exitosas.
* Restricciones de dominio y simplificación de árboles para guiar la evolución.
* Búsqueda local para mejorar las soluciones encontradas.

## Dependencias

* Compilador compatible con C++17 (ej., g++)
* CMake (versión 3.10 o superior recomendada)
* Sistema de construcción como Make o Ninja (Ninja se usa en el ejemplo)
* Soporte OpenMP habilitado en tu compilador

## Construcción

1.  **Navega al directorio raíz del proyecto (`Algoritmo-Genetico-de-Formulas`).**
2.  **Crea un directorio de construcción dentro de `Code` (si no existe):**
    ```bash
    mkdir -p Code/build
    cd Code/build
    ```
3.  **Configura usando CMake (desde dentro de `Code/build`):**
    * Usando Ninja (si está instalado):
        ```bash
        cmake .. -G "Ninja"
        ```
    * Usando Make (opción por defecto):
        ```bash
        cmake ..
        ```
    * Para una build de Release optimizada (recomendado):
        ```bash
        # Con Ninja
        cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
        # Con Make
        cmake .. -DCMAKE_BUILD_TYPE=Release
        ```

4.  **Compila el proyecto (desde dentro de `Code/build`):**
    * Usando Ninja:
        ```bash
        ninja
        ```
    * Usando Make:
        ```bash
        make
        ```
    Esto creará un ejecutable (ej., `SymbolicRegressionGP` o `SymbolicRegressionGP.exe`) dentro del directorio `Code/build`.

## Ejecución

Ejecuta el programa compilado desde el directorio `Code/build`:

```bash
./SymbolicRegressionGP