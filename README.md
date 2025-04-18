# Fórmula Genética - Regresión Simbólica con Programación Genética

Este repositorio contiene una implementación de un sistema de Programación Genética diseñado para tareas de Regresión Simbólica (encontrar fórmulas matemáticas que se ajusten a puntos de datos dados).

## Implementación Actual

El directorio `Code` contiene la implementación actual:

* **Versión CPU/OpenMP:** Implementación basada en CPU utilizando un modelo de islas, OpenMP para paralelismo, y características avanzadas como parámetros adaptativos y optimización de Pareto. Ver [Code/README.md](Code/README.md) para detalles específicos de esta versión.

*(Nota: Originalmente se planeó una versión GPU (v3), pero actualmente solo la versión CPU está presente en el directorio `Code`)*

## Características

* Programación Genética para Regresión Simbólica.
* Modelo de islas para diversidad poblacional y evolución paralela.
* Paralelización con OpenMP para evaluación de aptitud en CPU.
* Parámetros genéticos adaptativos.
* Optimización multi-objetivo basada en Pareto (precisión vs. complejidad).
* Memoria de patrones para aprender sub-estructuras exitosas.
* Restricciones de dominio y simplificación de árboles.
* Búsqueda local para refinamiento de soluciones.

## Autor

* **Juan Manuel Peña Usuga**
* Estudiante de Ingeniería Informática (Quinto Semestre)
* Politécnico Colombiano Jaime Isaza Cadavid
* Fecha de actualización: 2025-04-17

## Licencia

Este proyecto está licenciado bajo la Licencia Apache 2.0 - ver el archivo `LICENSE` para más detalles (si existe, de lo contrario, considera añadir uno).