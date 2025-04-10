# Fórmula Genética - Regresión Simbólica con Programación Genética

Este repositorio contiene implementaciones de un sistema de Programación Genética diseñado para tareas de Regresión Simbólica (encontrar fórmulas matemáticas que se ajusten a puntos de datos dados).

## Versiones

Este proyecto incluye múltiples versiones con diferentes características:

* **v2:** Implementación basada en CPU utilizando modelo de islas, OpenMP para paralelismo, y características avanzadas como parámetros adaptativos y optimización de Pareto. Ver [v2/README.md](v2/README.md) para detalles.
* **v3:** Implementación acelerada por GPU (CUDA) basada en modelo de islas, diseñada para mayor rendimiento en hardware compatible. Ver [v3/README.md](v3/README.md) para detalles.

## Características

- Modelo de islas para diversidad poblacional
- Paralelización con OpenMP y CUDA
- Parámetros genéticos adaptativos
- Optimización multi-objetivo basada en Pareto
- Capacidades de regresión simbólica

## Licencia

Este proyecto está licenciado bajo la Licencia Apache 2.0 - ver el archivo [LICENSE](LICENSE) para más detalles.
