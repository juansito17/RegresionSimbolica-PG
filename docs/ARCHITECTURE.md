# Arquitectura de WarpSymbolic

`warpsymbolic` es el paquete de producción. Contiene la API, la representación
simbólica, el camino GPU oficial y sus comandos operativos.

`warpsymbolic.gpu` es el único camino oficial para la evolución de población.
Contiene la configuración, representación RPN GPU, evaluación, operadores,
selección, optimización de constantes, Pareto, memoria de patrones y kernels
CUDA.

`warpsymbolic.symbolic` contiene la gramática y las operaciones de fórmulas
que pueden reutilizar la API y los backends. El paquete GPU puede depender de
esta capa, pero no de `AlphaSymbolic.experimental`.

`warpsymbolic.cli` contiene la consola, SRBench y benchmarks GPU del camino
principal. No es una aplicación alternativa: es la interfaz operativa del
motor GPU.

`AlphaSymbolic` conserva la aplicación, UI, datos, redes neuronales, Beam
Search, MCTS, búsqueda híbrida y el puente al antiguo ejecutable C++. Estas
rutas pueden proponer experimentos, pero no se importan desde el motor GPU
principal.

El motor C++ completo vive en `legacy/cpp_engine` y se construye de forma
independiente con CMake. La extensión CUDA nativa usada por el backend actual
sí permanece dentro de `src/warpsymbolic/gpu/cuda`.
