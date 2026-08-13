# WarpSymbolic legacy C++/CUDA engine

Este directorio contiene el motor C++17 anterior. Es un baseline independiente
y no forma parte del camino oficial Python/CUDA de `src/warpsymbolic`.

## Compilar desde la raíz

```powershell
cmake -S legacy/cpp_engine -B .local/build/cpp_legacy -G "Visual Studio 17 2022"
cmake --build .local/build/cpp_legacy --config Debug
```

Ejecutable:

```text
.local/build/cpp_legacy/Debug/SymbolicRegressionGP.exe
```

Tests:

```powershell
cmake --build .local/build/cpp_legacy --target TestOperators --config Debug
.\.local\build\cpp_legacy\Debug\TestOperators.exe
```

El motor conserva sus árboles de expresión, operadores genéticos, modelo de
islas, fitness CPU/OpenMP, fitness CUDA y optimización de constantes para
comparaciones históricas. Las mejoras nuevas deben implementarse primero en
`src/warpsymbolic/gpu`.
