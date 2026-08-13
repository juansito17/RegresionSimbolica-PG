# WarpSymbolic

> GPU-first evolutionary symbolic regression research.

WarpSymbolic busca fórmulas matemáticas mediante programación genética,
evaluación RPN y evolución masiva en PyTorch/CUDA. El objetivo del proyecto es
investigar un motor evolutivo reproducible y de alto rendimiento; el proyecto
no declara SOTA hasta superar protocolos externos comparables.

## Arquitectura

```text
src/warpsymbolic/
├── api/             API pública y estimador sklearn
├── symbolic/        gramática y representación de fórmulas
├── gpu/             motor evolutivo principal y kernels CUDA
└── cli/             consola GPU, SRBench y benchmarks

src/AlphaSymbolic/
├── app.py           aplicación Gradio legacy/neural
├── ui/              interfaz y entrenamiento experimental
├── experimental/    redes neuronales, MCTS y búsqueda híbrida
├── data/            datos auxiliares
└── scripts/         compatibilidad y runners antiguos

tests/               unit, gpu, integration y e2e
research/            experimentos, notebooks y modelos locales
legacy/cpp_engine/   motor C++/CUDA anterior, separado del producto principal
```

El paquete `warpsymbolic` contiene el camino de producción GPU, su API, la
representación simbólica y sus comandos operativos. La UI, las redes
neuronales y los experimentos legacy viven en `AlphaSymbolic`.

Los artefactos locales no forman parte del árbol del código: compilaciones,
cachés, resultados, logs y temporales se agrupan en `.local/`, una carpeta
ignorada por Git. Los resultados históricos reproducibles permanecen en
`benchmarks/`.

El flujo de producción es:

```text
WarpSymbolicRegressor → TensorGeneticEngine → CUDA → fórmula + validación
```

## API pública

```python
from warpsymbolic import WarpSymbolicRegressor
from warpsymbolic.gpu import TensorGeneticEngine

model = WarpSymbolicRegressor(max_time=60, search_mode="legacy")
model.fit(X, y)
print(model.sympy_formula_)
```

La GPU es obligatoria para el ajuste de producción. El modo CPU solo se
activa explícitamente con `device="cpu"` para tests, parsing, validación o
debugging; no existe fallback silencioso.

## Instalación y comandos

```powershell
python -m pip install -e ".[benchmark]"
warp-symbolic datos.csv --target y
warp-symbolic-srbench --profile quick --resume
```

También se pueden ejecutar los módulos directamente:

```powershell
python -m AlphaSymbolic.app
python -m warpsymbolic.cli.benchmark_scientific --suite development --mode both
python -m warpsymbolic.cli.profile_gpu_engine
```

La extensión CUDA nativa está en `src/warpsymbolic/gpu/cuda/`. Para preparar
un build local fuera del código fuente:

```powershell
Push-Location src/warpsymbolic/gpu/cuda
python setup.py build_ext --build-temp ../../../../.local/build/cuda/temp --build-lib ../../../../.local/build/python
Pop-Location
```

## Compatibilidad

`AlphaSymbolic` permanece como namespace legacy para la aplicación, UI y
experimentos neuronales, además de los aliases de transición de la API y el
comando SRBench antiguo. Produce una advertencia de deprecación y no forma
parte del camino GPU de producción; el código nuevo debe usar `warpsymbolic`.

Consulta [`docs/MIGRATION_ALPHA_TO_WARP.md`](docs/MIGRATION_ALPHA_TO_WARP.md)
para migrar integraciones existentes y
[`docs/SRBENCH_REPRODUCIBILITY.md`](docs/SRBENCH_REPRODUCIBILITY.md) antes de
interpretar resultados de benchmarks.

## Investigación y resultados

- `docs/`: arquitectura, auditorías y reproducibilidad.
- `research/`: experimentos que no son parte del camino GPU oficial.
- `benchmarks/`: manifiestos, resultados crudos y reportes.
- `legacy/cpp_engine/`: baseline C++/CUDA independiente.

Los resultados históricos conservan sus nombres y hashes originales para no
romper la procedencia científica.

## Tests

```powershell
python -m pytest tests/unit tests/integration
python -m pytest tests/gpu -m gpu
python -m pytest tests/e2e
```

Los tests que requieren CUDA deben ejecutarse en una máquina con PyTorch CUDA
y la extensión nativa compilada.

## Licencia

Apache License 2.0. Consulta [`LICENSE`](LICENSE).
