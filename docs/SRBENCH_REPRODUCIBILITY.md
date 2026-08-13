# Reproducibilidad de la integración SRBench

Esta guía separa tres objetivos distintos: comprobar que la integración
funciona, ejecutar una evaluación local amplia y reproducir el protocolo
oficial. No son resultados intercambiables.

La fuente normativa es el repositorio oficial
[`cavalab/srbench`](https://github.com/cavalab/srbench). Esta integración fija
la revisión SRBench 2025
[`dc3f6daa93bf10955df8775256a6f8644f38fd93`](https://github.com/cavalab/srbench/commit/dc3f6daa93bf10955df8775256a6f8644f38fd93).
Los datos PMLB se fijan en
[`7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68`](https://github.com/EpistasisLab/pmlb/commit/7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68).
El entrypoint local es:

```text
python -m AlphaSymbolic.scripts.benchmark_srbench
```

La salida predeterminada es `benchmarks/srbench_2025.jsonl` y el caché
predeterminado es `cache/srbench_2025`; esta guía pasa ambas rutas
explícitamente para evitar ambigüedad.

Antes de una corrida, confirme la interfaz realmente instalada:

```text
python -m AlphaSymbolic.scripts.benchmark_srbench --help
```

## Perfiles

| Perfil | Propósito | Uso legítimo |
|---|---|---|
| `quick` | 2 datasets representativos (`579_fri_c0_250_5` y `first_principles_hubble`), 1 semilla y 60 s por tarea: 2 tareas. | Desarrollo, CI y smoke test. No permite comparar algoritmos ni afirmar SOTA. |
| `full` | 24 datasets (12 black-box y 12 first-principles), primeras 3 semillas oficiales y 600 s por tarea: 72 tareas. | Ablaciones y comparaciones internas cuando todas las variantes usan exactamente el mismo hardware y presupuesto. No equivale por sí solo a una corrida oficial. |
| `official` | Los mismos 24 datasets, 30 semillas oficiales y 3600 s de techo por tarea: 720 tareas. | Cobertura local 24×30 con la escala exterior de la revisión fijada. No reproduce el tuning upstream ni constituye un resultado oficial de SRBench. |

`smoke` es únicamente un alias de `quick` y se registra canónicamente como
`quick`. Al citar resultados se debe usar el nombre guardado en el JSONL.
Los valores efectivos escritos por el programa en cada registro —no los
defaults recordados por el operador— son la configuración autoritativa.

El runner aplica una política de presupuesto fija y registrada para tablas
`firstprinciples`: con menos de 32 filas de train limita el GP a 10 000
individuos, 30 generaciones y 60 s; con menos de 128, a 25 000 individuos,
60 generaciones y 180 s. El portfolio polinómico/fallback sigue usando todo el
train. `runner_metadata.budget_policy` y `params` guardan el caso efectivo.
Esta adaptación evita gastar minutos optimizando GP sobre 3–20 observaciones y
forma parte del algoritmo, no es un override escogido después de ver el test.

Los tres perfiles aplican un split reproducible 75/25 de scikit-learn, limitan
solo el train a un máximo de 40 000 filas, ajustan `StandardScaler` de `X` y de
`y` exclusivamente con train e invierten la transformación de `y` antes de
calcular métricas sobre el test original. El harness no hace tuning: la
configuración queda fijada por el runner. Por ello, los tres perfiles registran
siempre `official_protocol=false` y la lista `override_reasons` contiene como
mínimo `fixed_runner_skips_upstream_hyperparameter_tuning`. El perfil
`official` solo fija cobertura 24×30 y un techo de 3600 s por tarea; no cambia
esa marca. Cualquier override adicional también se registra y la corrida debe
presentarse como una evaluación local derivada.

El manifiesto versionado es
`AlphaSymbolic/scripts/srbench_2025_manifest.json`. El caché predeterminado se
divide en `cache/srbench_2025/datasets` y
`cache/srbench_2025/official_results`.

## Requisitos

- Un checkout limpio de AlphaSymbolic y Git.
- Python compatible con el proyecto; registre la versión exacta.
- PyTorch con CUDA y una GPU NVIDIA. Una ejecución con fallback CPU constituye
  otra configuración experimental.
- Driver NVIDIA, CUDA Toolkit y compilador C++ compatibles si se recompila la
  extensión nativa.
- Espacio para el caché de datasets y para resultados JSONL incrementales.
- Git LFS si se inspeccionan o descargan manualmente los datasets upstream.

SRBench documenta instalación nativa en Ubuntu y CentOS. En Windows, use WSL2
con Ubuntu o un contenedor Linux para el perfil `official`. Una rueda o `.pyd`
compilada para Windows no puede reutilizarse dentro de WSL/Linux.

## Adaptador SRBench

La integración versionada se divide en:

```text
AlphaSymbolic/sklearn_estimator.py
AlphaSymbolic/sklearn.py
integrations/srbench/experiment/methods/alphasymbolic/regressor.py
integrations/srbench/algorithms/alphasymbolic/metadata.yml
integrations/srbench/algorithms/alphasymbolic/requirements.txt
integrations/srbench/algorithms/alphasymbolic/install.sh
```

El estimator mantiene `__init__` sin efectos laterales, crea el motor dentro de
`fit`, expone `random_state` y `max_time`, evalúa fórmulas sin `eval` de Python
y exporta una expresión SymPy. El shim de SRBench define el estimator,
`model(...)`, complejidad, parámetros de prueba y grilla de tuning. Estos
archivos son parte del tratamiento: publíquelos con el commit usado.

`integrations/srbench/algorithms/alphasymbolic/install.sh` acepta
`ALPHASYMBOLIC_REPO` y `ALPHASYMBOLIC_REF`. En una corrida reproducible,
`ALPHASYMBOLIC_REF` debe ser un commit completo accesible, nunca el valor
flotante `main`.

## Preparación de AlphaSymbolic

### Windows PowerShell

Desde la raíz del repositorio:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[benchmark]"

Push-Location AlphaSymbolic\core\gpu\cuda
python setup.py build_ext --inplace
Pop-Location

python -c "import torch; print('torch=', torch.__version__, 'cuda_runtime=', torch.version.cuda, 'cuda=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python -c "from AlphaSymbolic.core.gpu.cuda_loader import load_rpn_cuda_native; m=load_rpn_cuda_native(); print('native_cuda=', m.__file__)"
python -m AlphaSymbolic.scripts.benchmark_srbench --help
```

Instale la rueda de PyTorch adecuada para su driver desde las instrucciones
oficiales de PyTorch si `torch.cuda.is_available()` devuelve `False`.

### Linux, WSL2 o contenedor

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[benchmark]"

pushd AlphaSymbolic/core/gpu/cuda
python setup.py build_ext --inplace
popd

python -c "import torch; print('torch=', torch.__version__, 'cuda_runtime=', torch.version.cuda, 'cuda=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python -c "from AlphaSymbolic.core.gpu.cuda_loader import load_rpn_cuda_native; m=load_rpn_cuda_native(); print('native_cuda=', m.__file__)"
python -m AlphaSymbolic.scripts.benchmark_srbench --help
```

No continúe con una medición oficial si la extensión esperada no carga o si el
motor selecciona un dispositivo distinto del documentado.

## Ejecución

Use una carpeta nueva por revisión, perfil y máquina. No mezcle registros de
commits, GPUs o perfiles diferentes en el mismo JSONL.

### Windows PowerShell

```powershell
$Commit = git rev-parse --short=12 HEAD
$RunRoot = "results\srbench\$Commit"
$Cache = "cache\srbench_2025"
New-Item -ItemType Directory -Force $RunRoot | Out-Null

python -m AlphaSymbolic.scripts.benchmark_srbench --profile quick    --cache-dir $Cache --output "$RunRoot\quick.jsonl"    --resume
python -m AlphaSymbolic.scripts.benchmark_srbench --profile full     --cache-dir $Cache --output "$RunRoot\full.jsonl"     --resume
python -m AlphaSymbolic.scripts.benchmark_srbench --profile official --cache-dir $Cache --output "$RunRoot\official.jsonl" --resume
```

### Linux, WSL2 o contenedor

```bash
commit="$(git rev-parse --short=12 HEAD)"
run_root="results/srbench/$commit"
cache="cache/srbench_2025"
mkdir -p "$run_root"

python -m AlphaSymbolic.scripts.benchmark_srbench --profile quick    --cache-dir "$cache" --output "$run_root/quick.jsonl"    --resume
python -m AlphaSymbolic.scripts.benchmark_srbench --profile full     --cache-dir "$cache" --output "$run_root/full.jsonl"     --resume
python -m AlphaSymbolic.scripts.benchmark_srbench --profile official --cache-dir "$cache" --output "$run_root/official.jsonl" --resume
```

La salida es incremental: una interrupción no debe invalidar las líneas JSON
ya cerradas. Para reanudar, vuelva a ejecutar exactamente el mismo perfil,
revisión y ruta de salida. No concatene archivos manualmente y no cambie el
presupuesto entre reanudaciones. La reanudación está activa por defecto; use
`--resume` de forma explícita en scripts de reproducción y `--no-resume` solo
para una salida nueva.

`quick` debe terminar antes de invertir recursos en `full` u `official`.
Ejecutar los tres perfiles no es obligatorio: para una medición local 24×30 es
preferible una corrida `official` limpia que reutilizar resultados de perfiles
anteriores.

Sin paralelismo, el límite nominal de `official` suma 720 horas GPU (30 días),
más preparación y postprocesamiento. Verifique el plan antes de iniciarlo; no
reduzca silenciosamente seeds, datasets o timeout para hacerlo terminar antes.

La interfaz completa también admite:

```text
--cache-dir PATH
--runner module:function
--algorithm NAME
--track {all,blackbox,firstprinciples}
--datasets DATASET_1,DATASET_2
--seeds SEED_1,SEED_2
--offline
--prepare-only
--rank
--rank-only
--ranking-output PATH
```

No use `--datasets` ni `--seeds` para recortar una corrida que se vaya a
etiquetar `official`. Son útiles para depurar y deben quedar registrados como
una configuración derivada.

## Registro obligatorio de hardware y software

Como mínimo publique:

- fabricante/modelo de CPU, núcleos asignados y RAM disponible;
- modelo exacto de GPU, VRAM, límite de potencia y modo de persistencia;
- driver NVIDIA, versión CUDA reportada por el driver, CUDA Toolkit/NVCC;
- sistema operativo/kernel o versión de Windows/WSL;
- Python, PyTorch, `torch.version.cuda` y dependencias congeladas;
- commit AlphaSymbolic, estado limpio/sucio y hash del diff si lo hubiera;
- commit SRBench, perfil, semillas, repeticiones, timeout, memoria y nivel de
  paralelismo efectivos;
- política de calentamiento, procesos concurrentes y, si se publica energía,
  instrumento, frecuencia de muestreo y ventana de integración.

### Captura en Windows PowerShell

```powershell
$RunRoot = "results\srbench\$(git rev-parse --short=12 HEAD)"
$DiffText = git diff --binary | Out-String
$Sha256 = [Security.Cryptography.SHA256]::Create()
$DiffHash = [BitConverter]::ToString(
  $Sha256.ComputeHash([Text.Encoding]::UTF8.GetBytes($DiffText))
).Replace("-", "").ToLowerInvariant()
$Sha256.Dispose()
@(
  "timestamp_utc=$([DateTime]::UtcNow.ToString('o'))"
  "alphasymbolic_commit=$(git rev-parse HEAD)"
  "alphasymbolic_tree=$(if (git status --porcelain) { 'dirty' } else { 'clean' })"
  "diff_sha256=$DiffHash"
  "python=$(python --version 2>&1)"
  "os=$((Get-CimInstance Win32_OperatingSystem).Caption)"
  "cpu=$((Get-CimInstance Win32_Processor).Name -join '; ')"
  "ram_bytes=$((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory)"
) | Set-Content "$RunRoot\environment.txt"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,power.limit,persistence_mode --format=csv,noheader | Add-Content "$RunRoot\environment.txt"
nvcc --version | Add-Content "$RunRoot\environment.txt"
python -c "import torch; print('torch='+torch.__version__); print('torch_cuda='+str(torch.version.cuda)); print('cudnn='+str(torch.backends.cudnn.version()))" | Add-Content "$RunRoot\environment.txt"
python -m pip freeze | Set-Content "$RunRoot\pip-freeze.txt"
git diff --binary | Set-Content "$RunRoot\working-tree.patch"
```

Para una publicación, prefiera un árbol limpio. El hash de un diff de
PowerShell puede depender de la normalización de saltos de línea; archive
también `working-tree.patch` o, mejor, haga un commit identificable.

### Captura en Linux/WSL

```bash
run_root="results/srbench/$(git rev-parse --short=12 HEAD)"
{
  printf 'timestamp_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'alphasymbolic_commit=%s\n' "$(git rev-parse HEAD)"
  if test -z "$(git status --porcelain)"; then
    printf 'alphasymbolic_tree=clean\n'
  else
    printf 'alphasymbolic_tree=dirty\n'
  fi
  printf 'diff_sha256=%s\n' "$(git diff --binary | sha256sum | cut -d' ' -f1)"
  uname -a
  lscpu
  free -b
  python --version
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,power.limit,persistence_mode --format=csv,noheader
  nvcc --version
  python -c "import torch; print('torch='+torch.__version__); print('torch_cuda='+str(torch.version.cuda)); print('cudnn='+str(torch.backends.cudnn.version()))"
} > "$run_root/environment.txt"
python -m pip freeze > "$run_root/pip-freeze.txt"
git diff --binary > "$run_root/working-tree.patch"
```

No ejecute navegadores, entrenamiento, minería, benchmarks gráficos ni otra
carga GPU simultánea. Registre cualquier control de frecuencia o potencia; no
compare una GPU desbloqueada con otra limitada sin declararlo.

## Verificación de datos y resultados

Cada dataset cacheado debe asociarse con:

1. URL o identificador upstream;
2. commit SRBench fijado;
3. tamaño en bytes;
4. SHA-256 del contenido consumido por el benchmark.

Ante un hash distinto, detenga la corrida. No “actualice” el hash esperado sin
explicar y revisar el cambio de datos. Un nombre de dataset igual no garantiza
contenido igual.

Para separar adquisición y cómputo, prepare el caché mientras hay red y ejecute
después con `--offline`:

```text
python -m AlphaSymbolic.scripts.benchmark_srbench --profile official --cache-dir cache/srbench_2025 --output results/srbench/<commit>/prepare.jsonl --prepare-only
python -m AlphaSymbolic.scripts.benchmark_srbench --profile official --cache-dir cache/srbench_2025 --output results/srbench/<commit>/official.jsonl --offline --resume
```

Cada línea de resultado usa `schema_version` y
`record_type="srbench_run"`. Los campos de procedencia y protocolo incluyen:

```text
run_id, timestamp_utc, profile, official_protocol, override_reasons, algorithm
dataset, dataset_group, random_state, trial
srbench_commit, pmlb_commit, dataset_sha256, protocol_sha256
alphasymbolic_commit, alphasymbolic_diff_sha256, alphasymbolic_source_sha256
split, scaling, fit_time_limit_sec, params
```

`alphasymbolic_source_sha256` cubre tanto archivos seguidos por Git como
archivos fuente nuevos todavía no añadidos; a diferencia del hash de
`git diff`, no deja fuera el adaptador o runner cuando aún son `untracked`.

Los campos de resultado incluyen `status`, `error`, `training_time_sec`,
`metrics.train`, `metrics.test`, `model_size` y `symbolic_model`; cada bloque de
métricas contiene `r2`, `mse` y `mae`. Conserve las filas fallidas: `status` y
`error` son parte del resultado.

Compruebe que el JSONL solo contiene líneas completas y JSON válido:

```powershell
python -c "import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); rows=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]; assert len(rows)==720, f'se esperaban 720 filas, hay {len(rows)}'; assert all(r.get('record_type')=='srbench_run' for r in rows); assert all(r.get('official_protocol') is False for r in rows), 'el harness fijo no debe declararse protocolo oficial'; assert all('fixed_runner_skips_upstream_hyperparameter_tuning' in r.get('override_reasons',[]) for r in rows), 'falta la razón de desviación obligatoria'; keys=[(r.get('algorithm'),r.get('dataset_group'),r.get('dataset'),r.get('random_state'),r.get('trial')) for r in rows]; assert len(keys)==len(set(keys)), 'tareas duplicadas'; print('rows=',len(rows)); print('profiles=',sorted({str(r.get('profile')) for r in rows})); print('srbench=',sorted({str(r.get('srbench_commit')) for r in rows})); print('protocols=',sorted({str(r.get('protocol_sha256')) for r in rows})); print('errors=',sum(bool(r.get('error')) for r in rows))" "results\srbench\<commit>\official.jsonl"
```

```bash
python -c "import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); rows=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]; assert len(rows)==720, f'se esperaban 720 filas, hay {len(rows)}'; assert all(r.get('record_type')=='srbench_run' for r in rows); assert all(r.get('official_protocol') is False for r in rows), 'el harness fijo no debe declararse protocolo oficial'; assert all('fixed_runner_skips_upstream_hyperparameter_tuning' in r.get('override_reasons',[]) for r in rows), 'falta la razón de desviación obligatoria'; keys=[(r.get('algorithm'),r.get('dataset_group'),r.get('dataset'),r.get('random_state'),r.get('trial')) for r in rows]; assert len(keys)==len(set(keys)), 'tareas duplicadas'; print('rows=',len(rows)); print('profiles=',sorted({str(r.get('profile')) for r in rows})); print('srbench=',sorted({str(r.get('srbench_commit')) for r in rows})); print('protocols=',sorted({str(r.get('protocol_sha256')) for r in rows})); print('errors=',sum(bool(r.get('error')) for r in rows))" "results/srbench/<commit>/official.jsonl"
```

Revise además que todas las filas declaren la revisión SRBench y el perfil
esperados; que `dataset_sha256` corresponda al manifiesto del caché; y que el
número de filas coincida con el plan efectivo registrado por el perfil. El
commit AlphaSymbolic se vincula mediante `environment.txt` y la ruta de corrida.
Los fallos y timeouts forman parte del denominador: nunca se deben eliminar al
agregar métricas.

Genere un hash del artefacto final después de cerrar la corrida:

```powershell
Get-FileHash -Algorithm SHA256 "results\srbench\<commit>\official.jsonl"
Get-FileHash -Algorithm SHA256 "results\srbench\<commit>\environment.txt"
Get-FileHash -Algorithm SHA256 "results\srbench\<commit>\pip-freeze.txt"
```

```bash
sha256sum \
  "results/srbench/<commit>/official.jsonl" \
  "results/srbench/<commit>/environment.txt" \
  "results/srbench/<commit>/pip-freeze.txt"
```

Publique esos hashes junto con los archivos, no solo una captura de pantalla o
una tabla agregada. Para comprobar una reproducción, compare primero commits,
hashes de datasets, configuración y número de tareas; compare después métricas
por dataset y distribuciones entre semillas. No exija igualdad bit a bit de
todos los valores flotantes entre drivers o arquitecturas GPU diferentes.

Genere el ranking desde el JSONL cerrado, sin volver a entrenar:

```text
python -m AlphaSymbolic.scripts.benchmark_srbench --profile official --rank-only --output results/srbench/<commit>/official.jsonl --ranking-output results/srbench/<commit>/official-ranking.json
```

No combine en un ranking filas con `protocol_sha256`, perfiles o recursos
distintos. Archive y calcule también el SHA-256 de `official-ranking.json`.
Para desglosar las pistas sin reentrenar, repita el mismo comando con
`--track blackbox` y `--track firstprinciples`; el ranking filtra entonces el
JSONL local al mismo conjunto de datasets que el Feather de referencia.

## Qué significa `official`

El perfil local `official` fija el track AlphaSymbolic de 24 datasets descrito
arriba y su cobertura 24×30 con un techo de 3600 s por tarea. El nombre del
perfil no convierte el resultado en oficial: el harness usa un runner de
configuración fija, no reproduce dentro de sí el grid search completo del
orquestador upstream y, por diseño, deja `official_protocol=false` con la razón
`fixed_runner_skips_upstream_hyperparameter_tuning`. Para una afirmación de
leaderboard o publicación:

1. el adaptador debe satisfacer la
   [guía de contribución de SRBench](https://cavalab.org/srbench/contributing/):
   API compatible con scikit-learn, `random_state`, límite `max_time` y modelo
   final como cadena compatible con SymPy;
2. datasets, splits, tuning, límites de tiempo/memoria y número de trials deben
   coincidir con la pista upstream que se declare;
3. AlphaSymbolic y sus comparadores deben ejecutarse bajo el mismo orquestador
   y política de recursos;
4. los resultados deben pasar la evaluación y postprocesamiento upstream;
5. una aceptación o corrida upstream debe enlazarse de forma separada.

La receta black-box de la revisión fijada usa 30 trials, límite de fit de
3600 s, máximo de 40 000 muestras, 10 000 MB por job, escalado y tuning mediante
`optimize_model`; la pista first-principles usa el mismo presupuesto general.
La [guía oficial](https://cavalab.org/srbench/user-guide/) advierte que una
corrida completa puede lanzar decenas de miles de experimentos. Sus resultados
históricos `v2.0` y los de la revisión 2025 tampoco deben combinarse.

## Límites honestos

- `quick` solo valida la tubería.
- `full` no demuestra superioridad frente a métodos ejecutados con otros
  presupuestos, tuning, hardware o versiones.
- `official` local es el track fijado de 24 datasets, no la totalidad de
  SRBench, y sigue siendo una reproducción independiente hasta ser validada por
  el flujo upstream.
- SRBench mide varias dimensiones. No hay una única noción de “SOTA”: reporte
  error predictivo, recuperación simbólica, complejidad, tiempo, fallos y
  recursos, preferiblemente como frentes de Pareto.
- El timeout solicitado y el tiempo real observado no son equivalentes. Reporte
  ambos e incluya inicialización/compilación según la convención del protocolo.
- Las semillas no eliminan el no determinismo de kernels CUDA, reducciones
  paralelas, versiones de librerías ni `--use_fast_math`.
- No seleccione la mejor semilla ni ajuste hiperparámetros sobre el test.
- Una expresión numéricamente precisa no es necesariamente simbólicamente
  equivalente; use la evaluación SymPy upstream y conserve los fallos de parseo.
- La métrica interna de candidatos-generación/s de AlphaSymbolic no es
  directamente comparable con GPops/s, evaluaciones punto a punto o tiempo de
  otros motores.
- N-Reinas A000170 del benchmark científico local es regresión de una secuencia;
  no es un solver de tableros ni una prueba de conteo, y no debe presentarse
  como resultado SRBench.
- Cambiar la gramática, operadores, precisión, transformación logarítmica,
  población, islas o simplificador crea otro tratamiento experimental y exige
  otro identificador de configuración.

La motivación para estandarizar tuning, restricciones de ejecución y recursos
está resumida en el trabajo
[“Call for Action: Towards the Next Generation of Symbolic Regression Benchmark”](https://arxiv.org/abs/2505.03977).
El informe local de rendimiento y corrección del motor está en
[`GPU_ENGINE_AUDIT_2026-07-26.md`](GPU_ENGINE_AUDIT_2026-07-26.md).

## Lista de comprobación para publicar

- [ ] Checkout AlphaSymbolic limpio y commit completo registrado.
- [ ] Commit SRBench igual a `dc3f6daa93bf10955df8775256a6f8644f38fd93`.
- [ ] Commit PMLB igual a `7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68`.
- [ ] Perfil efectivo `official`; cobertura 24×30, techo y comandos archivados.
- [ ] `official_protocol=false` en todas las filas y
      `fixed_runner_skips_upstream_hyperparameter_tuning` presente en
      `override_reasons`.
- [ ] Hashes SHA-256 de todos los datasets verificados antes de entrenar.
- [ ] GPU nativa cargada y hardware/software documentados.
- [ ] Sin cargas concurrentes; recursos idénticos entre comparadores.
- [ ] Todas las semillas, trials, fallos y timeouts conservados.
- [ ] JSONL válido, completo, sin claves duplicadas y con hash final.
- [ ] Métricas upstream y expresiones originales archivadas.
- [ ] Resultados agregados acompañados por datos por corrida.
- [ ] Claims limitados al protocolo, datasets, presupuesto y revisión medidos.

## Flujo adaptativo congelado

El candidato universal se congela antes de mirar los 24 datasets:

```powershell
python AlphaSymbolic/scripts/freeze_adaptive_config.py benchmarks/adaptive_config_candidate.json
```

El hash excluye únicamente la semilla de repetición e incluye los parámetros que alteran la búsqueda. El runner no ramifica por nombre ni grupo y usa la misma configuración en todas las filas. `search_mode="legacy"` sigue siendo el valor predeterminado hasta que `check_release_gates.py` confirme todos los gates.

Hay dos ejecuciones deliberadamente distintas:

- `integrations/srbench/run_local_24x30.sh`: reproducción independiente y reanudable; nunca se etiqueta como resultado oficial.
- `integrations/srbench/prepare_upstream.sh` y `run_upstream_24x30.sh`: checkout fijado de cavalab/srbench, contenedor del algoritmo, 30 repeticiones, `eco2ai`, agregación upstream y el verificador `experiment/assess_symbolic_model.py` upstream.

El límite externo upstream continúa en 3600 s para igualar la infraestructura; AlphaSymbolic conserva internamente su presupuesto congelado de 60 s. Una corrida sólo permite promoción si las 720 parejas dataset/semilla tienen un único hash, cobertura completa, energía disponible y primer puesto literal en las seis métricas definidas.
