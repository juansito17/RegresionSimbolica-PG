# AlphaSymbolic 🧠

> **Deep Reinforcement Learning para Regresión Simbólica**
> *Inspirado en AlphaZero y AlphaTensor*

AlphaSymbolic es una inteligencia artificial autónoma capaz de **descubrir fórmulas matemáticas** a partir de datos. Utiliza un enfoque **Híbrido Neuro-Evolutivo** que combina la intuición de una Red Neuronal (Transformer) con la precisión de un Motor Genético (GP) en C++.

## 🚀 Características Principales

### 🧠 Arquitectura Híbrida (Neuro-Symbolic)
- **Red Neuronal Transformer**: Actúa como la "Intuición". Genera hipótesis rápidas (Beam Search) sobre la estructura de la fórmula.
- **Motor Genético (C++)**: Actúa como el "Maestro". Refina las hipótesis de la red, ajusta constantes y resuelve los casos difíciles.
- **Hybrid Feedback Loop**: Un ciclo de mejora continua donde la red aprende de las correcciones del motor genético (Teacher-Student Distillation).

### ⚡ Potencia Ajustable (Nuevo)
- **Modo Lite (Laptop)**: Rápido y ligero (128 dim, 3 capas). Funciona en cualquier CPU/GPU básica. Ideal para desarrollo local.
- **Modo Pro (Colab/Cloud)**: Cerebro gigante (256 dim, 6 capas). Requiere GPU potente (T4/A100). Capaz de entender conceptos más profundos.

### 🎓 Aprendizaje y Curriculum
- **Hard Mining**: El sistema identifica activamente los problemas donde la red falla y desafía al Motor GP a resolverlos.
- **Teacher-Student**: La red neuronal (Alumno) se entrena replicando las soluciones exitosas del GP (Maestro).
- **Benchmarks Científicos**: Validado con el dataset de Feynman (Física) para redescubrir leyes fundamentales.

### ☁️ Listo para la Nube
- **Google Colab**: Incluye un script generador (`Code/notebooks/GoogleColab_Project.ipynb`) para correr todo el proyecto gratis en la nube de Google con un solo click.

---

## 🛠️ Instalación

1.  **Clonar repositorio**:
    ```bash
    git clone https://github.com/juansito17/RegresionSimbolica-PG.git
    cd AlphaSymbolic
    ```

2.  **Instalar dependencias**:
    ```bash
    # PyTorch con soporte CUDA (recomendado)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Librerías auxiliares
    pip install gradio scipy numpy matplotlib sympy
    ```

### ⚡ Aceleración por GPU (CUDA)
El motor genético utiliza una extensión en C++/CUDA para máxima velocidad. Al clonar el repositorio, debes compilarla manualmente:

1.  **Requisitos**: NVIDIA CUDA Toolkit y Visual Studio 2022 (con soporte para C++).
2.  **Compilar**:
    ```bash
    cd AlphaSymbolic/core/gpu/cuda
    ./build_extension.bat
    ```
> [!NOTE]
> Si el script falla, asegúrate de que la ruta a `vcvars64.bat` en el archivo `.bat` coincida con tu instalación de Visual Studio.

3.  **Ejecutar**:
    ```bash
    python app.py
    ```
    Visita `http://127.0.0.1:7860` en tu navegador.

---

## 🧪 Cómo Usar

### 1. Selecciona tu Cerebro
En la barra superior, elige entre **Lite** (rápido) y **Pro** (potente). Si cambias, la IA reiniciará sus pesos.

### 2. Entrenamiento (El Gimnasio)
Ve a la pestaña `Entrenamiento` y activa el **Self-Play Loop**.
- Verás: "Buscando..." -> "Entrenando..."
- La IA generará datos, intentará resolverlos, y aprenderá de sus errores.
- **Tip**: Déjalo correr 1000 iteraciones para ver resultados mágicos.

### 3. Búsqueda Híbrida
Ve a `Buscar Fórmula`.
- Escribe tus datos X e Y (ej: `1,2,3` y `2,4,6`).
- Dale a **Buscar Fórmula**.
- El sistema lanzará un **Neural Beam Search** para generar candidatos y el **Motor GP** los refinará en milisegundos.

### 4. Benchmark (El Test de CI)
Ve a `Benchmark (IQ Test)`.
- Dale a **Iniciar Examen**.
- La IA se enfrentará a 10 problemas clásicos de regresión simbólica sin haberlos visto antes.

### 5. Herramientas Avanzadas (Scripts)
- **Benchmark Físico**: Ejecuta `python run_benchmark_feynman.py` para probar el modelo con leyes físicas reales (Gravedad, Relatividad, etc.).
- **Rescate de Datos**: Si cierras la app, usa `python rescue_data.py` para extraer las fórmulas aprendidas de los logs de la consola y guardarlas en CSV.

---

## 📂 Despliegue en Google Colab

Si no tienes GPU potente, usa Google Colab:
1. Sube el archivo `Code/notebooks/GoogleColab_Project.ipynb` a tu Google Drive.
2. Ábrelo y cambia el entorno a **T4 GPU**.
3. Ejecuta todo.
4. Obtendrás un link público (Gradio) para usar tu IA desde cualquier lugar.

---

## 🧠 Estructura del Proyecto

- `core/`: Modelo Transformer (PyTorch) y Gramática Matemática.
- `search/`: Algoritmos de Búsqueda (MCTS Paralelo, Beam Search).
- `ui/`: Interfaz gráfica moderna con Gradio.
- `data/`: Generadores de ecuaciones y Benchmarks.
- `utils/`: Optimizador de constantes (BFGS) y runners.

---
*Creado con ❤️ e Inteligencia Artificial.*
