# AlphaSymbolic 🧠

> **Deep Reinforcement Learning para Regresión Simbólica**
> *Inspirado en AlphaZero y AlphaTensor*

AlphaSymbolic es una inteligencia artificial autónoma capaz de **descubrir fórmulas matemáticas** a partir de datos. No utiliza fuerza bruta; aprende a "jugar" con las matemáticas usando una red neuronal y búsqueda de árbol de Monte Carlo (MCTS).

## 🚀 Características Principales

### 🧠 Arquitectura AlphaZero
- **Red Neuronal Transformer**: Codifica los datos (X, Y) y decodifica la fórmula token a token.
- **Value Head**: Intuye si una fórmula parcial va por buen camino antes de terminarla.
- **MCTS Híbrido**: Combina la "imaginación" de la red neuronal con la precisión de la búsqueda por árbol.

### ⚡ Potencia Ajustable (Nuevo)
- **Modo Lite (Laptop)**: Rápido y ligero (128 dim, 3 capas). Funciona en cualquier CPU/GPU básica. Ideal para desarrollo local.
- **Modo Pro (Colab/Cloud)**: Cerebro gigante (256 dim, 6 capas). Requiere GPU potente (T4/A100). Capaz de entender conceptos más profundos.

### 🎓 Aprendizaje Continuo
- **Self-Play**: La IA se inventa sus propios problemas para practicar, como un estudiante estudiando para un examen.
- **Curriculum Learning**: Empieza con sumas simples y avanza hasta trigonometría y exponentes.
- **Benchmark IQ**: Un examen estandarizado de 10 problemas (Feynman, Nguyen) para medir su coeficiente intelectual matemático.

### ☁️ Listo para la Nube
- **Google Colab**: Incluye un script generador (`AlphaSymbolic_Colab.ipynb`) para correr todo el proyecto gratis en la nube de Google con un solo click.

---

## 🛠️ Instalación

1.  **Clonar repositorio**:
    ```bash
    git clone https://github.com/juansito17/AlphaSymbolic.git
    cd AlphaSymbolic
    ```

2.  **Instalar dependencias**:
    ```bash
    # PyTorch con soporte CUDA (recomendado)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Librerías auxiliares
    pip install gradio scipy numpy matplotlib sympy
    ```

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

### 3. Búsqueda (El Examen)
Ve a `Buscar Fórmula`.
- Escribe tus datos X e Y (ej: `1,2,3` y `2,4,6`).
- Dale a **Buscar Fórmula**.
- El sistema usará **MCTS** para navegar el espacio de posibilidades y encontrar la ecuación exacta.

### 4. Benchmark (El Test de CI)
Ve a `Benchmark (IQ Test)`.
- Dale a **Iniciar Examen**.
- La IA se enfrentará a 10 problemas clásicos de regresión simbólica sin haberlos visto antes.

---

## 📂 Despliegue en Google Colab

Si no tienes GPU potente, usa Google Colab:
1. Sube el archivo `AlphaSymbolic_Colab.ipynb` a tu Google Drive.
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
