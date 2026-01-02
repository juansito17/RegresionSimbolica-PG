# 🧠 AlphaSymbolic

> **Regresión Simbólica con Deep Reinforcement Learning (AlphaZero-Style)**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](../LICENSE)

Sistema de descubrimiento automático de fórmulas matemáticas usando redes neuronales Transformer y Monte Carlo Tree Search, inspirado en AlphaTensor de DeepMind.

---

## 🚀 Inicio Rápido

```bash
# 1. Instalar dependencias
cd AlphaSymbolic
pip install -r requirements.txt

# 2. Entrenar el modelo (opcional, ya hay uno pre-entrenado)
python train_enhanced.py --epochs 500

# 3. Resolver un problema
python search_pro.py
```

---

## 📖 Guía de Uso

### 🔍 Resolver un Problema Simple

```python
from search_pro import solve_pro
import numpy as np

# Tus datos
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([5, 7, 9, 11, 13, 15, 17, 19, 21, 23])  # y = 2x + 3

# Buscar fórmula
result, pareto = solve_pro(x, y)
print(result['final_formula'])  # Debería encontrar "2*x + 3"
```

### 🎓 Entrenar el Modelo

```bash
# Entrenamiento básico (rápido, ~5 min)
python train.py

# Entrenamiento avanzado con curriculum learning (~30 min)
python train_enhanced.py --epochs 1000 --batch 64

# Self-Play AlphaZero (mejora continua, ~horas)
python self_play.py --iterations 100 --problems 20
```

### 📊 Modos de Búsqueda

| Modo | Comando | Velocidad | Precisión |
|------|---------|-----------|-----------|
| **Beam Search** | `--method beam` | ⚡ Rápido | ⭐⭐⭐ |
| **MCTS** | `--method mcts` | 🐢 Lento | ⭐⭐⭐⭐ |

```bash
# Beam Search (recomendado)
python search_pro.py --method beam --beam-width 20

# MCTS (más exhaustivo)
python search_pro.py --method mcts --mcts-sims 500
```

---

## 🔧 Operadores Soportados

| Tipo | Operadores |
|------|------------|
| **Aritméticos** | `+`, `-`, `*`, `/`, `pow`, `mod` |
| **Trigonométricos** | `sin`, `cos`, `tan` |
| **Exponenciales** | `exp`, `log`, `sqrt` |
| **Especiales** | `abs`, `floor`, `ceil`, `gamma` |
| **Constantes** | `pi`, `e`, `C` (optimizable) |

---

## 📂 Estructura del Proyecto

```
AlphaSymbolic/
├── 🧠 Core (Núcleo)
│   ├── grammar.py          # Gramática y árboles de expresión
│   ├── model.py            # Red neuronal Transformer
│   ├── mcts.py             # Monte Carlo Tree Search
│   └── beam_search.py      # Búsqueda por haz
│
├── 📈 Optimización
│   ├── optimize_constants.py  # Optimización numérica (scipy)
│   ├── simplify.py            # Simplificación algebraica (SymPy)
│   └── pareto.py              # Frente de Pareto
│
├── 🎓 Entrenamiento
│   ├── train.py              # Entrenamiento básico
│   ├── train_enhanced.py     # Curriculum + Value Loss
│   ├── self_play.py          # AlphaZero Loop
│   └── synthetic_data.py     # Generador de datos
│
├── 🔧 Avanzado
│   ├── multivar.py           # Multi-variable f(x1, x2, ...)
│   ├── gpu_eval.py           # Evaluación batch GPU
│   ├── cpp_binding.py        # Integración C++
│   ├── pattern_memory.py     # Memoria de patrones
│   └── detect_pattern.py     # Detección de patrones
│
├── 🚀 Ejecución
│   ├── search.py             # Búsqueda simple
│   └── search_pro.py         # Pipeline completo
│
└── 📋 Configuración
    └── requirements.txt
```

---

## 🔬 Pipeline de Búsqueda

```
┌─────────────────────────────────────────────────────────────┐
│                    ALPHASIMBOLIC PIPELINE                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. DETECCIÓN DE PATRÓN                                     │
│     └─ Analiza Y: ¿lineal? ¿cuadrático? ¿periódico?        │
│                                                             │
│  2. BÚSQUEDA NEURONAL (Beam/MCTS)                          │
│     └─ Transformer genera estructuras candidatas           │
│                                                             │
│  3. OPTIMIZACIÓN DE CONSTANTES                              │
│     └─ scipy minimiza RMSE para cada C                     │
│                                                             │
│  4. FRENTE DE PARETO                                        │
│     └─ Selecciona mejores (precisión vs simplicidad)       │
│                                                             │
│  5. SIMPLIFICACIÓN                                          │
│     └─ SymPy limpia fórmula final                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Ejemplos

### Ejemplo 1: Encontrar una fórmula lineal

```python
import numpy as np
from search_pro import solve_pro

x = np.linspace(0, 10, 20)
y = 3 * x - 5  # Fórmula objetivo

result, _ = solve_pro(x, y)
# Output: "3*x - 5" o equivalente
```

### Ejemplo 2: Fórmula cuadrática

```python
x = np.linspace(-5, 5, 30)
y = x**2 + 2*x + 1  # (x+1)^2

result, _ = solve_pro(x, y, beam_width=20)
# Output: "(x + 1)^2" o "x^2 + 2*x + 1"
```

### Ejemplo 3: Multi-variable

```python
from multivar import MultiVarExpressionTree, MultiVarDataGenerator

# Crear generador para 2 variables
gen = MultiVarDataGenerator(num_variables=2)

# Generar datos
x_dict = {'x0': np.array([1,2,3]), 'x1': np.array([4,5,6])}
# Buscar f(x0, x1) tal que y = x0 + x1*2
```

---

## ⚙️ Configuración Avanzada

### Aumentar Precisión
```bash
python search_pro.py --beam-width 30 --method beam
```

### Entrenar Más Tiempo
```bash
python train_enhanced.py --epochs 5000
python self_play.py --iterations 500
```

### Usar GPU
El sistema detecta automáticamente CUDA:
```python
import torch
print(torch.cuda.is_available())  # True si GPU disponible
```

---

## 📈 Rendimiento

| Configuración | Tiempo | RMSE Típico |
|--------------|--------|-------------|
| Beam (width=10) | ~2s | ~1e-2 |
| Beam (width=30) | ~10s | ~1e-4 |
| MCTS (500 sims) | ~30s | ~1e-5 |
| MCTS + Self-Play | ~horas | ~1e-8 |

---

## 🤝 Comparación con el Algoritmo Genético Original

| Característica | GA Original | AlphaSymbolic |
|---------------|-------------|---------------|
| Método | Mutación/Cruce | Deep RL + MCTS |
| Aprendizaje | No (heurístico) | Sí (red neuronal) |
| Velocidad | Rápido | Más lento pero smarter |
| Escalabilidad | Limitada | GPU paralelo |
| Multi-variable | No | ✅ Sí |

---

## 📄 Licencia

Apache 2.0 - Ver [LICENSE](../LICENSE)

---

<div align="center">

**Desarrollado con 🧠 por AlphaSymbolic Team**

</div>
