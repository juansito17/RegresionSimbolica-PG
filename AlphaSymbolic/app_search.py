"""
Search/Solve functions for AlphaSymbolic Gradio App.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import gradio as gr

from grammar import ExpressionTree
from beam_search import BeamSearch
from simplify import simplify_tree
from pareto import ParetoFront
from detect_pattern import detect_pattern
from app_core import get_model


def parse_data(x_str, y_str):
    """Parse comma-separated input strings."""
    try:
        x = np.array([float(v.strip()) for v in x_str.split(',')], dtype=np.float64)
        y = np.array([float(v.strip()) for v in y_str.split(',')], dtype=np.float64)
        if len(x) != len(y):
            return None, None, "❌ X e Y deben tener igual longitud"
        return x, y, None
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"


def create_fit_plot(x, y, y_pred, formula):
    """Create a plot showing data vs prediction."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    ax.scatter(x, y, color='#00d4ff', s=100, label='Datos Reales', zorder=3, edgecolors='white', linewidth=1)
    
    sort_idx = np.argsort(x)
    ax.plot(x[sort_idx], y_pred[sort_idx], color='#ff6b6b', linewidth=3, label=f'Predicción', zorder=2)
    
    ax.set_xlabel('X', color='white', fontsize=12)
    ax.set_ylabel('Y', color='white', fontsize=12)
    ax.set_title('📈 Ajuste de la Fórmula', color='white', fontsize=14, fontweight='bold')
    ax.legend(facecolor='#16213e', edgecolor='#00d4ff', labelcolor='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='white')
    
    for spine in ax.spines.values():
        spine.set_color('#00d4ff')
    
    plt.tight_layout()
    return fig


def solve_formula(x_str, y_str, beam_width, progress=gr.Progress()):
    """Main solving function."""
    x, y, error = parse_data(x_str, y_str)
    if error:
        return error, None, "", "", ""
    
    MODEL, DEVICE = get_model()
    
    progress(0.1, desc=f"🔍 Analizando patrón... [{DEVICE.type.upper()}]")
    pattern = detect_pattern(x, y)
    
    progress(0.3, desc=f"🧠 Buscando fórmulas... [{DEVICE.type.upper()}]")
    start_time = time.time()
    searcher = BeamSearch(MODEL, DEVICE, beam_width=int(beam_width), max_length=25)
    results = searcher.search(x, y)
    search_time = time.time() - start_time
    
    if not results:
        return "❌ No se encontraron fórmulas válidas", None, "", "", ""
    
    progress(0.7, desc="⚙️ Optimizando constantes...")
    pareto = ParetoFront()
    pareto.add_from_results(results)
    best = pareto.get_best_by_rmse()
    
    if not best:
        return "❌ Error en optimización", None, "", "", ""
    
    progress(0.9, desc="✨ Simplificando...")
    tree = ExpressionTree(best.tokens)
    simplified = simplify_tree(tree)
    y_pred = tree.evaluate(x, constants=best.constants)
    
    fig = create_fit_plot(x, y, y_pred, simplified)
    
    # Format results
    result_html = f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; border: 2px solid #00d4ff;">
        <h2 style="color: #00d4ff; margin: 0; font-size: 24px;">🎯 Fórmula Encontrada</h2>
        <div style="background: #0f0f23; padding: 15px; border-radius: 10px; margin: 15px 0; border-left: 4px solid #ff6b6b;">
            <code style="color: #ff6b6b; font-size: 28px; font-weight: bold;">{simplified}</code>
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
            <div style="background: #0f0f23; padding: 10px; border-radius: 8px; text-align: center;">
                <span style="color: #888;">RMSE</span><br>
                <span style="color: #00d4ff; font-size: 16px; font-weight: bold;">{best.rmse:.6f}</span>
            </div>
            <div style="background: #0f0f23; padding: 10px; border-radius: 8px; text-align: center;">
                <span style="color: #888;">Nodos</span><br>
                <span style="color: #00d4ff; font-size: 16px; font-weight: bold;">{best.complexity}</span>
            </div>
            <div style="background: #0f0f23; padding: 10px; border-radius: 8px; text-align: center;">
                <span style="color: #888;">Tiempo</span><br>
                <span style="color: #00d4ff; font-size: 16px; font-weight: bold;">{search_time:.2f}s</span>
            </div>
            <div style="background: #0f0f23; padding: 10px; border-radius: 8px; text-align: center;">
                <span style="color: #888;">Device</span><br>
                <span style="color: #4ade80; font-size: 16px; font-weight: bold;">{DEVICE.type.upper()}</span>
            </div>
        </div>
        <div style="margin-top: 15px; padding: 10px; background: #0f0f23; border-radius: 8px;">
            <span style="color: #888;">Patrón:</span> 
            <span style="color: #ffd93d;">{pattern['type']}</span> 
            <span style="color: #666;">({pattern['confidence']:.0%})</span>
        </div>
    </div>
    """
    
    # Predictions table
    pred_html = '<table style="width: 100%; border-collapse: collapse; background: #1a1a2e; border-radius: 10px; overflow: hidden;">'
    pred_html += '<tr style="background: #16213e;"><th style="padding: 10px; color: #00d4ff;">X</th><th style="color: #00d4ff;">Pred</th><th style="color: #00d4ff;">Real</th><th style="color: #00d4ff;">Δ</th></tr>'
    for i in range(min(8, len(x))):
        delta = abs(y_pred[i] - y[i])
        color = "#4ade80" if delta < 0.1 else "#fbbf24" if delta < 1 else "#ef4444"
        pred_html += f'<tr style="border-bottom: 1px solid #333;"><td style="padding: 8px; color: white; text-align: center;">{x[i]:.2f}</td><td style="color: white; text-align: center;">{y_pred[i]:.4f}</td><td style="color: white; text-align: center;">{y[i]:.4f}</td><td style="color: {color}; text-align: center; font-weight: bold;">{delta:.4f}</td></tr>'
    pred_html += '</table>'
    
    # Alternatives
    alt_html = '<div style="background: #1a1a2e; padding: 15px; border-radius: 10px;">'
    alt_html += '<h4 style="color: #00d4ff; margin-top: 0;">🔀 Alternativas</h4>'
    for i, sol in enumerate(pareto.solutions[:4]):
        alt_html += f'<div style="padding: 5px 10px; margin: 5px 0; background: #0f0f23; border-radius: 5px; border-left: 3px solid {"#00d4ff" if i == 0 else "#666"};"><code style="color: {"#ff6b6b" if i == 0 else "#888"};">{sol.formula}</code> <span style="color: #666; font-size: 12px;">RMSE: {sol.rmse:.4f}</span></div>'
    alt_html += '</div>'
    
    return result_html, fig, pred_html, alt_html, simplified


def generate_example(tipo):
    """Generate example data."""
    if tipo == "lineal":
        x = np.linspace(1, 10, 10)
        y = 2 * x + 3
    elif tipo == "cuadratico":
        x = np.linspace(-5, 5, 11)
        y = x**2 + 1
    elif tipo == "trig":
        x = np.linspace(0, 6.28, 20)
        y = np.sin(x)
    elif tipo == "exp":
        x = np.linspace(0, 5, 15)
        y = 2 * np.exp(0.5 * x)
    else:
        x = np.linspace(1, 10, 10)
        y = 2 * x + 3
    
    return ", ".join([f"{v:.2f}" for v in x]), ", ".join([f"{v:.4f}" for v in y])
