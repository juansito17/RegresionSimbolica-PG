"""
Search/Solve functions for AlphaSymbolic Gradio App.
Supports both Beam Search and MCTS.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import gradio as gr

from core.grammar import ExpressionTree
from search.beam_search import BeamSearch
from search.mcts import MCTS
from search.hybrid_search import hybrid_solve
from utils.simplify import simplify_tree
from search.pareto import ParetoFront
from utils.detect_pattern import detect_pattern
from utils.optimize_constants import optimize_constants, substitute_constants
from ui.app_core import get_model


def parse_data(x_str, y_str):
    """Parse comma-separated input strings."""
    try:
        x = np.array([float(v.strip()) for v in x_str.split(',')], dtype=np.float64)
        y = np.array([float(v.strip()) for v in y_str.split(',')], dtype=np.float64)
        if len(x) != len(y):
            return None, None, "Error: X e Y deben tener igual longitud"
        return x, y, None
    except Exception as e:
        return None, None, f"Error: {str(e)}"


def create_fit_plot(x, y, y_pred, formula):
    """Create a plot showing data vs prediction."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    ax.scatter(x, y, color='#00d4ff', s=100, label='Datos Reales', zorder=3, edgecolors='white', linewidth=1)
    
    sort_idx = np.argsort(x)
    ax.plot(x[sort_idx], y_pred[sort_idx], color='#ff6b6b', linewidth=3, label='Prediccion', zorder=2)
    
    ax.set_xlabel('X', color='white', fontsize=12)
    ax.set_ylabel('Y', color='white', fontsize=12)
    ax.set_title('Ajuste de la Formula', color='white', fontsize=14, fontweight='bold')
    ax.legend(facecolor='#16213e', edgecolor='#00d4ff', labelcolor='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='white')
    
    for spine in ax.spines.values():
        spine.set_color('#00d4ff')
    
    plt.tight_layout()
    return fig


def solve_formula(x_str, y_str, beam_width, search_method, progress=gr.Progress()):
    """Main solving function with search method selection."""
    x, y, error = parse_data(x_str, y_str)
    if error:
        return error, None, "", "", ""
    
    MODEL, DEVICE = get_model()
    
    progress(0.1, desc=f"Analizando patron... [{DEVICE.type.upper()}]")
    pattern = detect_pattern(x, y)
    
    progress(0.3, desc=f"Buscando formulas ({search_method})... [{DEVICE.type.upper()}]")
    start_time = time.time()
    
    results = []
    
    if search_method == "Alpha-GP Hybrid":
        # Using hybrid search
        progress(0.4, desc="Fase 1: Neural Beam Search...")
        # Note: Hybrid search handles its own phases printing, but we want UI updates.
        # We pass beam_width. gp_timeout is increased to 30s to allow convergence on complex problems.
        hybrid_res = hybrid_solve(x, y, MODEL, DEVICE, beam_width=int(beam_width), gp_timeout=30)
        
        if hybrid_res:
            progress(0.9, desc="Procesando resultados GP...")
            # Convert infix string back to tokens for consistency
            tree = ExpressionTree.from_infix(hybrid_res['formula'])
            if tree.is_valid:
                 # Evaluate RMSE roughly (GP result should be good, but let's confirm)
                 # Optimization is already done by GP, but we might want to fine-tune 
                 # or at least extract constants if they are numbers in the string.
                 # The string from GP has numbers like 2.345 embedded.
                 # optimize_constants expects a tree with 'C' placeholders if we want to re-optimize.
                 # But GP output is fully instantiated.
                 # So we just evaluate.
                 
                 y_pred_check = tree.evaluate(x)
                 rmse_check = np.sqrt(np.mean((y_pred_check - y)**2))
                 
                 results = [{
                     'tokens': tree.tokens,
                     'formula': tree.get_infix(),
                     'rmse': rmse_check,
                     'constants': {} # Constants are baked into the formula string
                 }]
    
    elif search_method == "Beam Search":
        searcher = BeamSearch(MODEL, DEVICE, beam_width=int(beam_width), max_length=25)
        results = searcher.search(x, y)
    else:  # MCTS
        mcts = MCTS(MODEL, DEVICE, max_simulations=int(beam_width) * 10)
        result = mcts.search(x, y)
        if result and result.get('tokens'):
            tokens = result['tokens']
            tree = ExpressionTree(tokens)
            if tree.is_valid:
                constants, rmse = optimize_constants(tree, x, y)
                results = [{
                    'tokens': tokens,
                    'formula': tree.get_infix(),
                    'rmse': rmse,
                    'constants': constants
                }]
    
    search_time = time.time() - start_time
    
    if not results:
        return "No se encontraron formulas validas", None, "", "", ""
    
    progress(0.7, desc="Optimizando constantes...")
    pareto = ParetoFront()
    pareto.add_from_results(results)
    best = pareto.get_best_by_rmse()
    
    if not best:
        return "Error en optimizacion", None, "", "", ""
    
    progress(0.9, desc="Simplificando...")
    tree = ExpressionTree(best.tokens)
    simplified = simplify_tree(tree)
    y_pred = tree.evaluate(x, constants=best.constants)
    
    # Substitute constants for display
    substituted_formula = simplified
    if best.constants:
        try:
            positions = tree.root.get_constant_positions()
            # We use the raw infix for substitution to ensure matching C positions
            raw_infix = tree.get_infix()
            substituted_formula = substitute_constants(raw_infix, best.constants, positions)
        except:
            substituted_formula = simplified
    
    fig = create_fit_plot(x, y, y_pred, simplified)
    
    # Format results
    result_html = f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; border: 2px solid #00d4ff;">
        <h2 style="color: #00d4ff; margin: 0; font-size: 24px;">Formula Encontrada</h2>
        <div style="background: #0f0f23; padding: 15px; border-radius: 10px; margin: 15px 0; border-left: 4px solid #ff6b6b;">
            <code style="color: #ff6b6b; font-size: 28px; font-weight: bold;">{substituted_formula}</code>
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
                <span style="color: #888;">Metodo</span><br>
                <span style="color: #4ade80; font-size: 16px; font-weight: bold;">{search_method}</span>
            </div>
        </div>
        <div style="margin-top: 15px; padding: 10px; background: #0f0f23; border-radius: 8px;">
            <span style="color: #888;">Patron:</span> 
            <span style="color: #ffd93d;">{pattern['type']}</span> 
            <span style="color: #666;">({pattern['confidence']:.0%})</span>
            <span style="color: #888; margin-left: 20px;">Device:</span>
            <span style="color: #4ade80;">{DEVICE.type.upper()}</span>
        </div>
    """
    
    # Add constants if any
    # Add constants if any
    if best.constants:
        # Sort and format cleanly
        sorted_items = sorted(best.constants.items(), key=lambda x: str(x[0]))
        clean_consts = []
        for i, (k, v) in enumerate(sorted_items):
            clean_consts.append(f"C_{i+1}: {v:.4f}")
        const_str = "  |  ".join(clean_consts)
        
        result_html += f"""
        <div style="margin-top: 10px; padding: 10px; background: #0f0f23; border-radius: 8px; border-left: 3px solid #ffd93d;">
            <span style="color: #888;">Constantes:</span>
            <span style="color: #fff; font-family: monospace; margin-left: 10px;">{const_str}</span>
        </div>
        """
        
    result_html += "</div>"
    
    # Predictions table
    pred_html = '<table style="width: 100%; border-collapse: collapse; background: #1a1a2e; border-radius: 10px; overflow: hidden;">'
    pred_html += '<tr style="background: #16213e;"><th style="padding: 10px; color: #00d4ff;">X</th><th style="color: #00d4ff;">Pred</th><th style="color: #00d4ff;">Real</th><th style="color: #00d4ff;">Delta</th></tr>'
    for i in range(min(50, len(x))):
        delta = abs(y_pred[i] - y[i])
        color = "#4ade80" if delta < 0.1 else "#fbbf24" if delta < 1 else "#ef4444"
        pred_html += f'<tr style="border-bottom: 1px solid #333;"><td style="padding: 8px; color: white; text-align: center;">{x[i]:.2f}</td><td style="color: white; text-align: center;">{y_pred[i]:.4f}</td><td style="color: white; text-align: center;">{y[i]:.4f}</td><td style="color: {color}; text-align: center; font-weight: bold;">{delta:.4f}</td></tr>'
    pred_html += '</table>'
    
    # Alternatives
    alt_html = '<div style="background: #1a1a2e; padding: 15px; border-radius: 10px;">'
    alt_html += '<h4 style="color: #00d4ff; margin-top: 0;">Alternativas</h4>'
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
