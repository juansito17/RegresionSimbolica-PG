"""
Search/Solve functions for AlphaSymbolic Gradio App.
Supports both Beam Search and MCTS.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import gradio as gr
from AlphaSymbolic.ui.data_io import parse_input_data
from AlphaSymbolic.ui.formatting import (
    alternatives_list,
    escape_html,
    formula_card,
    metric_grid,
    prediction_table,
    status_panel,
)
from AlphaSymbolic.ui.logging_utils import format_exception, get_logger

from AlphaSymbolic.core.grammar import ExpressionTree
from AlphaSymbolic.search.beam_search import BeamSearch
from AlphaSymbolic.search.mcts import MCTS
from AlphaSymbolic.search.hybrid_search import hybrid_solve
from AlphaSymbolic.utils.simplify import simplify_tree
from AlphaSymbolic.search.pareto import ParetoFront
from AlphaSymbolic.utils.detect_pattern import detect_pattern
from AlphaSymbolic.utils.optimize_constants import optimize_constants, substitute_constants
from AlphaSymbolic.ui.app_core import get_model

logger = get_logger("UI.SEARCH")


def parse_data(x_str, y_str):
    """
    Parse input strings.
    Supports:
    1. 1D Comma-separated: "1, 2, 3"
    2. 2D Multi-line/Semi-colon: "1,2; 3,4" or "1 2\n3 4"
    """
    parsed = parse_input_data(x_str, y_str)
    return parsed.x, parsed.y, parsed.error


def create_fit_plot(x, y, y_pred, formula):
    """Create a plot showing data vs prediction."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=(8, 5), facecolor='#0f172a')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0f172a')
    
    # Check dimensions
    
    # Modern dark theme styling
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    
    # Plot Real Data
    if x.ndim > 1 and x.shape[1] > 1:
        # Multi-variable: Plot only target values (Y) vs Index
        indices = np.arange(len(y))
        ax.scatter(indices, y, color='#0ea5e9', s=72, label='Datos reales', zorder=3, edgecolors='white', linewidth=1.2)
    else:
        # 1D: Plot X vs Y
        if x.ndim > 1: x = x.flatten()
        ax.scatter(x, y, color='#0ea5e9', s=72, label='Datos reales', zorder=3, edgecolors='white', linewidth=1.2)

    # Plot Prediction (Handle NaNs)
    if y_pred is not None:
        if x.ndim > 1 and x.shape[1] > 1:
            indices = np.arange(len(y_pred))
            # Filter NaNs
            valid = np.isfinite(y_pred)
            if valid.any():
                ax.plot(indices[valid], y_pred[valid], color='#f97316', linewidth=2.5, label='Predicción', zorder=2)
        else:
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_pred_sorted = y_pred[sort_idx]
            
            # Filter NaNs
            valid = np.isfinite(y_pred_sorted)
            if valid.any():
                ax.plot(x_sorted[valid], y_pred_sorted[valid], color='#f97316', linewidth=2.5, label='Predicción', zorder=2)
        
        ax.set_xlabel('X', color='white', fontsize=12)
        ax.set_ylabel('Y', color='white', fontsize=12)
        ax.set_title('Ajuste de la fórmula', color='white', fontsize=14, fontweight='bold')
        ax.legend(facecolor='#111827', edgecolor='#334155', labelcolor='white')

    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='white')
    
    for spine in ax.spines.values():
        spine.set_color('#475569')
    
    fig.tight_layout()
    return fig


def solve_formula(x_str, y_str, beam_width, search_method, max_workers=4, pop_size=100000, use_log=False, progress=gr.Progress()):
    """Main solving function with search method selection."""
    x, y, error = parse_data(x_str, y_str)
    if error:
        return status_panel(error, "error"), None, "", "", ""
    
    MODEL, DEVICE = get_model()
    
    progress(0.1, desc=f"Analizando patron... [{DEVICE.type.upper()}]")
    pattern = detect_pattern(x, y)
    
    progress(0.3, desc=f"Buscando formulas ({search_method})... [{DEVICE.type.upper()}]")
    start_time = time.time()
    
    results = []
    
    num_vars = 1 if x.ndim == 1 else x.shape[1]
    
    if search_method == "Alpha-GP Hybrid":
        # Using hybrid search
        progress(0.4, desc="Fase 1: Neural Beam Search...")
        # Note: Hybrid search handles its own phases printing, but we want UI updates.
        # We pass beam_width. gp_timeout is increased to 30s to allow convergence on complex problems.
        hybrid_res = hybrid_solve(x, y, MODEL, DEVICE, beam_width=int(beam_width), gp_timeout=30, num_variables=num_vars, max_workers=max_workers, pop_size=pop_size, use_log=use_log)
        
        if hybrid_res and hybrid_res.get('formula'):
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
                 
                 # Use clamped input for safety during RMSE check
                 x_safe = np.clip(x, -100, 100)
                 y_pred_check = tree.evaluate(x_safe)
                 
                 # Handle NaNs in RMSE check
                 if np.any(np.isnan(y_pred_check)):
                     y_pred_check = np.nan_to_num(y_pred_check, nan=0.0)
                     
                 rmse_check = np.sqrt(np.mean((y_pred_check - y)**2))
                 if np.isnan(rmse_check): rmse_check = 1e6 # High but finite fallack
                 
                 results = [{
                     'tokens': tree.tokens,
                     'formula': tree.get_infix(),
                     'rmse': rmse_check,
                     'constants': {} # Constants are baked into the formula string
                 }]
    
    elif search_method == "Beam Search":
        searcher = BeamSearch(MODEL, DEVICE, beam_width=int(beam_width), max_length=25, num_variables=num_vars)
        results = searcher.search(x, y)
    else:  # MCTS
        mcts = MCTS(MODEL, DEVICE, max_simulations=int(beam_width) * 10, num_variables=num_vars)
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
        return status_panel("No se encontraron fórmulas válidas.", "warning"), None, "", "", ""
    
    progress(0.7, desc="Optimizando constantes...")
    pareto = ParetoFront()
    pareto.add_from_results(results)
    best = pareto.get_best_by_rmse()
    
    if not best:
        return status_panel("Error en optimización.", "error"), None, "", "", ""
    
    progress(0.9, desc="Procesando...") 
    tree = ExpressionTree(best.tokens)
    
    # Use the stored formula string directly (this is what GP/search found)
    display_formula = best.formula
    
    # If we have constants to substitute (Beam Search / MCTS with C placeholders)
    if best.constants:
        try:
            positions = tree.root.get_constant_positions()
            raw_infix = tree.get_infix()
            display_formula = substitute_constants(raw_infix, best.constants, positions)
        except:
            logger.debug("No se pudo sustituir constantes.", exc_info=True)
    
    # Try to simplify algebraically (x0 + x0 -> 2*x0, etc.)
    try:
        simplified = simplify_tree(tree)
        # Only use simplified if it:
        # 1. Is valid (not just a number, not "Invalid")
        # 2. Still contains a variable (x or x0-x9)  
        # 3. Is shorter or similar length
        if simplified and simplified != "Invalid":
            has_variable = any(v in simplified for v in ['x', 'x0', 'x1', 'x2', 'x3'])
            is_not_just_number = not simplified.replace('.', '').replace('-', '').isdigit()
            if has_variable and is_not_just_number:
                display_formula = simplified
    except:
            logger.debug("No se pudo simplificar la fórmula.", exc_info=True)
    
    eval_warning = ""
    # Safe Evaluation for Plotting
    try:
        # STABILITY FIX: Clamp x for plotting to avoid overflow/nans
        x_safe = np.clip(x, -100, 100)
        y_pred = tree.evaluate(x_safe, constants=best.constants)
        
        # Handle NaNs
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
             y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e9, neginf=-1e9)
    except Exception as e:
        logger.error("Error evaluando fórmula para plot: %s", format_exception(e))
        y_pred = np.zeros_like(y)
        eval_warning = status_panel(f"Error generando plot: {e}", "error")
    
    fig = create_fit_plot(x, y, y_pred, display_formula)
    
    result_html = (
        eval_warning
        + formula_card(display_formula, "Fórmula encontrada")
        + metric_grid(
            [
                ("RMSE", f"{best.rmse:.6f}"),
                ("Nodos", best.complexity),
                ("Tiempo", f"{search_time:.2f}s"),
                ("Método", search_method),
                ("Patrón", f"{escape_html(pattern['type'])} ({pattern['confidence']:.0%})"),
                ("Dispositivo", DEVICE.type.upper()),
            ]
        )
    )
    
    # Add constants if any
    # Add constants if any
    if best.constants:
        # Sort and format cleanly
        sorted_items = sorted(best.constants.items(), key=lambda x: str(x[0]))
        clean_consts = []
        for i, (k, v) in enumerate(sorted_items):
            clean_consts.append(f"C_{i+1}: {v:.4f}")
        const_str = "  |  ".join(clean_consts)
        
        result_html += status_panel(f"Constantes: {const_str}", "info")
    
    # Predictions table
    pred_rows = []
    for i in range(min(50, len(y))):
        delta = abs(y_pred[i] - y[i])
        
        # Display X nicely
        x_val_str = ""
        if x.ndim > 1 and x.shape[1] > 1:
             x_val_str = f"[{', '.join([f'{v:.1f}' for v in x[i]])}]"
        else:
             xv = x[i] if x.ndim == 1 else x[i,0]
             x_val_str = f"{xv:.2f}"
        pred_rows.append((x_val_str, f"{y_pred[i]:.4f}", f"{y[i]:.4f}", f"{delta:.4f}"))
    pred_html = prediction_table(pred_rows)
    
    # Alternatives
    alt_html = alternatives_list([(sol.formula, f"{sol.rmse:.4f}") for sol in pareto.solutions[:4]])
    
    return result_html, fig, pred_html, alt_html, display_formula


def generate_example(tipo):
    """Generate example data."""
    if tipo == "lineal":
        x = np.linspace(1, 10, 10)
        y = 2 * x + 3
        x_fmt = "{:.2f}"
        y_fmt = "{:.4f}"
    elif tipo == "cuadratico":
        x = np.linspace(-5, 5, 11)
        y = x**2 + 1
        x_fmt = "{:.2f}"
        y_fmt = "{:.4f}"
    elif tipo == "trig":
        x = np.round(np.linspace(0, 2 * np.pi, 21), 6)
        y = np.sin(x)
        y[np.isclose(y, 0, atol=1e-6)] = 0.0
        x_fmt = "{:.6f}"
        y_fmt = "{:.6f}"
    elif tipo == "exp":
        x = np.linspace(0, 5, 15)
        y = 2 * np.exp(0.5 * x)
        x_fmt = "{:.2f}"
        y_fmt = "{:.4f}"
    elif tipo == "mv_lineal":
        x0, x1 = np.meshgrid(np.linspace(1, 5, 4), np.linspace(1, 5, 4))
        x = np.column_stack((x0.flatten(), x1.flatten()))
        x = np.array([[float(f"{v:.2f}") for v in row] for row in x])
        y = 2 * x[:, 0] + 3 * x[:, 1] - 1
        x_fmt = "{:.2f}"
        y_fmt = "{:.4f}"
    elif tipo == "mv_cuadratico":
        x0, x1 = np.meshgrid(np.linspace(-2, 2, 4), np.linspace(-2, 2, 4))
        x = np.column_stack((x0.flatten(), x1.flatten()))
        x = np.array([[float(f"{v:.2f}") for v in row] for row in x])
        y = x[:, 0]**2 + x[:, 1]**2 + 1
        x_fmt = "{:.2f}"
        y_fmt = "{:.4f}"
    elif tipo == "mv_trig":
        x0, x1 = np.meshgrid(np.linspace(0, np.pi, 4), np.linspace(0, np.pi, 4))
        x = np.column_stack((x0.flatten(), x1.flatten()))
        x = np.array([[float(f"{v:.4f}") for v in row] for row in x])
        y = np.sin(x[:, 0]) + np.cos(x[:, 1])
        x_fmt = "{:.4f}"
        y_fmt = "{:.4f}"
    elif tipo == "mv_exp":
        x0, x1 = np.meshgrid(np.linspace(0, 3, 4), np.linspace(0, 3, 4))
        x = np.column_stack((x0.flatten(), x1.flatten()))
        x = np.array([[float(f"{v:.2f}") for v in row] for row in x])
        y = np.exp(0.5 * x[:, 0]) + 2 * x[:, 1]
        x_fmt = "{:.2f}"
        y_fmt = "{:.4f}"
    elif tipo == "mv_lineal_3d":
        x0, x1, x2 = np.meshgrid(np.linspace(1, 3, 3), np.linspace(1, 3, 3), np.linspace(1, 3, 3))
        x = np.column_stack((x0.flatten(), x1.flatten(), x2.flatten()))
        x = np.array([[float(f"{v:.2f}") for v in row] for row in x])
        y = x[:, 0] + 2 * x[:, 1] - 1.5 * x[:, 2] + 3
        x_fmt = "{:.2f}"
        y_fmt = "{:.4f}"
    elif tipo == "mv_cuadratico_3d":
        x0, x1, x2 = np.meshgrid(np.linspace(-2, 2, 3), np.linspace(-2, 2, 3), np.linspace(-2, 2, 3))
        x = np.column_stack((x0.flatten(), x1.flatten(), x2.flatten()))
        x = np.array([[float(f"{v:.2f}") for v in row] for row in x])
        y = x[:, 0] * x[:, 1] + x[:, 2]**2 - 1
        x_fmt = "{:.2f}"
        y_fmt = "{:.4f}"
    elif tipo == "mv_4d":
        x0, x1, x2, x3 = np.meshgrid(np.linspace(1, 2, 2), np.linspace(1, 2, 2), np.linspace(0, np.pi, 2), np.linspace(0, 1, 2))
        x = np.column_stack((x0.flatten(), x1.flatten(), x2.flatten(), x3.flatten()))
        x = np.array([[float(f"{v:.4f}") for v in row] for row in x])
        y = x[:, 0] * x[:, 1] + np.sin(x[:, 2]) - np.exp(0.5 * x[:, 3])
        x_fmt = "{:.4f}"
        y_fmt = "{:.4f}"
    else:
        x = np.linspace(1, 10, 10)
        y = 2 * x + 3
        x_fmt = "{:.2f}"
        y_fmt = "{:.4f}"
    
    if x.ndim == 1:
        x_str = ", ".join([x_fmt.format(v) for v in x])
    else:
        x_str = "\n".join(" ".join(x_fmt.format(v) for v in row) for row in x)
        
    y_str = ", ".join([y_fmt.format(v) for v in y])
    return x_str, y_str
