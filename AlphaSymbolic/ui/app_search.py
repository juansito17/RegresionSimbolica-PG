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

from AlphaSymbolic.core.grammar import ExpressionTree
from AlphaSymbolic.search.beam_search import BeamSearch
from AlphaSymbolic.search.mcts import MCTS
from AlphaSymbolic.search.hybrid_search import hybrid_solve
from AlphaSymbolic.utils.simplify import simplify_tree
from AlphaSymbolic.search.pareto import ParetoFront
from AlphaSymbolic.utils.detect_pattern import detect_pattern
from AlphaSymbolic.utils.optimize_constants import optimize_constants, substitute_constants
from AlphaSymbolic.ui.app_core import get_model


def parse_data(x_str, y_str):
    """
    Parse input strings.
    Supports:
    1. 1D Comma-separated: "1, 2, 3"
    2. 2D Multi-line/Semi-colon: "1,2; 3,4" or "1 2\n3 4"
    """
    try:
        # Pre-process: standardize separators
        x_str = x_str.strip()
        y_str = y_str.strip()
        
        # Check for multi-line or semi-colon (Multi-Variable)
        is_multivar = '\n' in x_str or ';' in x_str
        
        if is_multivar:
            # Split into rows
            rows = [r.strip() for r in x_str.replace(';', '\n').split('\n') if r.strip()]
            # Parse each row
            x_data = []
            for r in rows:
                # Handle comma or space
                vals = [float(v) for v in r.replace(',', ' ').split()]
                x_data.append(vals)
            x = np.array(x_data, dtype=np.float64)
            
            # Y should also be checked, usually 1D but input might be multi-line
            y_data = [float(v) for v in y_str.replace(';', '\n').replace(',', ' ').split()]
            y = np.array(y_data, dtype=np.float64)
            
        else:
            # Legacy 1D
            x = np.array([float(v.strip()) for v in x_str.split(',')], dtype=np.float64)
            y = np.array([float(v.strip()) for v in y_str.split(',')], dtype=np.float64)
            
            # Ensure X is (N, 1) or (N,) depending on usage. 
            # Logic mostly expects (N,) for 1D, but model needs (N, 1).
            # Let's keep (N,) for 1D to not break existing plots, handling shape later.
        
        if len(x) != len(y):
            return None, None, f"Error: Cantidad de muestras X ({len(x)}) != Y ({len(y)})"
            
        return x, y, None
    except Exception as e:
        return None, None, f"Error parseando datos: {str(e)}"


def create_fit_plot(x, y, y_pred, formula):
    """Create a plot showing data vs prediction."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=(8, 5), facecolor='#1a1a2e')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#1a1a2e')
    
    # Check dimensions
    
    # Modern dark theme styling
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    
    # Plot Real Data
    if x.ndim > 1 and x.shape[1] > 1:
        # Multi-variable: Plot only target values (Y) vs Index
        indices = np.arange(len(y))
        ax.scatter(indices, y, color='#00d4ff', s=100, label='Datos Reales', zorder=3, edgecolors='white', linewidth=1.5)
    else:
        # 1D: Plot X vs Y
        if x.ndim > 1: x = x.flatten()
        ax.scatter(x, y, color='#00d4ff', s=100, label='Datos Reales', zorder=3, edgecolors='white', linewidth=1.5)

    # Plot Prediction (Handle NaNs)
    if y_pred is not None:
        if x.ndim > 1 and x.shape[1] > 1:
            indices = np.arange(len(y_pred))
            # Filter NaNs
            valid = np.isfinite(y_pred)
            if valid.any():
                ax.plot(indices[valid], y_pred[valid], color='#ff6b6b', linewidth=3, label='Prediccion', zorder=2)
        else:
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_pred_sorted = y_pred[sort_idx]
            
            # Filter NaNs
            valid = np.isfinite(y_pred_sorted)
            if valid.any():
                ax.plot(x_sorted[valid], y_pred_sorted[valid], color='#ff6b6b', linewidth=3, label='Prediccion', zorder=2)
        
        ax.set_xlabel('X', color='white', fontsize=12)
        ax.set_ylabel('Y', color='white', fontsize=12)
        ax.set_title('Ajuste de la Formula', color='white', fontsize=14, fontweight='bold')
        ax.legend(facecolor='#16213e', edgecolor='#00d4ff', labelcolor='white')

    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='white')
    
    for spine in ax.spines.values():
        spine.set_color('#00d4ff')
    
    fig.tight_layout()
    return fig


def solve_formula(x_str, y_str, beam_width, search_method, max_workers=4, pop_size=100000, use_log=False, progress=gr.Progress()):
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
        return "No se encontraron formulas validas", None, "", "", ""
    
    progress(0.7, desc="Optimizando constantes...")
    pareto = ParetoFront()
    pareto.add_from_results(results)
    best = pareto.get_best_by_rmse()
    
    if not best:
        return "Error en optimizacion", None, "", "", ""
    
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
            pass
    
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
        pass
    
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
        print(f"Plot Eval Error: {e}")
        y_pred = np.zeros_like(y)
        eval_warning = f"<div style='color: #ef4444; font-weight: bold; margin-bottom: 10px;'>Plot Error: {str(e)}</div>"
    
    fig = create_fit_plot(x, y, y_pred, display_formula)
    
    # Format results
    result_html = f"""
    {eval_warning}
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; border: 2px solid #00d4ff;">
        <h2 style="color: #00d4ff; margin: 0; font-size: 24px;">Formula Encontrada</h2>
        <div style="background: #0f0f23; padding: 15px; border-radius: 10px; margin: 15px 0; border-left: 4px solid #ff6b6b;">
            <code style="color: #ff6b6b; font-size: 28px; font-weight: bold;">{display_formula}</code>
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
    for i in range(min(50, len(y))):
        delta = abs(y_pred[i] - y[i])
        color = "#4ade80" if delta < 0.1 else "#fbbf24" if delta < 1 else "#ef4444"
        
        # Display X nicely
        x_val_str = ""
        if x.ndim > 1 and x.shape[1] > 1:
             x_val_str = f"[{', '.join([f'{v:.1f}' for v in x[i]])}]"
        else:
             xv = x[i] if x.ndim == 1 else x[i,0]
             x_val_str = f"{xv:.2f}"
             
        pred_html += f'<tr style="border-bottom: 1px solid #333;"><td style="padding: 8px; color: white; text-align: center;">{x_val_str}</td><td style="color: white; text-align: center;">{y_pred[i]:.4f}</td><td style="color: white; text-align: center;">{y[i]:.4f}</td><td style="color: {color}; text-align: center; font-weight: bold;">{delta:.4f}</td></tr>'
    pred_html += '</table>'
    
    # Alternatives
    alt_html = '<div style="background: #1a1a2e; padding: 15px; border-radius: 10px;">'
    alt_html += '<h4 style="color: #00d4ff; margin-top: 0;">Alternativas</h4>'
    for i, sol in enumerate(pareto.solutions[:4]):
        alt_html += f'<div style="padding: 5px 10px; margin: 5px 0; background: #0f0f23; border-radius: 5px; border-left: 3px solid {"#00d4ff" if i == 0 else "#666"};"><code style="color: {"#ff6b6b" if i == 0 else "#888"};">{sol.formula}</code> <span style="color: #666; font-size: 12px;">RMSE: {sol.rmse:.4f}</span></div>'
    alt_html += '</div>'
    
    return result_html, fig, pred_html, alt_html, display_formula


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
