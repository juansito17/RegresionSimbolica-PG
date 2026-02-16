import gradio as gr
import torch
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
import html
from typing import List, Optional
from core.gpu import TensorGeneticEngine
from core.gpu.config import GpuGlobals
from ui.app_search import parse_data, create_fit_plot, generate_example
from ui.app_core import get_device
import pandas as pd

# Global state to manage the live engine
LIVE_ENGINE = None

def get_gpu_live_tab():
    """Defines the UI for the GPU Evolution tab."""
    with gr.Row():
        with gr.Column(scale=1, min_width=400):
            gr.Markdown("## ⚡ Evolución GPU en Tiempo Real")
            gr.HTML('<div style="margin-bottom: 20px; padding: 5px 15px; background: rgba(34, 197, 94, 0.1); border-radius: 20px; color: #22c55e; font-size: 0.8rem; border: 1px solid rgba(34, 197, 94, 0.2); display: inline-block;">🛡️ MODO PURE GPU GA (Sin Red Neuronal)</div>')
            # Load from CSV & Examples
            with gr.Accordion("📂 Cargar Datos", open=False):
                with gr.Row():
                    csv_file = gr.File(label="Cargar desde CSV", file_types=[".csv", ".txt"])
                with gr.Row():
                    btn_lin = gr.Button("Lineal")
                    btn_quad = gr.Button("Cuad")
                    btn_trig = gr.Button("Trig")
                    btn_exp = gr.Button("Exp")

            x_input = gr.Textbox(label="Features (X)", placeholder="1, 2, 3...", lines=3)
            y_input = gr.Textbox(label="Target (Y)", placeholder="2, 4, 6...", lines=3)
            
            with gr.Row():
                pop_size = gr.Slider(10_000, 4_000_000, value=1_000_000, step=10_000, label="Población")
                islands = gr.Slider(1, 100, value=50, step=1, label="Islas")
            
            with gr.Row():
                max_const = gr.Slider(1, 50, value=10, step=1, label="Máx Constantes")
                timeout = gr.Slider(0, 300, value=60, step=5, label="Timeout (s)", info="0 para infinito")
                use_log = gr.Checkbox(label="Use Log Transform", value=False, info="Fit to log(Y) (Recommended for exponential data)")
            
            # Operator toggles
            with gr.Accordion("🔧 Operadores Disponibles", open=False):
                gr.Markdown("*Desactiva operadores para simplificar las fórmulas generadas*")
                with gr.Row():
                    op_sin = gr.Checkbox(label="sin", value=True)
                    op_cos = gr.Checkbox(label="cos", value=True)
                    op_tan = gr.Checkbox(label="tan", value=False)
                    op_log = gr.Checkbox(label="log", value=True)
                    op_exp = gr.Checkbox(label="exp", value=True)
                with gr.Row():
                    op_asin = gr.Checkbox(label="asin", value=False)
                    op_acos = gr.Checkbox(label="acos", value=False)
                    op_atan = gr.Checkbox(label="atan", value=False)
                with gr.Row():
                    op_sqrt = gr.Checkbox(label="sqrt", value=True)
                    op_abs = gr.Checkbox(label="abs", value=True)
                    op_floor = gr.Checkbox(label="floor", value=False)
                    op_ceil = gr.Checkbox(label="ceil", value=False)
                    op_sign = gr.Checkbox(label="sign", value=False)
                with gr.Row():
                    op_gamma = gr.Checkbox(label="gamma", value=True)
                    op_lgamma = gr.Checkbox(label="lgamma", value=True)
                    op_fact = gr.Checkbox(label="fact!", value=True)
                    op_mod = gr.Checkbox(label="%", value=False, info="Modulo (slow)")
                    op_pow = gr.Checkbox(label="^", value=True, info="Power")

            with gr.Row():
                stop_btn = gr.Button("🛑 DETENER", variant="stop")
                start_btn = gr.Button("🚀 INICIAR EVOLUCIÓN", variant="primary")
            
            status_html = gr.HTML(value='<div style="padding: 10px; background: #0f0f23; border-radius: 8px; border-left: 3px solid #64748b; color: #64748b;">Esperando inicio...</div>')

        with gr.Column(scale=1, min_width=400):
            gr.Markdown("## Dashboard de Evolución")
            current_best_formula = gr.HTML(label="Mejor Fórmula Actual")
            stats_display = gr.HTML(label="Estadísticas")
            live_plot = gr.Plot(label="Ajuste en Tiempo Real")
    
    # --- Helper Functions ---
    def load_csv(file):
        if file is None: return None, None
        try:
            df = pd.read_csv(file.name, header=None)
            # Assume last column is Y, rest are X
            if df.shape[1] < 2:
                return "Error: CSV debe tener al menos 2 columnas", ""
            
            x_data = df.iloc[:, :-1].values
            y_data = df.iloc[:, -1].values
            
            # Convert to string format for Textbox
            # X: Multi-line if multi-var, or comma sep?
            # parse_data handles semi-colon or newlines for multi-row
            # Let's use standard: rows separated by newline, features by space
            x_str = ""
            for row in x_data:
                x_str += " ".join(map(str, row)) + "\n"
            
            y_str = " ".join(map(str, y_data)) # Y usually 1D space/comma sep
            
            return x_str.strip(), y_str.strip()
        except Exception as e:
            return f"Error leyendo CSV: {e}", ""

    # --- Event Handlers ---
    csv_file.upload(load_csv, inputs=[csv_file], outputs=[x_input, y_input])
    
    btn_lin.click(lambda: generate_example("lineal"), outputs=[x_input, y_input])
    btn_quad.click(lambda: generate_example("cuadratico"), outputs=[x_input, y_input])
    btn_trig.click(lambda: generate_example("trig"), outputs=[x_input, y_input])
    btn_exp.click(lambda: generate_example("exp"), outputs=[x_input, y_input])

    def stop_evolution():
        global LIVE_ENGINE
        if LIVE_ENGINE:
            LIVE_ENGINE.stop_flag = True
        return '<div style="padding: 10px; background: #0f0f23; border-radius: 8px; border-left: 3px solid #ef4444; color: #ef4444;">Deteniendo motor...</div>'

    start_btn.click(
        run_live_gpu_evolution,
        inputs=[x_input, y_input, pop_size, islands, max_const, timeout, use_log,
                op_sin, op_cos, op_tan, op_log, op_exp, op_sqrt, op_abs, op_floor, op_ceil, op_sign,
                op_gamma, op_lgamma, op_fact, op_mod, op_pow, op_asin, op_acos, op_atan],
        outputs=[status_html, current_best_formula, stats_display, live_plot]
    )
    stop_btn.click(stop_evolution, outputs=[status_html])

def run_live_gpu_evolution(x_str, y_str, pop_size, n_islands, max_constants, timeout_sec, use_log_transform,
                           op_sin, op_cos, op_tan, op_log, op_exp, op_sqrt, op_abs, op_floor, op_ceil, op_sign,
                           op_gamma, op_lgamma, op_fact, op_mod, op_pow, op_asin, op_acos, op_atan):
    """Generator that runs the GPU engine and yields updates to the UI."""
    global LIVE_ENGINE
    
    # 1. Parse Data
    x, y, error = parse_data(x_str, y_str)
    if error:
        yield f'<div style="color: #ef4444;">{error}</div>', "", "", None
        return

    device = get_device()
    num_vars = 1 if x.ndim == 1 else x.shape[1]
    
    # 2. Clean up previous engine if any (VRAM safety)
    if LIVE_ENGINE:
        del LIVE_ENGINE
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    # --- IMPORTANT: Configure GpuGlobals BEFORE creating Engine ---
    # (Engine's Grammar reads these at construction time)
    
    # Log Transform -> Loss Function & Explicit Data Transform flag
    GpuGlobals.USE_LOG_TRANSFORMATION = use_log_transform
    if use_log_transform:
        GpuGlobals.LOSS_FUNCTION = 'RMSLE'
    else:
        GpuGlobals.LOSS_FUNCTION = 'RMSE'
        
    # Disable Neural features for this tab (Pure GPU mode)
    GpuGlobals.USE_NEURAL_FLASH = False
    GpuGlobals.USE_ALPHA_MCTS = False
    
    # Population config
    GpuGlobals.POP_SIZE = int(pop_size)
    GpuGlobals.NUM_ISLANDS = int(n_islands)
    GpuGlobals.GENERATIONS = 1_000_000_000 # Effectively infinite for live mode
    
    # Operator toggles from UI - MUST be set before Engine init
    GpuGlobals.USE_OP_SIN = op_sin
    GpuGlobals.USE_OP_COS = op_cos
    GpuGlobals.USE_OP_TAN = op_tan
    GpuGlobals.USE_OP_ASIN = op_asin
    GpuGlobals.USE_OP_ACOS = op_acos
    GpuGlobals.USE_OP_ATAN = op_atan
    GpuGlobals.USE_OP_LOG = op_log
    GpuGlobals.USE_OP_EXP = op_exp
    GpuGlobals.USE_OP_SQRT = op_sqrt
    GpuGlobals.USE_OP_ABS = op_abs
    GpuGlobals.USE_OP_FLOOR = op_floor
    GpuGlobals.USE_OP_CEIL = op_ceil
    GpuGlobals.USE_OP_SIGN = op_sign
    GpuGlobals.USE_OP_GAMMA = op_gamma
    GpuGlobals.USE_OP_LGAMMA = op_lgamma
    GpuGlobals.USE_OP_FACT = op_fact
    GpuGlobals.USE_OP_MOD = op_mod
    GpuGlobals.USE_OP_POW = op_pow
    
    # Disable Sniper temporarily to debug crash
    GpuGlobals.USE_SNIPER = False
    
    # 3. Initialize Engine (NOW it will read the correct operator flags)
    yield f'<div style="color: #22d3ee;">Inicializando Motor GPU ({pop_size:,} individuos)...</div>', "", "", None
    
    LIVE_ENGINE = TensorGeneticEngine(
        device=device,
        pop_size=int(pop_size),
        n_islands=int(n_islands),
        num_variables=num_vars,
        max_constants=int(max_constants),
        model=None # Explicitly no model for Pure GPU mode
    )
    
    LIVE_ENGINE.stop_flag = False
    
    # State for the callback
    state = {
        "gen": 0,
        "best_rmse": float('inf'),
        "best_formula": "",
        "start_time": time.time(),
        "last_update": 0,
        "speed": 0
    }

    def live_callback(gen, best_rmse, best_rpn, best_consts, is_new_best, island_idx):
        state["gen"] = gen
        state["best_rmse"] = best_rmse
        if is_new_best or not state["best_formula"]:
            state["best_formula"] = LIVE_ENGINE.rpn_to_infix(best_rpn, best_consts)
        
        # Calculate speed
        now = time.time()
        elapsed = now - state["start_time"]
        if elapsed > 0:
            state["speed"] = (gen * pop_size) / elapsed

    # Run engine in a small increments or with a frequent callback
    # The actual engine.run is a blocking loop with a callback.
    # To make it yield, we'd need to modify engine.run to be a generator
    # OR run it in a thread and use a queue.
    # But Gradio generators expect the loop to be IN the generator.
    
    # Let's modify engine.run slightly (in our minds) or just wrap it.
    # Actually, I can't easily make engine.run yield without big changes.
    # Alternative: run engine.run in a separate thread and have the generator poll a result queue.
    
    import queue
    import threading
    
    update_queue = queue.Queue()
    
    def wrapped_callback(gen, best_rmse, best_rpn, best_consts, is_new_best, island_idx):
        if is_new_best or gen % 10 == 0:
            # Simplify before displaying
            try:
                rpn_batch = best_rpn.unsqueeze(0)  # [1, L]
                consts_batch = best_consts.unsqueeze(0)  # [1, K]
                simp_pop, simp_consts, _ = LIVE_ENGINE.gpu_simplifier.simplify_batch(rpn_batch, consts_batch)
                best_rpn_simp = simp_pop[0]
                best_consts_simp = simp_consts[0] if simp_consts is not None else best_consts
            except:
                best_rpn_simp = best_rpn
                best_consts_simp = best_consts
            
            # Initial assumption: Simplified is best
            final_rpn = best_rpn_simp
            final_consts = best_consts_simp
            formula = LIVE_ENGINE.rpn_to_infix(final_rpn, final_consts)
            
            # FALLBACK: If simplified is invalid, revert to original
            if formula == "Invalid":
                print(f"[DEBUG UI] Simplified Invalid! Reverting to original for plot/display...")
                
                # Debug Tokens
                try:
                    toks = [LIVE_ENGINE.grammar.id_to_token.get(t.item(), str(t.item())) for t in best_rpn_simp if t.item() in LIVE_ENGINE.grammar.id_to_token]
                    print(f"Bad Simp Tokens: {toks}")
                except: pass

                # REVERT to Original
                final_rpn = best_rpn
                final_consts = best_consts
                formula = LIVE_ENGINE.rpn_to_infix(final_rpn, final_consts)
                
                # If still invalid, print original tokens
                if formula == "Invalid":
                    print(f"[DEBUG UI] Original ALSO Invalid!")
                    try:
                        toks_orig = [LIVE_ENGINE.grammar.id_to_token.get(t.item(), str(t.item())) for t in best_rpn if t.item() in LIVE_ENGINE.grammar.id_to_token]
                        print(f"Bad Orig Tokens: {toks_orig}")
                    except: pass
            
            update_queue.put({
                "gen": gen,
                "rmse": best_rmse,
                "formula": formula, # Now matches final_rpn
                "new_best": is_new_best,
                "island": island_idx,
                "rpn": final_rpn.clone(),       # CORRECTED: Use valid RPN
                "consts": final_consts.clone()  # CORRECTED: Use valid Consts
            })

    # Prepare inputs for engine
    # Re-use logic from run_gpu_console.py for x_input
    if num_vars == 1:
        x_input_tensor = x if x.ndim == 2 else x.reshape(-1, 1)
    else:
        x_input_tensor = x
        
    thread = threading.Thread(
        target=LIVE_ENGINE.run,
        args=(x_input_tensor, y, None, timeout_sec if timeout_sec > 0 else None, wrapped_callback)
    )
    thread.start()
    
    start_time = time.time()
    last_gen = 0
    start_time = time.time()
    last_gen = 0
    last_fig = None
    last_best_html = ""
    last_stats_html = ""
    
    while thread.is_alive() or not update_queue.empty():
        try:
            # Poll for updates
            data = update_queue.get(timeout=0.1)
            
            gen = data["gen"]
            rmse = data["rmse"]
            formula = data["formula"]
            
            elapsed = time.time() - start_time
            speed = (gen * pop_size) / elapsed if elapsed > 0 else 0
            
            status = f"""
            <div style="padding: 10px; background: #0f172a; border-radius: 8px; border-left: 3px solid #22d3ee;">
                <b style="color: #22d3ee;">Ejecutando...</b> | Gen: {gen:,} | Velocidad: {speed:,.0f} evals/s
            </div>
            """
            
            # Sanitize HTML to prevent rendering issues with <, >
            formula_safe = html.escape(str(formula)) if formula else "..."
            
            best_html = f"""
            <div style="background: #0f0f23; padding: 15px; border-radius: 10px; border-left: 4px solid #ff6b6b; margin-bottom: 10px; min-height: 60px; display: flex; align-items: center;">
                <code style="color: #ff6b6b; font-size: 20px; font-weight: bold; overflow-wrap: break-word; white-space: pre-wrap; width: 100%;">{formula_safe}</code>
            </div>
            """
            
            stats_html = f"""
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                <div style="background: #0f172a; padding: 10px; border-radius: 8px; text-align: center;">
                    <span style="color: #888; font-size: 0.8rem;">RMSE</span><br>
                    <span style="color: #22d3ee; font-size: 1.2rem; font-weight: bold;">{rmse:.6e}</span>
                </div>
                <div style="background: #0f172a; padding: 10px; border-radius: 8px; text-align: center;">
                    <span style="color: #888; font-size: 0.8rem;">Generación</span><br>
                    <span style="color: #22d3ee; font-size: 1.2rem; font-weight: bold;">{gen:,}</span>
                </div>
                <div style="background: #0f172a; padding: 10px; border-radius: 8px; text-align: center;">
                    <span style="color: #888; font-size: 0.8rem;">Tiempo</span><br>
                    <span style="color: #22d3ee; font-size: 1.2rem; font-weight: bold;">{elapsed:.1f}s</span>
                </div>
                <div style="background: #0f172a; padding: 10px; border-radius: 8px; text-align: center;">
                    <span style="color: #888; font-size: 0.8rem;">Islas</span><br>
                    <span style="color: #22d3ee; font-size: 1.2rem; font-weight: bold;">{n_islands}</span>
                </div>
            </div>
            """
            
            # Create plot (only every few updates to save CPU)
            fig = None
            if data["new_best"] or gen % 100 == 0:
                try:
                    # Evaluate using GPU engine directly for 100% accuracy
                    rpn_tensor = data["rpn"].unsqueeze(0).to(LIVE_ENGINE.device)
                    consts_tensor = data["consts"].unsqueeze(0).to(LIVE_ENGINE.device, dtype=LIVE_ENGINE.dtype)
                    
                    # Prepare x properly for engine
                    if num_vars == 1:
                        x_t = torch.tensor(x if x.ndim == 1 else x.flatten(), device=LIVE_ENGINE.device, dtype=LIVE_ENGINE.dtype).unsqueeze(0) # [1, N]
                    else:
                        x_t = torch.tensor(x.T if x.shape[0] != num_vars else x, device=LIVE_ENGINE.device, dtype=LIVE_ENGINE.dtype)
                    
                    preds, sp, err = LIVE_ENGINE.evaluator.vm.eval(rpn_tensor, x_t, consts_tensor)
                    is_valid = (sp == 1) & (~err)
                    
                    y_pred = preds.squeeze().cpu().numpy()
                    valid_mask = is_valid.squeeze().cpu().numpy()
                    y_pred = np.where(valid_mask, y_pred, np.nan)
                    
                    # Inverse transform if needed
                    if use_log_transform: 
                        y_pred = np.exp(y_pred)
                        
                    fig = create_fit_plot(x, y, y_pred, formula)
                    last_fig = fig
                except Exception as e:
                    # Fallback if GPU eval fails
                    pass
            
            last_best_html = best_html
            last_stats_html = stats_html
            yield status, best_html, stats_html, last_fig
            
        except queue.Empty:
            if not thread.is_alive():
                break
            continue

    elapsed = time.time() - start_time
    yield (
        f'<div style="padding: 10px; background: #0f172a; border-radius: 8px; border-left: 3px solid #22c55e;"><b>Finalizado</b> en {elapsed:.2f}s</div>',
        last_best_html, # Persist last best
        last_stats_html, # Persist last stats
        last_fig  # Persist final plot
    )
