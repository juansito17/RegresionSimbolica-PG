import gradio as gr
import torch
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
from AlphaSymbolic.core.gpu import TensorGeneticEngine
from AlphaSymbolic.core.gpu.config import GpuGlobals
from AlphaSymbolic.ui.app_search import parse_data, create_fit_plot, generate_example
from AlphaSymbolic.ui.app_core import get_device
import pandas as pd

from AlphaSymbolic.ui.formatting import formula_card, metric_grid, status_panel
from AlphaSymbolic.ui.live_state import LiveRunState
from AlphaSymbolic.ui.logging_utils import configure_logging, format_exception, get_logger
from AlphaSymbolic.core.grammar import ExpressionTree

logger = get_logger("UI.GPU")
ENGINE_CLS = TensorGeneticEngine

# Backward-compatible handle for legacy imports/debug consoles.
LIVE_ENGINE = None


def _fill_live_plot_predictions_with_formula(x, y_pred, formula):
    """Use the displayed formula as CPU fallback for NaN points in the live plot."""
    if y_pred is None or np.all(np.isfinite(y_pred)):
        return y_pred
    try:
        tree = ExpressionTree.from_infix(formula)
        if not tree or not tree.is_valid:
            return y_pred
        cpu_pred = np.asarray(tree.evaluate(x), dtype=float)
        if cpu_pred.shape != np.asarray(y_pred).shape:
            cpu_pred = cpu_pred.reshape(np.asarray(y_pred).shape)
        return np.where(np.isfinite(y_pred), y_pred, cpu_pred)
    except Exception:
        logger.debug("No se pudo completar predicciones del plot con la formula CPU.", exc_info=True)
        return y_pred

def get_gpu_live_tab(verbose_state=None):
    """Defines the UI for the GPU Evolution tab."""
    live_state = gr.State(LiveRunState())
    with gr.Row():
        with gr.Column(scale=1, min_width=360, elem_classes="as-sidebar"):
            gr.Markdown("## ⚡ Evolución GPU en Tiempo Real")
            gr.HTML('<div class="as-badge">MODO PURE GPU GA · Sin red neuronal</div>')
            # Load from CSV & Examples
            with gr.Accordion("📂 Cargar Datos", open=False):
                with gr.Row():
                    csv_file = gr.File(label="Cargar desde CSV", file_types=[".csv", ".txt"])
                gr.Markdown("**Ejemplos 1D**")
                with gr.Row():
                    btn_lin = gr.Button("Lineal")
                    btn_quad = gr.Button("Cuad")
                    btn_trig = gr.Button("Trig")
                    btn_exp = gr.Button("Exp")
                gr.Markdown("**Ejemplos Multivariable (2D)**")
                with gr.Row():
                    btn_mv_lin = gr.Button("MV Lineal")
                    btn_mv_quad = gr.Button("MV Cuad")
                    btn_mv_trig = gr.Button("MV Trig")
                    btn_mv_exp = gr.Button("MV Exp")
                gr.Markdown("**Ejemplos Multivariable (3D / 4D)**")
                with gr.Row():
                    btn_mv_lin3d = gr.Button("MV Lineal (3D)")
                    btn_mv_quad3d = gr.Button("MV Cuad (3D)")
                    btn_mv_4d = gr.Button("MV Complejo (4D)")

            x_input = gr.Textbox(label="Features (X)", placeholder="1, 2, 3...", lines=3)
            y_input = gr.Textbox(label="Target (Y)", placeholder="2, 4, 6...", lines=3)
            
            gr.Markdown("### Ajustes de ejecución")
            with gr.Group(elem_classes="as-execution-settings"):
                with gr.Row():
                    pop_size = gr.Slider(
                        10_000,
                        4_000_000,
                        value=250_000,
                        step=10_000,
                        label="Población",
                        info="Seguro para GPU de 4 GB.",
                    )
                    islands = gr.Slider(
                        1,
                        100,
                        value=20,
                        step=1,
                        label="Islas",
                        info="Subpoblaciones paralelas.",
                    )

                with gr.Row():
                    max_const = gr.Slider(
                        1,
                        50,
                        value=10,
                        step=1,
                        label="Constantes",
                        info="Máximo por fórmula.",
                    )
                    timeout = gr.Slider(
                        0,
                        300,
                        value=60,
                        step=5,
                        label="Timeout",
                        info="0 = sin límite.",
                    )
                    use_log = gr.Checkbox(
                        label="Usar log(Y)",
                        value=False,
                        info="Para datos exponenciales.",
                    )
            
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
                stop_btn = gr.Button("DETENER", variant="stop", interactive=False)
                start_btn = gr.Button("INICIAR EVOLUCIÓN", variant="primary")

        with gr.Column(scale=1, min_width=360, elem_classes="as-main-panel"):
            gr.Markdown("## Dashboard de Evolución")
            current_best_formula = gr.HTML(label="Mejor Fórmula Actual", value=formula_card("", "Mejor fórmula actual"))
            stats_display = gr.HTML(label="Estadísticas", value=metric_grid([("RMSE", "—"), ("Generación", "—"), ("Tiempo", "—"), ("Islas", "—")]))
            live_plot = gr.Plot(label="Ajuste en Tiempo Real")
            status_html = gr.HTML(value='<div style="padding: 10px; background: #0f0f23; border-radius: 8px; border-left: 3px solid #64748b; color: #64748b;">Esperando inicio...</div>')
    
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
    btn_mv_lin.click(lambda: generate_example("mv_lineal"), outputs=[x_input, y_input])
    btn_mv_quad.click(lambda: generate_example("mv_cuadratico"), outputs=[x_input, y_input])
    btn_mv_trig.click(lambda: generate_example("mv_trig"), outputs=[x_input, y_input])
    btn_mv_exp.click(lambda: generate_example("mv_exp"), outputs=[x_input, y_input])
    btn_mv_lin3d.click(lambda: generate_example("mv_lineal_3d"), outputs=[x_input, y_input])
    btn_mv_quad3d.click(lambda: generate_example("mv_cuadratico_3d"), outputs=[x_input, y_input])
    btn_mv_4d.click(lambda: generate_example("mv_4d"), outputs=[x_input, y_input])

    def stop_evolution(run_state):
        if run_state is None:
            run_state = LiveRunState()
        run_state.request_stop()
        return status_panel("Deteniendo motor...", "warning"), run_state, gr.update(interactive=True), gr.update(interactive=False)

    def run_live_gpu_evolution_with_state(*args):
        *ui_args, run_state, verbose_enabled = args
        if run_state is None:
            run_state = LiveRunState()
        for status, best, stats, plot in run_live_gpu_evolution(*ui_args, run_state=run_state, verbose=verbose_enabled):
            finished = ("Finalizado" in status) or ("Error" in status) or ("No se pudo" in status)
            yield (
                status,
                best,
                stats,
                plot,
                run_state,
                gr.update(interactive=finished),
                gr.update(interactive=not finished),
            )

    start_btn.click(
        run_live_gpu_evolution_with_state,
        inputs=[x_input, y_input, pop_size, islands, max_const, timeout, use_log,
                op_sin, op_cos, op_tan, op_log, op_exp, op_sqrt, op_abs, op_floor, op_ceil, op_sign,
                op_gamma, op_lgamma, op_fact, op_mod, op_pow, op_asin, op_acos, op_atan, live_state, verbose_state or gr.State(False)],
        outputs=[status_html, current_best_formula, stats_display, live_plot, live_state, start_btn, stop_btn],
        concurrency_limit=1
    )
    stop_btn.click(stop_evolution, inputs=[live_state], outputs=[status_html, live_state, start_btn, stop_btn], queue=False)

def run_live_gpu_evolution(x_str, y_str, pop_size, n_islands, max_constants, timeout_sec, use_log_transform,
                           op_sin, op_cos, op_tan, op_log, op_exp, op_sqrt, op_abs, op_floor, op_ceil, op_sign,
                           op_gamma, op_lgamma, op_fact, op_mod, op_pow, op_asin, op_acos, op_atan,
                           run_state=None, verbose=False):
    """Generator that runs the GPU engine and yields updates to the UI."""
    global LIVE_ENGINE
    configure_logging(verbose)
    if run_state is None:
        run_state = LiveRunState()
    
    # 1. Parse Data
    x, y, error = parse_data(x_str, y_str)
    if error:
        yield status_panel(error, "error"), formula_card("", "Mejor fórmula actual"), metric_grid([("RMSE", "—"), ("Generación", "—"), ("Tiempo", "—"), ("Islas", "—")]), None
        return

    device = get_device()
    num_vars = 1 if x.ndim == 1 else x.shape[1]
    
    # 2. Clean up previous engine if any (VRAM safety)
    if run_state.is_running():
        run_state.request_stop()
    if run_state.engine:
        del run_state.engine
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    run_state.reset()

    config_names = [
        "USE_LOG_TRANSFORMATION", "LOSS_FUNCTION", "USE_NEURAL_FLASH", "USE_ALPHA_MCTS",
        "POP_SIZE", "NUM_ISLANDS", "GENERATIONS", "USE_OP_SIN", "USE_OP_COS", "USE_OP_TAN",
        "USE_OP_ASIN", "USE_OP_ACOS", "USE_OP_ATAN", "USE_OP_LOG", "USE_OP_EXP", "USE_OP_SQRT",
        "USE_OP_ABS", "USE_OP_FLOOR", "USE_OP_CEIL", "USE_OP_SIGN", "USE_OP_GAMMA",
        "USE_OP_LGAMMA", "USE_OP_FACT", "USE_OP_MOD", "USE_OP_POW", "USE_SNIPER"
    ]
    previous_config = {name: getattr(GpuGlobals, name, None) for name in config_names}
    
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
    
    # Disable Sniper (user requested pure GP evolution only)
    GpuGlobals.USE_SNIPER = False
    
    # 3. Initialize Engine (NOW it will read the correct operator flags)
    yield status_panel(f"Inicializando motor GPU ({int(pop_size):,} individuos)...", "info"), formula_card("", "Mejor fórmula actual"), metric_grid([("RMSE", "—"), ("Generación", "—"), ("Tiempo", "0.0s"), ("Islas", int(n_islands))]), None
    
    try:
        run_state.engine = ENGINE_CLS(
            device=device,
            pop_size=int(pop_size),
            n_islands=int(n_islands),
            num_variables=num_vars,
            max_constants=int(max_constants),
            model=None # Explicitly no model for Pure GPU mode
        )
    except Exception as e:
        for name, value in previous_config.items():
            setattr(GpuGlobals, name, value)
        logger.error("No se pudo inicializar el motor GPU: %s", format_exception(e))
        yield status_panel(f"No se pudo inicializar el motor GPU: {e}", "error"), formula_card("", "Mejor fórmula actual"), metric_grid([("RMSE", "—"), ("Generación", "—"), ("Tiempo", "—"), ("Islas", int(n_islands))]), None
        return
    
    LIVE_ENGINE = run_state.engine
    run_state.engine.stop_flag = False
    
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
    
    def wrapped_callback(gen, best_rmse, best_rpn, best_consts, is_new_best, island_idx):
        if is_new_best or gen % 10 == 0:
            # Simplify before displaying
            try:
                rpn_batch = best_rpn.unsqueeze(0)  # [1, L]
                consts_batch = best_consts.unsqueeze(0)  # [1, K]
                simp_pop, simp_consts, _ = run_state.engine.gpu_simplifier.simplify_batch(rpn_batch, consts_batch)
                best_rpn_simp = simp_pop[0]
                best_consts_simp = simp_consts[0] if simp_consts is not None else best_consts
            except Exception:
                best_rpn_simp = best_rpn
                best_consts_simp = best_consts
            
            # Initial assumption: Simplified is best
            final_rpn = best_rpn_simp
            final_consts = best_consts_simp
            formula = run_state.engine.rpn_to_infix(final_rpn, final_consts)
            
            # FALLBACK: If simplified is invalid, revert to original
            if formula == "Invalid":
                logger.debug("Simplificación inválida; revirtiendo al RPN original.")
                
                # Debug Tokens
                try:
                    toks = [run_state.engine.grammar.id_to_token.get(t.item(), str(t.item())) for t in best_rpn_simp if t.item() in run_state.engine.grammar.id_to_token]
                    logger.debug("Tokens simplificados inválidos: %s", toks)
                except Exception:
                    logger.debug("No se pudieron decodificar tokens inválidos.", exc_info=True)

                # REVERT to Original
                final_rpn = best_rpn
                final_consts = best_consts
                formula = run_state.engine.rpn_to_infix(final_rpn, final_consts)
                
                # If still invalid, print original tokens
                if formula == "Invalid":
                    logger.debug("El RPN original también es inválido.")
                    try:
                        toks_orig = [run_state.engine.grammar.id_to_token.get(t.item(), str(t.item())) for t in best_rpn if t.item() in run_state.engine.grammar.id_to_token]
                        logger.debug("Tokens originales inválidos: %s", toks_orig)
                    except Exception:
                        logger.debug("No se pudieron decodificar tokens originales.", exc_info=True)
            
            run_state.updates.put({
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
        
    def engine_target():
        try:
            run_state.engine.run(x_input_tensor, y, None, timeout_sec if timeout_sec > 0 else None, wrapped_callback)
        except Exception as e:
            logger.error("Error durante evolución GPU: %s", format_exception(e))
            run_state.updates.put({"error": str(e)})

    run_state.thread = threading.Thread(target=engine_target)
    run_state.thread.start()
    
    start_time = time.time()
    last_gen = 0
    start_time = time.time()
    last_gen = 0
    last_fig = None
    last_best_html = ""
    last_stats_html = ""
    
    while run_state.thread.is_alive() or not run_state.updates.empty():
        try:
            # Poll for updates
            data = run_state.updates.get(timeout=0.1)
            if "error" in data:
                last_best_html = last_best_html or formula_card("", "Mejor fórmula actual")
                last_stats_html = last_stats_html or metric_grid([("RMSE", "—"), ("Generación", "—"), ("Tiempo", f"{time.time() - start_time:.1f}s"), ("Islas", int(n_islands))])
                yield status_panel(f"Error durante evolución GPU: {data['error']}", "error"), last_best_html, last_stats_html, last_fig
                break
            
            gen = data["gen"]
            rmse = data["rmse"]
            formula = data["formula"]
            
            elapsed = time.time() - start_time
            speed = (gen * pop_size) / elapsed if elapsed > 0 else 0
            
            status = status_panel(f"Ejecutando · Gen {gen:,} · {speed:,.0f} evals/s", "info")
            best_html = formula_card(formula, "Mejor fórmula actual")
            stats_html = metric_grid([
                ("RMSE", f"{rmse:.6e}"),
                ("Generación", f"{gen:,}"),
                ("Tiempo", f"{elapsed:.1f}s"),
                ("Islas", int(n_islands)),
            ])
            
            # Create plot (only every few updates to save CPU)
            fig = None
            if data["new_best"] or gen % 100 == 0:
                try:
                    # Evaluate using GPU engine directly for 100% accuracy
                    rpn_tensor = data["rpn"].unsqueeze(0).to(run_state.engine.device)
                    consts_tensor = data["consts"].unsqueeze(0).to(run_state.engine.device, dtype=run_state.engine.dtype)
                    
                    # Prepare x properly for engine
                    if num_vars == 1:
                        x_t = torch.tensor(x if x.ndim == 1 else x.flatten(), device=run_state.engine.device, dtype=run_state.engine.dtype).unsqueeze(0) # [1, N]
                    else:
                        x_t = torch.tensor(x.T if x.shape[0] != num_vars else x, device=run_state.engine.device, dtype=run_state.engine.dtype)
                    
                    preds, sp, err = run_state.engine.evaluator.vm.eval(rpn_tensor, x_t, consts_tensor)
                    is_valid = (sp == 1) & (~err)
                    
                    y_pred = preds.squeeze().cpu().numpy()
                    valid_mask = is_valid.squeeze().cpu().numpy()
                    y_pred = np.where(valid_mask, y_pred, np.nan)
                    
                    # Inverse transform if needed
                    if use_log_transform: 
                        y_pred = np.exp(y_pred)

                    y_pred = _fill_live_plot_predictions_with_formula(x, y_pred, formula)
                    fig = create_fit_plot(x, y, y_pred, formula)
                    last_fig = fig
                except Exception as e:
                    logger.debug("No se pudo refrescar el plot live: %s", format_exception(e), exc_info=True)
            
            last_best_html = best_html
            last_stats_html = stats_html
            run_state.last_output = (status, best_html, stats_html, last_fig)
            yield status, best_html, stats_html, last_fig
            
        except queue.Empty:
            if not run_state.thread.is_alive() or run_state.stop_event.is_set():
                # Signal the engine to stop if user pressed the button
                if run_state.stop_event.is_set():
                    run_state.request_stop()
                break
            continue

    elapsed = time.time() - start_time
    for name, value in previous_config.items():
        setattr(GpuGlobals, name, value)
    yield (
        status_panel(f"Finalizado en {elapsed:.2f}s", "success"),
        last_best_html, # Persist last best
        last_stats_html, # Persist last stats
        last_fig  # Persist final plot
    )
