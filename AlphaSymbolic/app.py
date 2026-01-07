"""
AlphaSymbolic - Gradio Web Interface
With GPU/CPU toggle and search method selection.
"""
import gradio as gr
import torch

from ui.app_core import load_model, get_device, get_device_info, set_device, get_training_errors, request_stop_training
from ui.app_training import train_basic, train_curriculum, train_self_play, train_supervised, train_hybrid_feedback_loop
from ui.app_search import solve_formula, generate_example
from ui.app_benchmark import get_benchmark_tab


def toggle_device(use_gpu):
    """Toggle between GPU and CPU."""
    device_info = set_device(use_gpu)
    color = "#4ade80" if "CUDA" in device_info else "#fbbf24" if "MPS" in device_info else "#888"
    return f'<div style="padding: 10px; background: #0f0f23; border-radius: 8px; border-left: 3px solid {color};"><span style="color: {color}; font-weight: bold;">{device_info}</span></div>'


def create_app():
    """Create the Gradio app."""
    
    with gr.Blocks(title="AlphaSymbolic") as demo:
        
        # Header
        device_info = get_device_info()
        device_color = "#4ade80" if "CUDA" in device_info else "#fbbf24" if "MPS" in device_info else "#888"
        
        gr.HTML(f"""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #00d4ff22, transparent, #ff6b6b22); border-radius: 15px; margin-bottom: 20px;">
            <h1 style="color: #00d4ff; font-size: 42px; margin: 0;">AlphaSymbolic</h1>
            <p style="color: #888; font-size: 18px; margin: 5px 0;">Deep Reinforcement Learning para Regresion Simbolica</p>
        </div>
        """)
        
        # System Controls
        with gr.Row():
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(choices=["lite", "pro"], value="lite", label="Arquitectura (Cerebro)", interactive=True)
            with gr.Column(scale=3):
                model_status = gr.Textbox(label="Estado del Modelo", value="Lite (Laptop Optimized) - Vocabulario Extendido", interactive=False)
        
        def on_model_change(preset):
            status, _ = load_model(preset_name=preset)
            return status

        model_selector.change(on_model_change, model_selector, model_status)
        
        with gr.Tabs():
            # TAB 1: Search
            with gr.Tab("Buscar Formula"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<h3 style="color: #00d4ff;">Datos de Entrada</h3>')
                        x_input = gr.Textbox(label="Valores X", placeholder="1, 2, 3, 4, 5...", lines=2)
                        y_input = gr.Textbox(label="Valores Y", placeholder="5, 7, 9, 11, 13...", lines=2)
                        
                        with gr.Row():
                            search_method = gr.Radio(
                                choices=["Beam Search", "MCTS", "Alpha-GP Hybrid"],
                                value="Alpha-GP Hybrid",
                                label="Metodo de Busqueda"
                            )
                        
                        beam_slider = gr.Slider(5, 500, value=50, step=5, label="Beam Width / Simulaciones")
                        
                        solve_btn = gr.Button("Buscar Formula", variant="primary", size="lg")
                        
                        with gr.Row():
                            gr.Button("Lineal", size="sm").click(lambda: generate_example("lineal"), outputs=[x_input, y_input])
                            gr.Button("Cuadratico", size="sm").click(lambda: generate_example("cuadratico"), outputs=[x_input, y_input])
                            gr.Button("Seno", size="sm").click(lambda: generate_example("trig"), outputs=[x_input, y_input])
                            gr.Button("Exponencial", size="sm").click(lambda: generate_example("exp"), outputs=[x_input, y_input])
                    
                    with gr.Column(scale=2):
                        result_html = gr.HTML(label="Resultado")
                        plot_output = gr.Plot(label="Visualizacion")
                
                with gr.Row():
                    pred_html = gr.HTML(label="Predicciones")
                    alt_html = gr.HTML(label="Alternativas")
                
                raw_formula = gr.Textbox(visible=False)
                
                solve_btn.click(solve_formula, [x_input, y_input, beam_slider, search_method], 
                               [result_html, plot_output, pred_html, alt_html, raw_formula])
            
            # TAB 2: Training
            with gr.Tab("Entrenar Modelo"):
                with gr.Row():
                    gr.HTML("""
                    <div style="background: #16213e; padding: 20px; border-radius: 10px; flex: 1;">
                        <h3 style="color: #ffd93d; margin: 0;">Centro de Entrenamiento</h3>
                    </div>
                    """)
                    with gr.Column():
                        use_gpu = gr.Checkbox(label="Usar GPU", value=torch.cuda.is_available())
                        device_display = gr.HTML(value=f'<div style="padding: 10px; background: #0f0f23; border-radius: 8px; border-left: 3px solid {device_color};"><span style="color: {device_color}; font-weight: bold;">{device_info}</span></div>')
                        use_gpu.change(toggle_device, [use_gpu], [device_display])
                    with gr.Column():
                        delete_model_btn = gr.Button("🗑️ Borrar Modelo", variant="secondary", size="sm")
                        delete_status = gr.HTML()
                        
                        def delete_model_action():
                            import os
                            from ui.app_core import CURRENT_PRESET
                            filename = f"alpha_symbolic_model_{CURRENT_PRESET}.pth"
                            if os.path.exists(filename):
                                os.remove(filename)
                                return f'<div style="color: #4ade80; padding: 5px;">✅ Modelo [{CURRENT_PRESET}] eliminado. Reinicia la app para usar pesos nuevos.</div>'
                            return f'<div style="color: #888; padding: 5px;">No hay modelo [{CURRENT_PRESET}] guardado.</div>'
                        
                        delete_model_btn.click(delete_model_action, outputs=[delete_status])
                        
                        stop_train_btn = gr.Button("⏹️ Detener Entrenamiento", variant="stop", size="sm")
                        stop_status = gr.HTML()
                        stop_train_btn.click(request_stop_training, outputs=[stop_status])
                
                with gr.Tabs():
                    # Basic
                    with gr.Tab("Basico"):
                        gr.HTML('<p style="color: #888;">Entrenamiento rapido con datos sinteticos</p>')
                        with gr.Row():
                            with gr.Column():
                                epochs_basic = gr.Slider(10, 500, value=100, step=10, label="Epocas")
                                batch_basic = gr.Slider(16, 128, value=32, step=16, label="Batch Size")
                                points_basic = gr.Slider(10, 100, value=20, step=10, label="Puntos por Formula")
                                train_basic_btn = gr.Button("Entrenar Basico", variant="primary")
                            with gr.Column():
                                result_basic = gr.HTML()
                                plot_basic = gr.Plot()
                        train_basic_btn.click(train_basic, [epochs_basic, batch_basic, points_basic], [result_basic, plot_basic])
                    
                    # Curriculum
                    with gr.Tab("Curriculum"):
                        gr.HTML('''
                        <div style="background: #0f0f23; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                            <p style="color: #00d4ff; margin: 0;"><strong>Curriculum Learning</strong></p>
                            <p style="color: #888; margin: 5px 0 0 0;">Empieza con formulas simples y aumenta la dificultad.</p>
                        </div>
                        ''')
                        with gr.Row():
                            with gr.Column():
                                epochs_curriculum = gr.Slider(50, 2000, value=200, step=50, label="Epocas")
                                batch_curriculum = gr.Slider(16, 128, value=64, step=16, label="Batch Size")
                                points_curriculum = gr.Slider(10, 100, value=20, step=10, label="Puntos por Formula")
                                train_curriculum_btn = gr.Button("Entrenar Curriculum", variant="primary")
                            with gr.Column():
                                result_curriculum = gr.HTML()
                                plot_curriculum = gr.Plot()
                        train_curriculum_btn.click(train_curriculum, [epochs_curriculum, batch_curriculum, points_curriculum], [result_curriculum, plot_curriculum])
                    
                    # Self-Play
                    with gr.Tab("Self-Play"):
                        gr.HTML('''
                        <div style="background: #0f0f23; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 3px solid #ff6b6b;">
                            <p style="color: #ff6b6b; margin: 0;"><strong>AlphaZero Self-Play</strong></p>
                            <p style="color: #888; margin: 5px 0 0 0;">El modelo resuelve problemas y aprende de sus exitos.</p>
                        </div>
                        ''')
                        with gr.Row():
                            with gr.Column():
                                iterations_sp = gr.Slider(10, 1000, value=100, step=10, label="Iteraciones")
                                problems_sp = gr.Slider(5, 200, value=10, step=5, label="Problemas/Iter")
                                points_sp = gr.Slider(10, 100, value=20, step=10, label="Puntos por Formula")
                                train_sp_btn = gr.Button("Iniciar Self-Play", variant="primary")
                            with gr.Column():
                                result_sp = gr.HTML()
                                plot_sp = gr.Plot()
                        train_sp_btn.click(train_self_play, [iterations_sp, problems_sp, points_sp], [result_sp, plot_sp])
                
                    # Feedback Loop (Teacher-Student)
                    with gr.Tab("Feedback Loop (Hybrid)"):
                        gr.HTML('''
                        <div style="background: #0f0f23; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 3px solid #f1c40f;">
                            <p style="color: #f1c40f; margin: 0;"><strong>Teacher-Student Feedback Loop</strong></p>
                            <p style="color: #888; margin: 5px 0 0 0;">El modelo (Estudiante) intenta resolver problemas. Si falla, el Alpha-GP (Maestro) interviene y añade la solución al dataset.</p>
                        </div>
                        ''')
                        with gr.Row():
                            with gr.Column():
                                iterations_fb = gr.Slider(5, 500, value=20, step=5, label="Ciclos")
                                problems_fb = gr.Slider(5, 50, value=10, step=5, label="Problemas Difíciles / Ciclo")
                                timeout_fb = gr.Slider(5, 30, value=10, step=5, label="Timeout Maestro (s)")
                                train_fb_btn = gr.Button("Iniciar Feedback Loop", variant="primary")
                            with gr.Column():
                                result_fb = gr.HTML()
                                plot_fb = gr.Plot()
                        train_fb_btn.click(train_hybrid_feedback_loop, [iterations_fb, problems_fb, timeout_fb], [result_fb, plot_fb])
                
                # --- PRE-TRAINING (Warmup) ---
                with gr.Accordion("🎓 Escuela Primaria (Pre-Entrenamiento)", open=False):
                    gr.Markdown("Entrenamiento masivo supervisado de alta velocidad para aprender sintaxis basica. **Recomendado al inicio.**")
                    with gr.Row():
                        with gr.Column():
                            epochs_pre = gr.Slider(100, 10000, value=2000, step=100, label="Iteraciones Rápidas")
                            train_pre_btn = gr.Button("Iniciar Pre-Entrenamiento", variant="primary")
                        with gr.Column():
                            result_pre = gr.HTML()
                            plot_pre = gr.Plot()
                    train_pre_btn.click(train_supervised, [epochs_pre], [result_pre, plot_pre])

                # --- HALL OF SHAME (Error Analysis) ---
                with gr.Accordion("🕵️‍♂️ Hall of Shame (Analisis de Errores)", open=False):
                    gr.Markdown("Aquí se muestran los problemas donde el modelo falló drásticamente hoy.")
                    error_table = gr.DataFrame(
                        headers=["Time", "Target Formula", "Predicted", "Loss", "Stage"],
                        datatype=["str", "str", "str", "number", "str"],
                        interactive=False
                    )
                    refresh_errors_btn = gr.Button("🔄 Actualizar Errores", size="sm")
                    
                    def update_errors():
                        errors = get_training_errors()
                        # Reverse to show newest first
                        data = [[
                            e['time'], e['target'], e['predicted'], round(e['loss'], 2), e['stage']
                        ] for e in reversed(errors)]
                        return data
                    
                    refresh_errors_btn.click(update_errors, outputs=[error_table])
            
            # TAB 4: Benchmark
            get_benchmark_tab()

            # TAB 5: Info
            with gr.Tab("Informacion"):
                device_info_current = get_device_info()
                device_color_current = "#4ade80" if "CUDA" in device_info_current else "#fbbf24" if "MPS" in device_info_current else "#888"
                
                gr.HTML(f"""
                <div style="background: #1a1a2e; padding: 30px; border-radius: 15px;">
                    <h2 style="color: #00d4ff;">Que es AlphaSymbolic?</h2>
                    <p style="color: #ccc; line-height: 1.8;">
                        Sistema de <strong style="color: #ff6b6b;">regresion simbolica</strong> 
                        basado en <strong style="color: #00d4ff;">Deep Learning</strong> y 
                        <strong style="color: #ffd93d;">Monte Carlo Tree Search</strong>.
                    </p>
                    
                    <h3 style="color: #00d4ff; margin-top: 30px;">Dispositivo Actual</h3>
                    <p style="color: {device_color_current}; font-size: 20px;">{device_info_current}</p>
                    
                    <h3 style="color: #00d4ff; margin-top: 30px;">Metodos de Busqueda</h3>
                    <ul style="color: #ccc;">
                        <li><strong>Beam Search:</strong> Explora multiples candidatos en paralelo (rapido)</li>
                        <li><strong>MCTS:</strong> Monte Carlo Tree Search (mas preciso, lento)</li>
                        <li><strong>Alpha-GP Hybrid:</strong> Fusiona Neural Search con Algoritmo Genetico GPU (Extremo)</li>
                    </ul>
                    
                    <h3 style="color: #00d4ff; margin-top: 30px;">Operadores</h3>
                    <div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0;">
                        <span style="background: #0f0f23; padding: 5px 15px; border-radius: 20px; color: #00d4ff;">+</span>
                        <span style="background: #0f0f23; padding: 5px 15px; border-radius: 20px; color: #00d4ff;">-</span>
                        <span style="background: #0f0f23; padding: 5px 15px; border-radius: 20px; color: #00d4ff;">*</span>
                        <span style="background: #0f0f23; padding: 5px 15px; border-radius: 20px; color: #00d4ff;">/</span>
                        <span style="background: #0f0f23; padding: 5px 15px; border-radius: 20px; color: #ff6b6b;">sin</span>
                        <span style="background: #0f0f23; padding: 5px 15px; border-radius: 20px; color: #ff6b6b;">cos</span>
                        <span style="background: #0f0f23; padding: 5px 15px; border-radius: 20px; color: #ffd93d;">exp</span>
                        <span style="background: #0f0f23; padding: 5px 15px; border-radius: 20px; color: #ffd93d;">log</span>
                        <span style="background: #0f0f23; padding: 5px 15px; border-radius: 20px; color: #4ade80;">pow</span>
                        <span style="background: #0f0f23; padding: 5px 15px; border-radius: 20px; color: #4ade80;">sqrt</span>
                    </div>
                </div>
                """)
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: #666; margin-top: 30px;">
            <p>Powered by PyTorch - SymPy - Scipy - Gradio</p>
        </div>
        """)
    
    return demo




# --- Global Initialization for Hot Reloading ---
# IMPORTANT: For Windows Multiprocessing, we must protect entry point.
# However, Gradio needs 'demo' to be available for 'gradio app.py'.
# The issue is 'gradio app.py' imports this file, and multiprocessing spawns new processes that import it again.

if __name__ == "__main__":
    # If run directly (python app.py)
    print("Iniciando AlphaSymbolic (Global Init - Direct Execution)...")
    status_init, device_info_init = load_model() 
    print(f"   {status_init} | {device_info_init}")
    demo = create_app()
    print("Abriendo navegador...")
    demo.launch(share=True, inbrowser=True)
else:
    # If imported by 'gradio app.py' or multiprocessing workers
    # We only want to load the model if it's the Main Process (Gradio Server)
    # But multiprocessing workers import this too.
    # We can try to detect if we are a worker or the server.
    
    # Simple fix for Gradio Reload:
    # define demo globally but lazy load model?
    # No, let's keep it simple.
    
    print("AlphaSymbolic Module Imported.")
    # Attempt to load model only if not in a worker process?
    # Actually, for 'gradio app.py', this 'else' block runs.
    # We need 'demo' to be defined here.
    
    try:
        status_init, device_info_init = load_model() 
        print(f"   {status_init} | {device_info_init}")
    except Exception:
        pass # Might fail in workers, that's fine

    demo = create_app()
