"""
AlphaSymbolic - Gradio Web Interface
With GPU/CPU toggle and search method selection.
"""
import gradio as gr
import torch

from ui.app_core import load_model, get_device, get_device_info, set_device, get_training_errors, request_stop_training
from ui.app_training import train_basic, train_curriculum, train_self_play, train_supervised, train_hybrid_feedback_loop, train_from_memory
from ui.app_search import solve_formula, generate_example
from ui.app_benchmark import get_benchmark_tab
from ui.theme import get_theme, CUSTOM_CSS
import pandas as pd
import io


def toggle_device(use_gpu):
    """Toggle between GPU and CPU."""
    device_info = set_device(use_gpu)
    color = "#4ade80" if "CUDA" in device_info else "#fbbf24" if "MPS" in device_info else "#888"
    return f'<div style="padding: 10px; background: #0f0f23; border-radius: 8px; border-left: 3px solid {color};"><span style="color: {color}; font-weight: bold;">{device_info}</span></div>'


def create_app():
    """Create the Gradio app."""
    
    custom_theme = get_theme()
    
    with gr.Blocks(title="AlphaSymbolic") as demo:
        
        # Header
        device_info = get_device_info()
        device_color = "#4ade80" if "CUDA" in device_info else "#fbbf24" if "MPS" in device_info else "#888"
        

        def load_csv_data(file_obj):
            """Load CSV file to X/Y inputs."""
            if file_obj is None:
                return None, None
            
            try:
                # auto-detect separator
                try:
                    df = pd.read_csv(file_obj.name, sep=None, engine='python')
                except:
                    df = pd.read_csv(file_obj.name)
                
                if df.shape[1] < 2:
                    return None, "Error: El archivo debe tener al menos 2 columnas (X..., Y)"
                
                # Assume last column is Y, rest are X
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                
                # Format X string
                # If 1D: "1, 2, 3"
                # If 2D: "1 2; 3 4"
                if X.shape[1] == 1:
                    x_str = ", ".join(map(str, X.flatten()))
                else:
                    # Multi-line format
                    lines = [" ".join(map(str, row)) for row in X]
                    x_str = "\n".join(lines)
                
                y_str = ", ".join(map(str, y.flatten()))
                
                return x_str, y_str
            except Exception as e:
                return None, f"Error leyendo CSV: {str(e)}"

        # Header
        device_info = get_device_info()
        device_color = "#22d3ee" if "CUDA" in device_info else "#fbbf24"
        gpu_short = device_info.replace('NVIDIA GeForce ', '').replace(' Laptop GPU', '').replace('CUDA (', '').replace(')', '')
        
        gr.HTML(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 20px 30px; background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.9)); border-radius: 16px; margin-bottom: 15px; border: 1px solid rgba(6, 182, 212, 0.2); box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);">
            <div>
                <h1 style="margin: 0; font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, #06b6d4, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: 'Orbitron', sans-serif; letter-spacing: 2px;">
                    αSymbolic
                </h1>
                <p style="margin: 5px 0 0 0; color: #64748b; font-size: 0.9rem;">
                    Deep Reinforcement Learning & Symbolic Regression
                </p>
            </div>
            <div style="display: flex; align-items: center; gap: 15px;">
                <div style="text-align: right;">
                    <div style="background: {'rgba(34, 197, 94, 0.15)' if 'CUDA' in device_info else 'rgba(251, 191, 36, 0.15)'}; color: {'#22c55e' if 'CUDA' in device_info else '#fbbf24'}; padding: 8px 16px; border-radius: 25px; font-weight: 600; font-size: 0.85rem; border: 1px solid {'rgba(34, 197, 94, 0.3)' if 'CUDA' in device_info else 'rgba(251, 191, 36, 0.3)'};">
                        {'⚡ GPU' if 'CUDA' in device_info else '💻 CPU'} | {gpu_short}
                    </div>
                </div>
            </div>
        </div>
        """)
        
        # Model Selector - Compact inline
        with gr.Row():
            with gr.Column(scale=1):
                model_selector = gr.Radio(choices=["lite", "pro"], value="lite", label="Modelo", container=False)
            with gr.Column(scale=4):
                model_status = gr.HTML(value='<div style="padding: 8px 15px; background: rgba(34, 197, 94, 0.1); border-radius: 8px; color: #22c55e; font-size: 0.85rem; border: 1px solid rgba(34, 197, 94, 0.2);">✓ Lite Model (Optimized) - Vocabulary 2.0</div>')
        
        def on_model_change(preset):
            status, _ = load_model(preset_name=preset)
            return f'<div style="padding: 8px 15px; background: rgba(34, 197, 94, 0.1); border-radius: 8px; color: #22c55e; font-size: 0.85rem; border: 1px solid rgba(34, 197, 94, 0.2);">✓ {status}</div>'

        model_selector.change(on_model_change, model_selector, model_status)
        
        with gr.Tabs():
            # TAB 1: Search
            with gr.Tab("🔍 Buscar Formula"):
                with gr.Row():
                    # Column 1: Inputs + Config
                    with gr.Column(scale=1, min_width=400):
                        gr.Markdown("## Entrada")
                        x_input = gr.Textbox(label="Features (X)", placeholder="1, 2, 3...", lines=3)
                        y_input = gr.Textbox(label="Target (Y)", placeholder="2, 4, 6...", lines=3)
                        
                        with gr.Accordion("📁 Cargar desde CSV", open=False):
                            file_upload = gr.File(label="Seleccionar archivo", file_types=[".csv", ".txt"], file_count="single")
                            file_upload.change(load_csv_data, inputs=[file_upload], outputs=[x_input, y_input])
                        
                        with gr.Row():
                            gr.Button("Lineal", size="sm").click(lambda: generate_example("lineal"), outputs=[x_input, y_input])
                            gr.Button("Cuad", size="sm").click(lambda: generate_example("cuadratico"), outputs=[x_input, y_input])
                            gr.Button("Trig", size="sm").click(lambda: generate_example("trig"), outputs=[x_input, y_input])
                            gr.Button("Exp", size="sm").click(lambda: generate_example("exp"), outputs=[x_input, y_input])
                        
                        gr.Markdown("---")
                        search_method = gr.Radio(
                            choices=["Beam Search", "MCTS", "Alpha-GP Hybrid"],
                            value="Alpha-GP Hybrid",
                            label="Algoritmo de Búsqueda"
                        )
                        beam_slider = gr.Slider(5, 500, value=50, step=5, label="Intensidad (Beam Width)")
                        workers_slider = gr.Slider(1, 16, value=6, step=1, label="Workers (Paralelismo)", info="Procesos para el motor GP")
                        solve_btn = gr.Button("🚀 BUSCAR FÓRMULA", variant="primary", size="lg", elem_classes="primary-btn")
                        
                        with gr.Accordion("Tabla de Predicciones", open=False):
                            pred_html = gr.HTML(label="Predicciones")
                    
                    # Column 2: Results + Visualization
                    with gr.Column(scale=1, min_width=400):
                        gr.Markdown("## Resultados")
                        result_html = gr.HTML(label="Fórmula Encontrada")
                        plot_output = gr.Plot(label="Visualización del Ajuste")
                        alt_html = gr.HTML(label="Alternativas")
                
                raw_formula = gr.Textbox(visible=False)
                solve_btn.click(solve_formula, [x_input, y_input, beam_slider, search_method, workers_slider], 
                               [result_html, plot_output, pred_html, alt_html, raw_formula])
            
            # TAB 2: Training
            with gr.Tab("Entrenar Modelo"):
                # Training Control Panel - Compact Header
                gr.HTML(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 15px 20px; background: linear-gradient(135deg, rgba(6, 182, 212, 0.1), rgba(139, 92, 246, 0.1)); border-radius: 12px; margin-bottom: 10px; border: 1px solid rgba(6, 182, 212, 0.3);">
                    <div style="display: flex; align-items: center; gap: 15px;">
                        <span style="font-size: 1.5rem;"> </span>
                        <div>
                            <h3 style="margin: 0; color: #e2e8f0; font-size: 1.1rem;">Centro de Entrenamiento</h3>
                            <span style="color: #64748b; font-size: 0.8rem;">Gestiona el aprendizaje del modelo</span>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 20px;">
                        <div style="text-align: center;">
                            <span style="background: rgba(34, 197, 94, 0.2); color: #22c55e; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;">
                                {'🟢 GPU' if torch.cuda.is_available() else '🟡 CPU'}
                            </span>
                            <div style="color: {device_color}; font-size: 0.7rem; margin-top: 4px;">{device_info.replace('NVIDIA GeForce ', '').replace(' Laptop GPU', '')}</div>
                        </div>
                    </div>
                </div>
                """)
                
                with gr.Row():
                    use_gpu = gr.Checkbox(label="Usar GPU", value=torch.cuda.is_available(), visible=False)
                    device_display = gr.HTML(visible=False)
                    use_gpu.change(toggle_device, [use_gpu], [device_display])
                    stop_train_btn = gr.Button("Detener Todo", variant="stop", size="sm", scale=1)
                    delete_model_btn = gr.Button("Reset Pesos", variant="secondary", size="sm", scale=1)
                
                stop_status = gr.HTML(visible=False)
                delete_status = gr.HTML(visible=False)
                
                def delete_model_action():
                    import os
                    from ui.app_core import CURRENT_PRESET
                    filename = f"alpha_symbolic_model_{CURRENT_PRESET}.pth"
                    if os.path.exists(filename):
                        os.remove(filename)
                        return "Modelo Eliminado"
                    return "No existe modelo"
                
                # Global Training Config
                with gr.Row():
                    reset_state_btn = gr.Button("⚠️ Reset Estado", variant="secondary", size="sm")

                def reset_training_state():
                    from ui.app_training import TRAINING_STATUS
                    TRAINING_STATUS["running"] = False
                    return "Estado reseteado. Intenta entrenar de nuevo."

                reset_state_btn.click(reset_training_state, outputs=[stop_status])
                delete_model_btn.click(delete_model_action, outputs=[delete_status])
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
                                train_sp_btn = gr.Button("Iniciar Self-Play", variant="primary", elem_classes="primary-btn")
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
                        train_fb_btn.click(train_hybrid_feedback_loop, [iterations_fb, problems_fb, timeout_fb, workers_slider], [result_fb, plot_fb])
                
                # --- PRE-TRAINING (Warmup) ---
                with gr.Accordion("🎓 Escuela Primaria (Pre-Entrenamiento)", open=False):
                    gr.Markdown("Entrenamiento masivo supervisado de alta velocidad para aprender sintaxis basica. **Recomendado al inicio.**")
                    with gr.Row():
                        with gr.Column():
                            epochs_pre = gr.Slider(100, 10000, value=2000, step=100, label="Iteraciones Rápidas")
                            train_pre_btn = gr.Button("Iniciar Pre-Entrenamiento", variant="primary", elem_classes="primary-btn")
                        with gr.Column():
                            result_pre = gr.HTML()
                            plot_pre = gr.Plot()
                    train_pre_btn.click(train_supervised, [epochs_pre], [result_pre, plot_pre])
                
                # --- MEMORY TRAINING (Offline RL) ---
                with gr.Accordion("🧠 Entrenamiento de Memoria (Offline)", open=False):
                    gr.Markdown("Re-entrena el modelo usando las fórmulas descubiertas y guardadas en `learned_formulas.csv`. Ideal para consolidar conocimientos.")
                    with gr.Row():
                        with gr.Column():
                            epochs_mem = gr.Slider(10, 500, value=50, step=10, label="Epocas de Repaso")
                            train_mem_btn = gr.Button("Iniciar Entrenamiento de Memoria", variant="primary")
                        with gr.Column():
                            result_mem = gr.HTML()
                            plot_mem = gr.Plot()
                    train_mem_btn.click(train_from_memory, [epochs_mem], [result_mem, plot_mem])

                # --- HALL OF SHAME (Error Analysis) ---
                with gr.Accordion("Hall of Shame (Analisis de Errores)", open=False):
                    gr.Markdown("Aquí se muestran los problemas donde el modelo falló drásticamente hoy.")
                    error_table = gr.DataFrame(
                        headers=["Time", "Target Formula", "Predicted", "Loss", "Stage"],
                        datatype=["str", "str", "str", "number", "str"],
                        interactive=False
                    )
                    refresh_errors_btn = gr.Button("Actualizar Errores", size="sm")
                    
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
    from ui.theme import CUSTOM_CSS, get_theme
    demo.launch(share=True, inbrowser=True, theme=get_theme(), css=CUSTOM_CSS)
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
