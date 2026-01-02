"""
AlphaSymbolic - Gradio Web Interface
With GPU/CPU toggle and search method selection.
"""
import gradio as gr
import torch

from ui.app_core import load_model, get_device, get_device_info, set_device
from ui.app_training import train_basic, train_curriculum, train_self_play
from ui.app_search import solve_formula, generate_example


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
                                choices=["Beam Search", "MCTS"],
                                value="Beam Search",
                                label="Metodo de Busqueda"
                            )
                        
                        beam_slider = gr.Slider(5, 50, value=15, step=5, label="Beam Width / Simulaciones")
                        
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
                            if os.path.exists("alpha_symbolic_model.pth"):
                                os.remove("alpha_symbolic_model.pth")
                                return '<div style="color: #ff6b6b; padding: 5px;">✅ Modelo eliminado. Reinicia la app para usar pesos nuevos.</div>'
                            return '<div style="color: #888; padding: 5px;">No hay modelo guardado.</div>'
                        
                        delete_model_btn.click(delete_model_action, outputs=[delete_status])
                
                with gr.Tabs():
                    # Basic
                    with gr.Tab("Basico"):
                        gr.HTML('<p style="color: #888;">Entrenamiento rapido con datos sinteticos</p>')
                        with gr.Row():
                            with gr.Column():
                                epochs_basic = gr.Slider(10, 500, value=100, step=10, label="Epocas")
                                batch_basic = gr.Slider(16, 128, value=32, step=16, label="Batch Size")
                                train_basic_btn = gr.Button("Entrenar Basico", variant="primary")
                            with gr.Column():
                                result_basic = gr.HTML()
                                plot_basic = gr.Plot()
                        train_basic_btn.click(train_basic, [epochs_basic, batch_basic], [result_basic, plot_basic])
                    
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
                                train_curriculum_btn = gr.Button("Entrenar Curriculum", variant="primary")
                            with gr.Column():
                                result_curriculum = gr.HTML()
                                plot_curriculum = gr.Plot()
                        train_curriculum_btn.click(train_curriculum, [epochs_curriculum, batch_curriculum], [result_curriculum, plot_curriculum])
                    
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
                                iterations_sp = gr.Slider(10, 200, value=30, step=10, label="Iteraciones")
                                problems_sp = gr.Slider(5, 50, value=10, step=5, label="Problemas/Iter")
                                train_sp_btn = gr.Button("Iniciar Self-Play", variant="primary")
                            with gr.Column():
                                result_sp = gr.HTML()
                                plot_sp = gr.Plot()
                        train_sp_btn.click(train_self_play, [iterations_sp, problems_sp], [result_sp, plot_sp])
            
            # TAB 3: Info
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


if __name__ == "__main__":
    print("Iniciando AlphaSymbolic...")
    status, device_info = load_model()
    print(f"   {status} | {device_info}")
    print("Abriendo navegador...")
    
    app = create_app()
    app.launch(share=False, inbrowser=True)
