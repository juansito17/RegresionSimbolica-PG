"""
AlphaSymbolic - Gradio Web Interface (Modular Version)
Main entry point that uses modular components.
"""
import gradio as gr
import torch

from app_core import load_model, get_device
from app_training import train_basic, train_curriculum, train_self_play
from app_search import solve_formula, generate_example


# Custom CSS
CUSTOM_CSS = """
.gradio-container {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%) !important;
}
.gr-button-primary {
    background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%) !important;
    border: none !important;
}
.gr-button-secondary {
    background: #16213e !important;
    border: 1px solid #00d4ff !important;
    color: #00d4ff !important;
}
"""


def create_app():
    """Create the Gradio app."""
    
    with gr.Blocks(title="AlphaSymbolic", theme=gr.themes.Base(), css=CUSTOM_CSS) as demo:
        
        # Header with device info
        device = get_device()
        device_color = "#4ade80" if device.type == "cuda" else "#fbbf24" if device.type == "mps" else "#888"
        device_name = device.type.upper()
        if device.type == "cuda":
            device_name += f" ({torch.cuda.get_device_name(0)})"
        
        gr.HTML(f"""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #00d4ff22, transparent, #ff6b6b22); border-radius: 15px; margin-bottom: 20px;">
            <h1 style="color: #00d4ff; font-size: 42px; margin: 0;">🧠 AlphaSymbolic</h1>
            <p style="color: #888; font-size: 18px; margin: 5px 0;">Deep Reinforcement Learning para Regresión Simbólica</p>
            <p style="color: {device_color}; font-size: 14px; margin: 5px 0;">🖥️ Dispositivo: {device_name}</p>
        </div>
        """)
        
        with gr.Tabs():
            # TAB 1: Search
            with gr.Tab("🔍 Buscar Fórmula"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<h3 style="color: #00d4ff;">📊 Datos de Entrada</h3>')
                        x_input = gr.Textbox(label="Valores X", placeholder="1, 2, 3, 4, 5...", lines=2)
                        y_input = gr.Textbox(label="Valores Y", placeholder="5, 7, 9, 11, 13...", lines=2)
                        beam_slider = gr.Slider(5, 50, value=15, step=5, label="🎯 Beam Width")
                        
                        solve_btn = gr.Button("🔍 Buscar Fórmula", variant="primary", size="lg")
                        
                        with gr.Row():
                            gr.Button("📈 Lineal", size="sm").click(lambda: generate_example("lineal"), outputs=[x_input, y_input])
                            gr.Button("📊 Cuadrático", size="sm").click(lambda: generate_example("cuadratico"), outputs=[x_input, y_input])
                            gr.Button("🌊 Seno", size="sm").click(lambda: generate_example("trig"), outputs=[x_input, y_input])
                            gr.Button("📈 Exp", size="sm").click(lambda: generate_example("exp"), outputs=[x_input, y_input])
                    
                    with gr.Column(scale=2):
                        result_html = gr.HTML(label="Resultado")
                        plot_output = gr.Plot(label="Visualización")
                
                with gr.Row():
                    pred_html = gr.HTML(label="Predicciones")
                    alt_html = gr.HTML(label="Alternativas")
                
                raw_formula = gr.Textbox(visible=False)
                
                solve_btn.click(solve_formula, [x_input, y_input, beam_slider], 
                               [result_html, plot_output, pred_html, alt_html, raw_formula])
            
            # TAB 2: Training
            with gr.Tab("🎓 Entrenar Modelo"):
                gr.HTML(f"""
                <div style="background: #16213e; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h3 style="color: #ffd93d; margin: 0;">⚡ Centro de Entrenamiento</h3>
                    <p style="color: #888;">Dispositivo: <span style="color: {device_color};">{device_name}</span></p>
                </div>
                """)
                
                with gr.Tabs():
                    # Basic
                    with gr.Tab("📚 Básico"):
                        gr.HTML('<p style="color: #888;">Entrenamiento rápido con datos sintéticos</p>')
                        with gr.Row():
                            with gr.Column():
                                epochs_basic = gr.Slider(10, 500, value=50, step=10, label="📈 Épocas")
                                batch_basic = gr.Slider(16, 128, value=32, step=16, label="📦 Batch Size")
                                train_basic_btn = gr.Button("🚀 Entrenar Básico", variant="primary")
                            with gr.Column():
                                result_basic = gr.HTML()
                                plot_basic = gr.Plot()
                        train_basic_btn.click(train_basic, [epochs_basic, batch_basic], [result_basic, plot_basic])
                    
                    # Curriculum
                    with gr.Tab("📈 Curriculum"):
                        gr.HTML('''
                        <div style="background: #0f0f23; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                            <p style="color: #00d4ff; margin: 0;">🎓 <strong>Curriculum Learning</strong></p>
                            <p style="color: #888; margin: 5px 0 0 0;">Empieza con fórmulas simples y aumenta la dificultad.</p>
                        </div>
                        ''')
                        with gr.Row():
                            with gr.Column():
                                epochs_curriculum = gr.Slider(50, 2000, value=200, step=50, label="📈 Épocas")
                                batch_curriculum = gr.Slider(16, 128, value=64, step=16, label="📦 Batch Size")
                                train_curriculum_btn = gr.Button("🎓 Entrenar Curriculum", variant="primary")
                            with gr.Column():
                                result_curriculum = gr.HTML()
                                plot_curriculum = gr.Plot()
                        train_curriculum_btn.click(train_curriculum, [epochs_curriculum, batch_curriculum], [result_curriculum, plot_curriculum])
                    
                    # Self-Play
                    with gr.Tab("🔄 Self-Play"):
                        gr.HTML('''
                        <div style="background: #0f0f23; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 3px solid #ff6b6b;">
                            <p style="color: #ff6b6b; margin: 0;">🧠 <strong>AlphaZero Self-Play</strong></p>
                            <p style="color: #888; margin: 5px 0 0 0;">El modelo resuelve problemas y aprende de sus éxitos. ¡El más poderoso!</p>
                        </div>
                        ''')
                        with gr.Row():
                            with gr.Column():
                                iterations_sp = gr.Slider(10, 200, value=30, step=10, label="🔄 Iteraciones")
                                problems_sp = gr.Slider(5, 50, value=10, step=5, label="📊 Problemas/Iter")
                                train_sp_btn = gr.Button("🧠 Iniciar Self-Play", variant="primary")
                            with gr.Column():
                                result_sp = gr.HTML()
                                plot_sp = gr.Plot()
                        train_sp_btn.click(train_self_play, [iterations_sp, problems_sp], [result_sp, plot_sp])
            
            # TAB 3: Info
            with gr.Tab("ℹ️ Información"):
                gr.HTML(f"""
                <div style="background: #1a1a2e; padding: 30px; border-radius: 15px;">
                    <h2 style="color: #00d4ff;">🧠 ¿Qué es AlphaSymbolic?</h2>
                    <p style="color: #ccc; line-height: 1.8;">
                        Sistema de <strong style="color: #ff6b6b;">regresión simbólica</strong> 
                        basado en <strong style="color: #00d4ff;">Deep Learning</strong> y 
                        <strong style="color: #ffd93d;">Monte Carlo Tree Search</strong>.
                    </p>
                    
                    <h3 style="color: #00d4ff; margin-top: 30px;">🖥️ Dispositivo Actual</h3>
                    <p style="color: {device_color}; font-size: 20px;">{device_name}</p>
                    
                    <h3 style="color: #00d4ff; margin-top: 30px;">🔧 Operadores</h3>
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
            <p>Powered by PyTorch • SymPy • Scipy • Gradio</p>
        </div>
        """)
    
    return demo


if __name__ == "__main__":
    print("🚀 Iniciando AlphaSymbolic...")
    status, device_info = load_model()
    print(f"   {status} | {device_info}")
    print("🌐 Abriendo navegador...")
    
    app = create_app()
    app.launch(share=False, inbrowser=True)
