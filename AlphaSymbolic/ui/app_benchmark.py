import gradio as gr
from AlphaSymbolic.utils.benchmark_comparison import run_comparison_benchmark
from AlphaSymbolic.ui.app_core import get_model, DEVICE

def get_benchmark_tab():
    with gr.Tab("🥇 Benchmark (IQ Test)"):
        gr.Markdown("### Evaluar Inteligencia del Modelo (Comparativa)")
        gr.Markdown("Ejecuta una batería de **10 problemas estándar** comparando diferentes métodos de búsqueda.")
        
        with gr.Row():
            methods_chk = gr.CheckboxGroup(
                choices=["beam", "mcts", "hybrid"], 
                value=["hybrid"], 
                label="Métodos a Evaluar",
                info="Selecciona uno o más métodos para comparar."
            )
            timeout_slider = gr.Slider(
                minimum=5, 
                maximum=60, 
                value=30, 
                step=5, 
                label="Timeout GP (s)", 
                info="Tiempo máximo para Beta-GP por problema."
            )
        
        run_btn = gr.Button("🚀 Iniciar Benchmark Comparativo", variant="primary")
        
        progress_bar = gr.HTML("")
        
        # Area de resultados
        summary_html = gr.HTML("Resultados aparecerán aquí...")
        
        results_df = gr.Dataframe(
            headers=["Problema", "Nivel", "Método", "Formula", "RMSE", "Tiempo", "Estado"],
            label="Resultados Detallados",
            interactive=False
        )
        
        def run_bench(selected_methods, gp_timeout, progress=gr.Progress()):
            model_obj, device_obj = get_model()
            if not model_obj:
                return "<div>⚠️ Error: Modelo no cargado. Ve a la pestaña 'Config' y carga un modelo.</div>", None, []
            
            if not selected_methods:
                return "<div>⚠️ Error: Selecciona al menos un método.</div>", None, []
                
            progress(0, desc="Iniciando Benchmark...")
            
            # Run comparison
            try:
                result_data = run_comparison_benchmark(
                    model_obj, 
                    device_obj, 
                    methods=selected_methods,
                    gp_timeout=gp_timeout,
                    beam_width=50,
                    progress_callback=lambda p, desc: progress(p, desc=desc)
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"<div>❌ Error en Benchmark: {e}</div>", None, []
            
            results = result_data['results']
            summary_dict = result_data['summary']
            
            # Format dataframe
            rows = []
            for r in results:
                status_icon = "✅" if r['success'] else "❌"
                rmse_val = f"{r['rmse']:.5f}" if r['rmse'] < 1e6 else "> 10^6"
                rows.append([
                    r['problem_name'],
                    r['level'],
                    r['method'].upper(),
                    r['formula'],
                    rmse_val,
                    f"{r['time']:.2f}s",
                    status_icon
                ])
            
            # Generate HTML Summary
            html_content = "<div style='display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;'>"
            
            # Determine winner if multiple methods
            winner_method = None
            if len(selected_methods) > 1:
                winner_method = max(summary_dict.items(), key=lambda x: (x[1]['solved'], -x[1]['avg_rmse']))[0]
            
            for method, stats in summary_dict.items():
                is_winner = (method == winner_method)
                border_color = "#4CAF50" if is_winner else ("#FF9800" if stats['score'] > 50 else "#F44336")
                bg_color = "#1e1e2f"
                if is_winner:
                    bg_color = "#1b3a24" # Dark green tint for winner
                    
                trophy = "🏆 GANADOR" if is_winner else ""
                
                html_content += f"""
                <div style="background: {bg_color}; padding: 15px; border-radius: 10px; border: 2px solid {border_color}; min-width: 200px; text-align: center;">
                    <h2 style="color: {border_color}; margin: 0 0 10px 0;">{method.upper()} {trophy}</h2>
                    <div style="font-size: 24px; font-weight: bold; margin-bottom: 5px;">{stats['solved']} / {stats['total']}</div>
                    <div style="color: #ccc; font-size: 14px;">Resueltos</div>
                    <hr style="border-color: #444; margin: 10px 0;">
                    <div style="font-size: 14px;">Nota: <b>{stats['score']:.1f}%</b></div>
                    <div style="font-size: 14px;">Tiempo Avg: <b>{stats['avg_time']:.2f}s</b></div>
                </div>
                """
            html_content += "</div>"
            
            return html_content, rows
            
        run_btn.click(run_bench, inputs=[methods_chk, timeout_slider], outputs=[summary_html, results_df])
