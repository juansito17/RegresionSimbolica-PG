import gradio as gr
from AlphaSymbolic.utils.benchmark_comparison import run_comparison_benchmark
from AlphaSymbolic.ui.app_core import get_model
from AlphaSymbolic.ui.formatting import escape_html, metric_grid, status_panel
from AlphaSymbolic.ui.logging_utils import format_exception, get_logger

logger = get_logger("UI.BENCH")

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
                return status_panel("Modelo no cargado.", "error"), []
            
            if not selected_methods:
                return status_panel("Selecciona al menos un método.", "warning"), []
                
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
                logger.error("Error en benchmark: %s", format_exception(e))
                return status_panel(f"Error en benchmark: {e}", "error"), []
            
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
            html_content = '<div class="as-benchmark-summary">'
            
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
                
                html_content += (
                    f'<section class="as-panel as-benchmark-card" style="border-color:{border_color};background:{bg_color};">'
                    f'<div class="as-eyebrow">{escape_html(method.upper())} {escape_html(trophy)}</div>'
                    + metric_grid(
                        [
                            ("Resueltos", f"{stats['solved']} / {stats['total']}"),
                            ("Nota", f"{stats['score']:.1f}%"),
                            ("Tiempo avg", f"{stats['avg_time']:.2f}s"),
                        ]
                    )
                    + "</section>"
                )
            html_content += "</div>"
            
            return html_content, rows
            
        run_btn.click(run_bench, inputs=[methods_chk, timeout_slider], outputs=[summary_html, results_df])
