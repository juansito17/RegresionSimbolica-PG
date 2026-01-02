import gradio as gr
from utils.benchmark_runner import run_benchmark_suite
from ui.app_core import get_model, DEVICE

def get_benchmark_tab():
    with gr.Tab("🥇 Benchmark (IQ Test)"):
        gr.Markdown("### Evaluar Inteligencia del Modelo")
        gr.Markdown("Ejecuta una batería de **10 problemas estándar** para medir qué tanto ha aprendido el modelo.")
        
        run_btn = gr.Button("🚀 Iniciar Examen", variant="primary")
        
        progress_bar = gr.HTML("")
        
        with gr.Row():
            score_box = gr.Number(label="Puntuación (/100)", interactive=False)
            time_box = gr.Number(label="Tiempo Promedio (s)", interactive=False)
            
        results_df = gr.Dataframe(
            headers=["Nivel", "Nombre", "Formula Encontrada", "RMSE", "Estado", "Tiempo"],
            label="Resultados Detallados",
            interactive=False
        )
        
        def run_bench(progress=gr.Progress()):
            model_obj, device_obj = get_model()
            if not model_obj:
                return "<div>Error: Modelo no cargado</div>", 0, 0, []
            
            results, summary = run_benchmark_suite(
                model_obj, 
                device_obj, 
                progress_callback=lambda p, desc: progress(p, desc=desc)
            )
            
            # Format dataframe
            rows = []
            for r in results:
                rows.append([
                    r['level'],
                    r['name'],
                    r['found_formula'],
                    f"{r['rmse']:.5f}",
                    r['status'],
                    f"{r['time']:.2f}s"
                ])
            
            # Color score
            color = "green" if summary['score'] > 80 else "orange" if summary['score'] > 50 else "red"
            header = f"""
            <div style="background: #1e1e2f; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid {color};">
                <h1 style="color: {color}; margin: 0;">Nota Final: {summary['score']:.1f} / 100</h1>
                <p style="color: #ccc;">Problemas Resueltos: {summary['solved']} / {summary['total']}</p>
            </div>
            """
            
            return header, summary['score'], summary['avg_time'], rows
            
        run_btn.click(run_bench, outputs=[progress_bar, score_box, time_box, results_df])
