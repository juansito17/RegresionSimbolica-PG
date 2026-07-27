import gradio as gr

from AlphaSymbolic.ui.formatting import escape_html, metric_grid, status_panel
from AlphaSymbolic.ui.logging_utils import format_exception, get_logger
from AlphaSymbolic.utils.benchmark_comparison import run_comparison_benchmark


logger = get_logger("UI.BENCH")


def get_benchmark_tab():
    with gr.Tab("🥇 Benchmark"):
        gr.Markdown("### Benchmark con holdout independiente")
        gr.Markdown(
            "Ejecuta **10 problemas de regresión simbólica**. GPU-GP y el "
            "baseline polinomial son implementaciones distintas; el baseline "
            "es una comprobación de cordura, no un competidor SOTA."
        )

        with gr.Row():
            methods_chk = gr.CheckboxGroup(
                choices=[
                    ("AlphaSymbolic GPU-GP", "gpu_gp"),
                    ("Polinomio grado 5 (baseline)", "polynomial"),
                ],
                value=["gpu_gp"],
                label="Métodos a evaluar",
                info="Cada etiqueta ejecuta una implementación real diferente.",
            )
            timeout_slider = gr.Slider(
                minimum=5,
                maximum=60,
                value=30,
                step=5,
                label="Timeout GPU-GP (s)",
                info="Tiempo máximo por problema para AlphaSymbolic.",
            )

        run_btn = gr.Button("🚀 Iniciar benchmark", variant="primary")
        summary_html = gr.HTML("Los resultados aparecerán aquí.")
        results_df = gr.Dataframe(
            headers=[
                "Problema",
                "Nivel",
                "Método",
                "Fórmula",
                "RMSE train",
                "RMSE test",
                "NRMSE test",
                "Tiempo",
                "Estado",
            ],
            label="Resultados detallados",
            interactive=False,
        )

        def run_bench(selected_methods, gp_timeout, progress=gr.Progress()):
            if not selected_methods:
                return status_panel("Selecciona al menos un método.", "warning"), []

            progress(0, desc="Iniciando benchmark...")
            try:
                result_data = run_comparison_benchmark(
                    methods=selected_methods,
                    gp_timeout=gp_timeout,
                    progress_callback=lambda value, desc: progress(value, desc=desc),
                )
            except Exception as exc:
                logger.error("Error en benchmark: %s", format_exception(exc))
                return status_panel(f"Error en benchmark: {exc}", "error"), []

            rows = []
            for result in result_data["results"]:
                valid_train = result["train_rmse"] < 1e100
                valid_test = result["test_rmse"] < 1e100
                valid_nrmse = result["test_nrmse"] < 1e100
                rows.append(
                    [
                        result["problem_name"],
                        result["level"],
                        result["method"].upper(),
                        result["formula"],
                        f"{result['train_rmse']:.5g}" if valid_train else "inválido",
                        f"{result['test_rmse']:.5g}" if valid_test else "inválido",
                        f"{result['test_nrmse']:.5g}" if valid_nrmse else "inválido",
                        f"{result['time']:.2f}s",
                        "✅" if result["success"] else "❌",
                    ]
                )

            html_content = '<div class="as-benchmark-summary">'
            for method, stats in result_data["summary"].items():
                border_color = "#4CAF50" if stats["score"] > 50 else "#FF9800"
                html_content += (
                    '<section class="as-panel as-benchmark-card" '
                    f'style="border-color:{border_color};background:#1e1e2f;">'
                    f'<div class="as-eyebrow">{escape_html(method.upper())}</div>'
                    + metric_grid(
                        [
                            ("Resueltos", f"{stats['solved']} / {stats['total']}"),
                            (
                                "Ejecuciones válidas",
                                f"{stats['valid_runs']} / {stats['total']}",
                            ),
                            ("Fallos", stats["failed"]),
                            ("Tiempo avg", f"{stats['avg_time']:.2f}s"),
                        ]
                    )
                    + "</section>"
                )
            html_content += "</div>"
            return html_content, rows

        run_btn.click(
            run_bench,
            inputs=[methods_chk, timeout_slider],
            outputs=[summary_html, results_df],
        )
