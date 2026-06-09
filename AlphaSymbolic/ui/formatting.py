"""Small HTML formatting helpers for the Gradio UI."""

from __future__ import annotations

import html
from typing import Iterable, Sequence, Tuple


def escape_html(value) -> str:
    return html.escape("" if value is None else str(value))


def status_panel(message: str, tone: str = "neutral") -> str:
    colors = {
        "neutral": ("#64748b", "rgba(100, 116, 139, 0.12)"),
        "info": ("#0891b2", "rgba(8, 145, 178, 0.12)"),
        "success": ("#16a34a", "rgba(22, 163, 74, 0.12)"),
        "warning": ("#ca8a04", "rgba(202, 138, 4, 0.14)"),
        "error": ("#dc2626", "rgba(220, 38, 38, 0.14)"),
    }
    color, bg = colors.get(tone, colors["neutral"])
    return (
        f'<div class="as-status as-status-{tone}" style="border-left-color:{color};background:{bg};">'
        f"{escape_html(message)}</div>"
    )


def formula_card(formula: str, title: str = "Fórmula") -> str:
    return (
        '<section class="as-panel as-formula-panel">'
        f'<div class="as-eyebrow">{escape_html(title)}</div>'
        f'<code class="as-formula">{escape_html(formula or "Sin fórmula todavía")}</code>'
        "</section>"
    )


def metric_grid(metrics: Sequence[Tuple[str, object]]) -> str:
    cells = []
    for label, value in metrics:
        cells.append(
            '<div class="as-metric">'
            f'<span class="as-metric-label">{escape_html(label)}</span>'
            f'<strong>{escape_html(value)}</strong>'
            "</div>"
        )
    return f'<div class="as-metric-grid">{"".join(cells)}</div>'


def prediction_table(rows: Iterable[Tuple[str, object, object, object]]) -> str:
    body = []
    for x_value, pred, real, delta in rows:
        try:
            delta_float = float(delta)
        except Exception:
            delta_float = float("inf")
        tone = "good" if delta_float < 0.1 else "warn" if delta_float < 1 else "bad"
        body.append(
            "<tr>"
            f"<td>{escape_html(x_value)}</td>"
            f"<td>{escape_html(pred)}</td>"
            f"<td>{escape_html(real)}</td>"
            f'<td class="as-delta-{tone}">{escape_html(delta)}</td>'
            "</tr>"
        )
    return (
        '<table class="as-table">'
        "<thead><tr><th>X</th><th>Predicción</th><th>Real</th><th>Delta</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def alternatives_list(items: Sequence[Tuple[str, object]], selected_index: int = 0) -> str:
    if not items:
        return status_panel("No hay alternativas para mostrar.", "neutral")
    chunks = ['<section class="as-panel"><div class="as-eyebrow">Alternativas</div>']
    for i, (formula, rmse) in enumerate(items):
        selected = " as-alt-selected" if i == selected_index else ""
        chunks.append(
            f'<div class="as-alt{selected}">'
            f'<code>{escape_html(formula)}</code>'
            f'<span>RMSE: {escape_html(rmse)}</span>'
            "</div>"
        )
    chunks.append("</section>")
    return "".join(chunks)
