import gradio as gr

def get_theme():
    """
    Create a custom AlphaSymbolic theme.
    Dark mode by default, Neon accents.
    """
    theme = gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="indigo",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#0f172a", # Slate 900
        body_background_fill_dark="#0f172a",
        block_background_fill="#1e293b", # Slate 800
        block_background_fill_dark="#1e293b",
        block_border_width="1px",
        block_border_color="#334155",
        input_background_fill="#334155", # Slate 700
        input_border_color="#475569",
        button_primary_background_fill="#06b6d4", # Cyan 500
        button_primary_background_fill_hover="#0891b2", # Cyan 600
        button_primary_text_color="white",
        slider_color="#06b6d4",
    )
    return theme

CUSTOM_CSS = """
body {
    background: #0b1120 !important;
    color: #e5e7eb !important;
}

.gradio-container {
    max-width: 1480px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 18px 24px !important;
}

.as-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 18px 22px;
    margin-bottom: 14px;
    background: #111827;
    border: 1px solid #263244;
    border-radius: 8px;
}

.as-logo {
    margin: 0;
    font-size: 1.75rem;
    line-height: 1.1;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: 0;
}

.as-subtitle {
    margin: 4px 0 0 0;
    color: #94a3b8;
    font-size: 0.9rem;
}

.as-device-pill,
.as-badge {
    display: inline-flex;
    align-items: center;
    width: fit-content;
    padding: 6px 10px;
    color: #a7f3d0;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.28);
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
}

.as-toolbar {
    align-items: center;
    margin-bottom: 8px;
}

.as-sidebar,
.as-main-panel {
    min-width: 0 !important;
}

.as-panel,
.as-formula-panel {
    background: #111827;
    border: 1px solid #263244;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 12px;
}

.as-eyebrow {
    color: #93c5fd;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.as-formula {
    display: block;
    width: 100%;
    color: #fecaca;
    font-size: clamp(0.95rem, 1.3vw, 1.25rem);
    line-height: 1.45;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
}

.as-status {
    padding: 10px 12px;
    border-left: 3px solid #64748b;
    border-radius: 6px;
    color: #e5e7eb;
    font-weight: 600;
    margin-bottom: 10px;
}

.as-metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px;
    margin: 10px 0;
}

.as-metric {
    background: #0f172a;
    border: 1px solid #253047;
    border-radius: 8px;
    padding: 10px;
}

.as-metric-label {
    display: block;
    color: #94a3b8;
    font-size: 0.78rem;
    margin-bottom: 4px;
}

.as-metric strong {
    color: #67e8f9;
    overflow-wrap: anywhere;
}

.primary-btn {
    background: #0891b2 !important;
    border: none !important;
    box-shadow: none !important;
}
.primary-btn:hover {
    background: #0e7490 !important;
}

.as-table {
    width: 100%;
    border-collapse: collapse;
    background: #111827;
    border-radius: 8px;
    overflow: hidden;
}
.as-table th,
.as-table td {
    padding: 8px 10px;
    border-bottom: 1px solid #263244;
    text-align: center;
}
.as-table th {
    color: #93c5fd !important;
    background: #0f172a;
}
.as-delta-good { color: #86efac; font-weight: 700; }
.as-delta-warn { color: #fde68a; font-weight: 700; }
.as-delta-bad { color: #fca5a5; font-weight: 700; }

.as-alt {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    padding: 8px 0;
    border-top: 1px solid #263244;
}
.as-alt code {
    overflow-wrap: anywhere;
    white-space: pre-wrap;
}
.as-alt span {
    color: #94a3b8;
    white-space: nowrap;
}
.as-alt-selected code {
    color: #fecaca;
}

.as-benchmark-summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
}

.as-execution-settings {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    margin: 0 0 18px 0 !important;
}

.as-execution-settings > div {
    background: transparent !important;
    border: 0 !important;
    padding: 0 !important;
}

.as-execution-settings .block {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 4px 10px 4px !important;
    min-width: 0 !important;
}

.as-execution-settings .wrap {
    background: transparent !important;
    border: 0 !important;
    padding: 0 !important;
}

.as-execution-settings .label-wrap {
    margin-bottom: 6px !important;
    align-items: center !important;
}

.as-execution-settings label span {
    background: #0891b2 !important;
    color: #ffffff !important;
    border-radius: 6px !important;
    padding: 5px 8px !important;
    font-weight: 800 !important;
}

.as-execution-settings input[type="number"] {
    max-width: 112px !important;
    background: #334155 !important;
    border-color: #475569 !important;
}

.as-execution-settings .info,
.as-execution-settings p {
    color: #9fc0db !important;
    font-size: 0.78rem !important;
    line-height: 1.35 !important;
    margin: 4px 0 10px 0 !important;
}

textarea,
input,
.wrap {
    min-width: 0 !important;
}

.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

@media (max-width: 820px) {
    .gradio-container {
        padding: 12px !important;
    }
    .as-header {
        align-items: flex-start;
        flex-direction: column;
    }
    .as-logo {
        font-size: 1.45rem;
    }
    .as-alt {
        flex-direction: column;
    }
}
"""
