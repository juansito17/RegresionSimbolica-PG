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
/* AlphaSymbolic Custom CSS */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

body {
    background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f172a 100%) !important;
}

h1.logo-text {
    font-family: 'Orbitron', sans-serif;
    background: linear-gradient(to right, #22d3ee, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 20px rgba(34, 211, 238, 0.5);
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 1rem 2rem !important;
}

/* Glassmorphism Cards */
.glass-panel {
    background: rgba(30, 41, 59, 0.7) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    border-radius: 1rem;
}

/* Custom Tabs */
.tabs.svelte-1ogx8lh > ul {
    border-bottom: 2px solid #334155 !important;
}
.tab-nav.svelte-1ogx8lh {
    border: none !important;
    color: #94a3b8 !important;
}
.selected.svelte-1ogx8lh {
    color: #22d3ee !important;
    border-bottom: 2px solid #22d3ee !important;
    font-weight: bold;
}

/* Buttons */
.primary-btn {
    background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%) !important;
    border: none !important;
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.4) !important;
    transition: all 0.3s ease;
}
.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 25px rgba(6, 182, 212, 0.6) !important;
}

/* Table Styling */
table {
    border-radius: 8px;
    overflow: hidden;
}
thead {
    background: #1e293b;
}
th {
    color: #22d3ee !important;
    font-family: 'Orbitron', sans-serif;
    text-transform: uppercase;
    font-size: 0.8rem;
}
tr:nth-child(even) {
    background: rgba(255,255,255,0.02);
}

/* Plotly Dark Fix */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}
"""
