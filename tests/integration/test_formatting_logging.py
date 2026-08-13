import logging

from AlphaSymbolic.ui.formatting import formula_card, prediction_table, status_panel
from AlphaSymbolic.ui.logging_utils import ENV_VERBOSE, configure_logging, is_verbose_enabled, set_verbose


def test_formula_card_escapes_html():
    html = formula_card("<script>x</script>", "F")

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_prediction_table_marks_delta_tones():
    html = prediction_table([("1", "1.0", "1.0", "0.0"), ("2", "5.0", "1.0", "4.0")])

    assert "as-delta-good" in html
    assert "as-delta-bad" in html


def test_status_panel_escapes_message():
    html = status_panel("<b>boom</b>", "error")

    assert "<b>boom</b>" not in html
    assert "&lt;b&gt;boom&lt;/b&gt;" in html


def test_verbose_logging_sets_env(monkeypatch):
    monkeypatch.delenv(ENV_VERBOSE, raising=False)

    logger = configure_logging(True)

    assert is_verbose_enabled()
    assert logger.level == logging.DEBUG
    assert set_verbose(False) == "Modo verbose desactivado"
    assert not is_verbose_enabled()
