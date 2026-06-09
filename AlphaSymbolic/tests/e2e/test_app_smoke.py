import os
import subprocess
import sys
import time

import pytest


pytestmark = pytest.mark.e2e


def test_app_imports_create_app():
    from AlphaSymbolic.app import create_app

    app = create_app(verbose=False)
    assert app is not None


def test_gradio_app_opens_in_browser_if_playwright_available():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    env = os.environ.copy()
    env["ALPHASYMBOLIC_VERBOSE"] = "0"
    proc = subprocess.Popen(
        [sys.executable, "AlphaSymbolic/app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        url = "http://127.0.0.1:7860"
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1366, "height": 768})
            for _ in range(60):
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=1000)
                    if "AlphaSymbolic" in page.content():
                        break
                except Exception:
                    time.sleep(0.5)
            assert "AlphaSymbolic" in page.content()
            page.get_by_text("Buscar Formula").first.wait_for(timeout=5000)
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
