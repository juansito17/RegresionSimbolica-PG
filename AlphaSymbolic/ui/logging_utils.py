"""Logging helpers for AlphaSymbolic UI and scripts."""

from __future__ import annotations

import logging
import os
import traceback
from typing import Optional


ENV_VERBOSE = "ALPHASYMBOLIC_VERBOSE"
LOGGER_NAME = "AlphaSymbolic"


def is_verbose_enabled(value: Optional[bool] = None) -> bool:
    """Return whether verbose logging should be enabled."""
    if value is not None:
        return bool(value)
    raw = os.environ.get(ENV_VERBOSE, "")
    return raw.strip().lower() in {"1", "true", "yes", "on", "verbose"}


def configure_logging(verbose: Optional[bool] = None) -> logging.Logger:
    """Configure a shared logger and return it."""
    enabled = is_verbose_enabled(verbose)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if enabled else logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)

    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG if enabled else logging.INFO)

    os.environ[ENV_VERBOSE] = "1" if enabled else "0"
    return logger


def set_verbose(enabled: bool) -> str:
    """Set verbose mode from the UI and return a small status message."""
    configure_logging(enabled)
    state = "activado" if enabled else "desactivado"
    return f"Modo verbose {state}"


def set_verbose_with_value(enabled: bool) -> tuple[str, bool]:
    """Set verbose mode and return UI status plus the raw value for gr.State."""
    return set_verbose(enabled), bool(enabled)


def get_logger(component: str) -> logging.Logger:
    """Return a namespaced logger for a subsystem."""
    configure_logging()
    return logging.getLogger(f"{LOGGER_NAME}.{component}")


def format_exception(exc: BaseException, verbose: Optional[bool] = None) -> str:
    """Format exceptions with traceback only when verbose is active."""
    if is_verbose_enabled(verbose):
        return traceback.format_exc()
    return str(exc)
