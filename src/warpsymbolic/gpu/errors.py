"""Errors raised by the production GPU execution path."""


class GpuUnavailableError(RuntimeError):
    """Raised when a production operation requires CUDA but it is unavailable."""
