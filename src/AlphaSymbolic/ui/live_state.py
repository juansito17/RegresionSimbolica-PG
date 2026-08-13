"""Per-session state helpers for the live GPU tab."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Optional, Tuple


class LiveRunState:
    def __init__(
        self,
        engine: Any = None,
        thread: Optional[threading.Thread] = None,
        updates: Optional["queue.Queue[dict]"] = None,
        stop_event: Optional[threading.Event] = None,
        started_at: Optional[float] = None,
        last_output: Tuple[Any, Any, Any, Any] = ("", "", "", None),
    ):
        self.engine = engine
        self.thread = thread
        self.updates = updates or queue.Queue()
        self.stop_event = stop_event or threading.Event()
        self.started_at = started_at if started_at is not None else time.time()
        self.last_output = last_output

    def reset(self) -> None:
        self.stop_event.clear()
        self.updates = queue.Queue()
        self.started_at = time.time()
        self.last_output = ("", "", "", None)

    def request_stop(self) -> None:
        self.stop_event.set()
        if self.engine is not None:
            try:
                self.engine.stop_flag = True
            except Exception:
                pass

    def wait_for_stop(self, timeout: float = 5.0) -> bool:
        """Request cancellation and wait a bounded time for the worker."""
        self.request_stop()
        thread = self.thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        thread.join(max(0.0, float(timeout)))
        return not thread.is_alive()

    def clear_runtime(self) -> None:
        """Release per-run objects once a worker is no longer owned by the UI."""
        self.engine = None
        if self.thread is None or not self.thread.is_alive():
            self.thread = None

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def __getstate__(self):
        return {
            "engine": None,
            "thread": None,
            "updates": None,
            "stop_requested": self.stop_event.is_set(),
            "started_at": self.started_at,
            "last_output": self.last_output,
        }

    def __setstate__(self, state):
        self.engine = state.get("engine")
        self.thread = state.get("thread")
        self.updates = queue.Queue()
        self.stop_event = threading.Event()
        if state.get("stop_requested"):
            self.stop_event.set()
        self.started_at = state.get("started_at", time.time())
        self.last_output = state.get("last_output", ("", "", "", None))
