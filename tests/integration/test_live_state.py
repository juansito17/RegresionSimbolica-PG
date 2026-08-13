import threading

from AlphaSymbolic.ui.live_state import LiveRunState


class DummyEngine:
    def __init__(self):
        self.stop_flag = False


def test_live_run_state_stop_sets_engine_flag():
    state = LiveRunState(engine=DummyEngine())

    state.request_stop()

    assert state.stop_event.is_set()
    assert state.engine.stop_flag is True


def test_live_run_state_reset_clears_last_output():
    state = LiveRunState()
    state.last_output = ("a", "b", "c", "d")
    state.request_stop()

    state.reset()

    assert not state.stop_event.is_set()
    assert state.last_output == ("", "", "", None)


def test_wait_for_stop_joins_cooperative_worker():
    state = LiveRunState(engine=DummyEngine())

    def worker():
        while not state.engine.stop_flag:
            state.stop_event.wait(0.01)

    state.thread = threading.Thread(target=worker)
    state.thread.start()

    assert state.wait_for_stop(1.0)
    assert not state.thread.is_alive()
