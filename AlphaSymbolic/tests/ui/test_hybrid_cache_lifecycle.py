from AlphaSymbolic.search import hybrid_search


def test_gpu_engine_cache_is_explicitly_clearable(monkeypatch):
    released = []

    class CachedEngine:
        def __del__(self):
            released.append(True)

    hybrid_search._GPU_ENGINE_CACHE.clear()
    hybrid_search._GPU_ENGINE_CACHE[("test",)] = CachedEngine()
    monkeypatch.setattr(hybrid_search.torch.cuda, "is_available", lambda: False)

    hybrid_search.clear_gpu_engine_cache()

    assert not hybrid_search._GPU_ENGINE_CACHE
    assert released


def test_gpu_engine_cache_is_bounded_to_one_entry():
    assert hybrid_search._GPU_ENGINE_CACHE_MAX == 1
