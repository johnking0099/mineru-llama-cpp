"""Covers long-lived-service memory stability and graceful shutdown
(Tier2 #6 from the technical-validation phase)."""

import gc
import os

from mineru_llama_cpp import Engine, SamplingParams

MODEL = "/Users/jinzhenj/.mineru/models/MinerU2.5-Pro-2605-1.2B-Q8_0.gguf"
MMPROJ = "/Users/jinzhenj/.mineru/models/mmproj-MinerU2.5-Pro-2605-1.2B-Q8_0.gguf"


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB->MB on macOS


def test_many_requests_no_memory_growth(engine):
    # n_predict is capped so every request does roughly the same amount of
    # work -- without a cap, generation length varies wildly request to
    # request (observed 8 to 2000+ tokens with an unbounded prompt), and
    # the resulting compute-buffer sizing swings dwarf any real leak signal.
    n = 50
    sp = SamplingParams(temperature=0.0, top_k=1, n_predict=32)
    rss_before = _rss_mb()
    for i in range(n):
        result = engine.generate([{"role": "user", "content": f"Say something {i}"}], sp)
        assert result.content
    rss_after = _rss_mb()
    growth = rss_after - rss_before
    assert growth < 100, f"RSS grew {growth:.1f} MB over {n} requests (threshold 100 MB)"


def test_close_is_idempotent_and_engine_is_reconstructable():
    eng = Engine(MODEL, MMPROJ, n_ctx_seq=4096)
    assert eng.generate([{"role": "user", "content": "hi"}]).content
    eng.close()
    gc.collect()

    eng.close()  # must not raise on a second call

    # A fresh Engine must still load and work after the first one closed --
    # proves the C++ backend was actually released, not just Python-detached.
    eng2 = Engine(MODEL, MMPROJ, n_ctx_seq=4096)
    assert eng2.generate([{"role": "user", "content": "hi"}]).content
    eng2.close()


def test_context_manager_closes_on_exit():
    with Engine(MODEL, MMPROJ, n_ctx_seq=4096) as eng:
        assert eng.generate([{"role": "user", "content": "hi"}]).content
    assert eng._closed
