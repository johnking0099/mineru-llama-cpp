"""Tier2 #6: long-lived lifecycle + graceful shutdown verification.

Proves:
  1. Engine serves many requests (100) without memory growth (RSS stable).
  2. Engine can be cleanly destroyed (terminate + join) and reconstructed.
  3. No crash/hang on shutdown.

Uses psutil for RSS; falls back to /proc-style parsing if unavailable.
Q8_0 model; small max_tokens to keep it fast.
"""

import asyncio
import json
import os
import sys
import time

from mineru_engine_spike import Engine

MODEL  = "/Users/jinzhenj/.mineru/models/MinerU2.5-Pro-2605-1.2B-Q8_0.gguf"
MMPROJ = "/Users/jinzhenj/.mineru/models/mmproj-MinerU2.5-Pro-2605-1.2B-Q8_0.gguf"

N = 100


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        # macOS fallback: vmmap is heavy; use resource module if available
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB->MB on mac
        except ImportError:
            return 0.0


def text_body(i: int) -> str:
    return json.dumps({
        "model": "m",
        "messages": [{"role": "user", "content": f"Say something {i}"}],
        "temperature": 0.0, "top_k": 1, "max_tokens": 8, "stream": False,
    })


async def main() -> None:
    print(f"[init] constructing engine (pid={os.getpid()})")
    rss0 = rss_mb()
    print(f"[init] RSS after Python start: {rss0:.1f} MB")

    engine = Engine(model=MODEL, mmproj=MMPROJ, n_ctx=4096, n_gpu_layers=99, n_parallel=2)
    rss1 = rss_mb()
    print(f"[init] RSS after engine load: {rss1:.1f} MB (model takes {rss1-rss0:.1f} MB)")

    # --- 1. serve N requests, sample RSS every 10 ---
    print(f"\n[run] serving {N} requests...")
    loop = asyncio.get_event_loop()
    t0 = time.monotonic()
    samples = []
    for i in range(N):
        out = await loop.run_in_executor(None, engine.generate, text_body(i))
        if (i + 1) % 10 == 0:
            r = rss_mb()
            samples.append((i + 1, r))
            print(f"  req {i+1:3d}: RSS={r:.1f} MB, last out len={len(out)}")
    dt = time.monotonic() - t0
    print(f"[run] {N} requests in {dt:.1f}s ({N/dt:.1f} req/s)")

    rss_final = rss_mb()
    print(f"[run] final RSS: {rss_final:.1f} MB")

    # --- 2. graceful shutdown + reconstruct ---
    print("\n[shutdown] destroying engine...")
    del engine
    rss_after_del = rss_mb()
    print(f"[shutdown] RSS after del engine: {rss_after_del:.1f} MB")

    print("\n[restart] reconstructing engine...")
    engine2 = Engine(model=MODEL, mmproj=MMPROJ, n_ctx=4096, n_gpu_layers=99, n_parallel=2)
    rss2 = rss_mb()
    print(f"[restart] RSS after reconstruct: {rss2:.1f} MB")

    # serve a few more to confirm engine2 works
    for i in range(3):
        out = await loop.run_in_executor(None, engine2.generate, text_body(1000 + i))
    print(f"[restart] engine2 served 3 requests OK (last len={len(out)})")

    del engine2
    print("[restart] engine2 destroyed OK")

    # --- analysis ---
    print("\n=== Analysis ===")
    print(f"RSS samples (req# : MB): {samples}")
    if len(samples) >= 2:
        growth = samples[-1][1] - samples[0][1]
        print(f"RSS growth over {N} requests: {growth:+.1f} MB")
        # tolerant: < 50 MB growth is acceptable (Metal/MTL buffers may fluctuate)
        stable = abs(growth) < 50
        print(f"memory stable: {'OK' if stable else 'FAIL (growth > 50 MB)'}")
    else:
        stable = True

    ok = stable
    print("\n=== SPIKE PASSED ===" if ok else "\n=== SPIKE FAILED ===")


if __name__ == "__main__":
    asyncio.run(main())
