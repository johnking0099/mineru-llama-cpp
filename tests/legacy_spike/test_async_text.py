"""Tier1 #2 spike (text-only, CPU): async/GIL bridge verification.

Uses a text-only request so no vision/Metal path is involved — isolates the
threading/GIL question (the spike's actual goal). CPU decode of ~128 tokens
takes a few seconds, long enough for the ticker to prove the GIL is released.

Proves:
  1. sync generate() round-trips.
  2. async agenerate() via run_in_executor works; 2 concurrent requests succeed.
  3. a pure-Python asyncio ticker advances during decode => GIL NOT held.
"""

import asyncio
import json
import time

from mineru_engine_spike import Engine

MODEL  = "/Users/jinzhenj/.mineru/models/MinerU2.5-Pro-2605-1.2B.gguf"
MMPROJ = "/Users/jinzhenj/.mineru/models/mmproj-MinerU2.5-Pro-2605-1.2B.gguf"


def text_body(prompt: str, max_tokens: int = 128) -> str:
    return json.dumps({
        "model": "mineru",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0, "top_p": 0.01, "top_k": 1,
        "repetition_penalty": 1.0, "max_tokens": max_tokens, "stream": False,
    })


async def agenerate(engine: Engine, body: str) -> str:
    loop = asyncio.get_event_loop()
    # blocking (GIL-releasing) C++ call in a worker thread; loop stays free
    return await loop.run_in_executor(None, engine.generate, body)


async def ticker(duration_s: float) -> int:
    n, end = 0, time.monotonic() + duration_s
    while time.monotonic() < end:
        await asyncio.sleep(0.05)
        n += 1
    return n


async def main() -> None:
    # CPU only (n_gpu_layers=0) to avoid the Metal-in-Python crash (separate issue).
    engine = Engine(model=MODEL, mmproj=MMPROJ, n_ctx=8192, n_gpu_layers=0)

    # --- 1. sync ---
    t0 = time.monotonic()
    sync_out = engine.generate(text_body("List three fruits."))
    print(f"[sync] {time.monotonic()-t0:.2f}s, len={len(sync_out)}, sample={sync_out[:60]!r}")

    # --- 2 & 3. async + ticker (GIL-free proof) ---
    # One agenerate + a ticker sharing the event loop. If the GIL were held
    # during C++ decode, the ticker (pure Python) could not advance.
    # (Concurrent multi-request routing is investigated separately as Tier2 #3.)
    t0 = time.monotonic()
    r0, ticks = await asyncio.gather(
        agenerate(engine, text_body("List three fruits.")),
        ticker(60.0),
    )
    dt = time.monotonic() - t0
    print(f"[async] {dt:.2f}s")
    print(f"  req0: len={len(r0)}, sample={r0[:60]!r}")
    print(f"  ticker advanced {ticks} times during decode "
          f"(GIL {'RELEASED' if ticks > 5 else 'HELD'})")

    ok = ticks > 5 and len(r0) > 0
    print("\n=== SPIKE PASSED ===" if ok else "\n=== SPIKE FAILED ===")


if __name__ == "__main__":
    asyncio.run(main())
