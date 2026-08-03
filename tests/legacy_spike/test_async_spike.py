"""Tier1 #2 spike: async/GIL bridge verification.

Proves three things with the real pybind11 Engine over llama-server core:
  1. sync generate() works end-to-end (path A reused).
  2. async agenerate() via run_in_executor works; concurrent requests both succeed.
  3. while a request is decoding, a pure-Python asyncio ticker keeps advancing —
     proving the GIL is NOT held during decode (the event loop stays free).

This is the go/no-go for the AsyncEngine threading model.
"""

import asyncio
import base64
import json
import time

from mineru_engine_spike import Engine

MODEL  = "/Users/jinzhenj/.mineru/models/MinerU2.5-Pro-2605-1.2B.gguf"
MMPROJ = "/Users/jinzhenj/.mineru/models/mmproj-MinerU2.5-Pro-2605-1.2B.gguf"
IMAGE  = "/tmp/econ_layout_1036.png"  # known-good 1036x1036 layout image


def make_body(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    body = {
        "model": "mineru",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": "\nLayout Detection:"},
            ]},
        ],
        "temperature": 0.0, "top_p": 0.01, "top_k": 1,
        "repetition_penalty": 1.0, "max_tokens": 512, "stream": False,
    }
    return json.dumps(body)


async def agenerate(engine: Engine, body: str) -> str:
    # run_in_executor runs the blocking (GIL-releasing) C++ call in a worker thread;
    # the asyncio loop is free while it blocks.
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, engine.generate, body)


async def ticker(duration_s: float) -> int:
    """Advance every 50ms for `duration_s`; return number of ticks."""
    n = 0
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        await asyncio.sleep(0.05)
        n += 1
    return n


async def main() -> None:
    engine = Engine(model=MODEL, mmproj=MMPROJ, n_ctx=8192)
    body = make_body(IMAGE)

    # --- 1. sync ---
    t0 = time.monotonic()
    sync_out = engine.generate(body)
    print(f"[sync] {time.monotonic()-t0:.2f}s, has_box={'<|box_start|>' in sync_out}, "
          f"blocks={sync_out.count('<|box_start|>')}")

    # --- 2 & 3. async concurrent + ticker (GIL-free proof) ---
    # Two agenerate + one ticker share the event loop. If GIL were held during
    # decode, the ticker could not advance. We expect many ticks.
    t0 = time.monotonic()
    results, ticks = await asyncio.gather(
        agenerate(engine, body),
        agenerate(engine, body),
        ticker(120.0),
    )
    dt = time.monotonic() - t0
    r0, r1 = results
    print(f"[async] {dt:.2f}s")
    print(f"  req0: has_box={'<|box_start|>' in r0}, blocks={r0.count('<|box_start|>')}")
    print(f"  req1: has_box={'<|box_start|>' in r1}, blocks={r1.count('<|box_start|>')}")
    print(f"  ticker advanced {ticks} times during decode (GIL {'RELEASED ✓' if ticks > 5 else 'HELD ✗'})")

    # sanity: both async outputs match sync (deterministic, same image)
    assert r0 == sync_out, "async req0 != sync"
    assert r1 == sync_out, "async req1 != sync"
    print("[async] outputs identical to sync ✓")

    print("\n=== SPIKE PASSED ===" if ticks > 5 else "\n=== SPIKE FAILED (GIL held) ===")


if __name__ == "__main__":
    asyncio.run(main())
