"""Tier2 #5: error isolation verification.

Proves bad requests don't crash the Engine or block subsequent good ones:
  1. Malformed JSON body -> Python exception, engine alive.
  2. Invalid image_url base64 -> Python exception, engine alive.
  3. Over-long context (> n_ctx) -> Python exception or graceful truncation, engine alive.
  4. After each bad request, a good request still succeeds.

The key invariant: a long-lived Engine must serve many requests; one bad input
must not take down the loop thread or corrupt state.
"""

import asyncio
import json
import base64
import traceback

from mineru_engine_spike import Engine

MODEL  = "/Users/jinzhenj/.mineru/models/MinerU2.5-Pro-2605-1.2B-Q8_0.gguf"
MMPROJ = "/Users/jinzhenj/.mineru/models/mmproj-MinerU2.5-Pro-2605-1.2B-Q8_0.gguf"


def good_body(prompt: str = "hi", max_tokens: int = 8) -> str:
    return json.dumps({
        "model": "m",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "top_k": 1, "max_tokens": max_tokens, "stream": False,
    })


def expect_ok(engine: Engine, label: str) -> bool:
    try:
        out = engine.generate(good_body())
        print(f"  [after {label}] good request OK, len={len(out)}")
        return True
    except Exception as e:
        print(f"  [after {label}] good request FAILED: {e!r}")
        return False


async def main() -> None:
    engine = Engine(model=MODEL, mmproj=MMPROJ, n_ctx=2048, n_gpu_layers=99, n_parallel=1)

    results = []

    # --- baseline: good request works ---
    ok = expect_ok(engine, "baseline")
    results.append(("baseline_good", ok))

    # --- 1. malformed JSON ---
    print("\n[1] malformed JSON body")
    try:
        engine.generate("{not valid json")
        print("  no exception (unexpected)")
        results.append(("malformed_json", False))
    except Exception as e:
        print(f"  raised: {type(e).__name__}: {str(e)[:80]}")
        results.append(("malformed_json", True))
    results.append(("after_malformed", expect_ok(engine, "malformed")))

    # --- 2. invalid base64 image ---
    print("\n[2] invalid base64 image_url")
    bad = json.dumps({
        "model": "m",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!notvalidbase64!!!"}},
            {"type": "text", "text": "describe"},
        ]}],
        "max_tokens": 8, "stream": False,
    })
    try:
        out = engine.generate(bad)
        print(f"  no exception, got len={len(out)} (engine tolerated?)")
        results.append(("bad_base64", True))
    except Exception as e:
        print(f"  raised: {type(e).__name__}: {str(e)[:100]}")
        results.append(("bad_base64", True))
    results.append(("after_bad_base64", expect_ok(engine, "bad_base64")))

    # --- 3. over-long context (> n_ctx=2048) ---
    print("\n[3] over-long context (prompt >> n_ctx)")
    huge = "x " * 4000  # ~4000 tokens, well over 2048
    big = json.dumps({
        "model": "m",
        "messages": [{"role": "user", "content": huge}],
        "max_tokens": 4, "stream": False,
    })
    try:
        out = engine.generate(big)
        print(f"  no exception, got len={len(out)} (truncated gracefully)")
        results.append(("overlong_ctx", True))
    except Exception as e:
        print(f"  raised: {type(e).__name__}: {str(e)[:100]}")
        results.append(("overlong_ctx", True))
    results.append(("after_overlong", expect_ok(engine, "overlong")))

    # --- 4. empty messages ---
    print("\n[4] empty messages array")
    empty = json.dumps({"model": "m", "messages": [], "max_tokens": 4, "stream": False})
    try:
        out = engine.generate(empty)
        print(f"  no exception, got len={len(out)}")
        results.append(("empty_messages", True))
    except Exception as e:
        print(f"  raised: {type(e).__name__}: {str(e)[:100]}")
        results.append(("empty_messages", True))
    results.append(("after_empty", expect_ok(engine, "empty")))

    # --- final: engine still alive ---
    print("\n[final] engine still serves good requests")
    final_ok = expect_ok(engine, "final")

    passed = all(r[1] for r in results) and final_ok
    print(f"\n=== Summary ===")
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"\nfinal engine alive: {'YES' if final_ok else 'NO ✗'}")
    print("\n=== SPIKE PASSED ===" if passed else "\n=== SPIKE FAILED ===")


if __name__ == "__main__":
    asyncio.run(main())
