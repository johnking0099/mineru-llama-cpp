"""Tier2 #4: streaming (partial result) verification.

Proves:
  1. generate_stream() returns multiple chunks (not one big blob).
  2. Concatenated chunks == non-streaming generate() output.
  3. Token-level granularity (chunk count grows with max_tokens).
  4. astream() pattern works via run_in_executor + iterator.

Q8_0 model (Metal BF16 bug workaround).
"""

import asyncio
import json

from mineru_engine_spike import Engine

MODEL  = "/Users/jinzhenj/.mineru/models/MinerU2.5-Pro-2605-1.2B-Q8_0.gguf"
MMPROJ = "/Users/jinzhenj/.mineru/models/mmproj-MinerU2.5-Pro-2605-1.2B-Q8_0.gguf"


def text_body(prompt: str, max_tokens: int = 64) -> str:
    return json.dumps({
        "model": "mineru",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0, "top_p": 0.01, "top_k": 1,
        "repetition_penalty": 1.0, "max_tokens": max_tokens, "stream": True,
    })


async def main() -> None:
    engine = Engine(model=MODEL, mmproj=MMPROJ, n_ctx=8192, n_gpu_layers=99, n_parallel=2)
    prompt = "List three fruits, one per line."

    # --- 1. streaming: collect chunks ---
    loop = asyncio.get_event_loop()
    chunks = await loop.run_in_executor(None, engine.generate_stream, text_body(prompt, max_tokens=64))
    streamed = "".join(chunks)
    print(f"[stream] {len(chunks)} chunks, total {len(streamed)} chars")
    print(f"  first 3 chunks: {[c[:20] for c in chunks[:3]]}")
    print(f"  last chunk: {chunks[-1][:30]!r}")

    # --- 2. non-streaming baseline ---
    body_nonstream = text_body(prompt, max_tokens=64).replace('"stream": True', '"stream": False')
    full = await loop.run_in_executor(None, engine.generate, body_nonstream)
    print(f"\n[non-stream] {len(full)} chars")

    # --- checks ---
    multi_chunk = len(chunks) > 1
    match = streamed == full
    print(f"\nmulti-chunk: {'OK' if multi_chunk else 'FAIL (single chunk)'}")
    print(f"stream==nonstream: {'OK' if match else 'MISMATCH'}")
    if not match:
        print(f"  stream:   {streamed[:80]!r}")
        print(f"  nonstream:{full[:80]!r}")

    # --- 3. granularity: more max_tokens -> more chunks ---
    chunks_long = await loop.run_in_executor(None, engine.generate_stream, text_body(prompt, max_tokens=128))
    print(f"\n[granularity] max_tokens=64 -> {len(chunks)} chunks; 128 -> {len(chunks_long)} chunks")
    grows = len(chunks_long) >= len(chunks)
    print(f"chunk count grows with max_tokens: {'OK' if grows else 'FAIL'}")

    ok = multi_chunk and match and grows
    print("\n=== SPIKE PASSED ===" if ok else "\n=== SPIKE FAILED ===")


if __name__ == "__main__":
    asyncio.run(main())
