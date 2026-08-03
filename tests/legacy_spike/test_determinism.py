"""Tier3 #9: determinism (temp=0) reproducibility verification.

Proves with temperature=0 + fixed seed:
  1. Same input, same engine, run 3 times -> byte-identical output.
  2. Same input across different slots (n_parallel=4) -> identical output.
  3. (informational) without explicit seed (random) -> may differ, to confirm
     the seed field actually controls determinism.

Q8_0 model (Metal BF16 bug workaround). temp=0 = greedy decoding.
"""

import asyncio
import json

from mineru_engine_spike import Engine

MODEL  = "/Users/jinzhenj/.mineru/models/MinerU2.5-Pro-2605-1.2B-Q8_0.gguf"
MMPROJ = "/Users/jinzhenj/.mineru/models/mmproj-MinerU2.5-Pro-2605-1.2B-Q8_0.gguf"


def body(prompt: str, seed: int | None, max_tokens: int = 32) -> str:
    d = {
        "model": "m",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "top_p": 0.01, "top_k": 1,
        "repetition_penalty": 1.0, "max_tokens": max_tokens, "stream": False,
    }
    if seed is not None:
        d["seed"] = seed
    return json.dumps(d)


async def main() -> None:
    engine = Engine(model=MODEL, mmproj=MMPROJ, n_ctx=4096, n_gpu_layers=99, n_parallel=4)
    loop = asyncio.get_event_loop()

    prompt = "List three fruits, one per line."

    # --- 1. same engine, 3 sequential runs, fixed seed ---
    print("[1] same engine, 3 sequential runs, seed=42, temp=0")
    outs = []
    for i in range(3):
        out = await loop.run_in_executor(None, engine.generate, body(prompt, seed=42))
        outs.append(out)
        print(f"  run {i}: len={len(out)} hash={hash(out)} sample={out[:40]!r}")
    seq_identical = len(set(outs)) == 1
    print(f"  3 runs identical: {'OK' if seq_identical else 'FAIL'}")

    # --- 2. cross-slot: 4 concurrent runs (different slots), fixed seed ---
    print("\n[2] 4 concurrent runs across slots, seed=42, temp=0")
    conc_outs = await asyncio.gather(
        *[loop.run_in_executor(None, engine.generate, body(prompt, seed=42)) for _ in range(4)]
    )
    for i, o in enumerate(conc_outs):
        print(f"  slot run {i}: len={len(o)} hash={hash(o)}")
    cross_slot_identical = len(set(conc_outs)) == 1
    print(f"  4 slot runs identical: {'OK' if cross_slot_identical else 'FAIL'}")

    # --- 3. informational: no explicit seed (random) ---
    print("\n[3] informational: no explicit seed (random)")
    rand_outs = []
    for i in range(3):
        out = await loop.run_in_executor(None, engine.generate, body(prompt, seed=None))
        rand_outs.append(out)
        print(f"  run {i}: len={len(out)} hash={hash(out)}")
    # temp=0 greedy may still be deterministic even without seed
    rand_identical = len(set(rand_outs)) == 1
    print(f"  random-seed 3 runs identical: {rand_identical} (informational; temp=0 may still be greedy)")

    ok = seq_identical and cross_slot_identical
    print("\n=== SPIKE PASSED ===" if ok else "\n=== SPIKE FAILED ===")


if __name__ == "__main__":
    asyncio.run(main())
