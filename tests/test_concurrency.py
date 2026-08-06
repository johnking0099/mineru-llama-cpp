"""Covers multi-slot concurrency + result routing (Tier2 #3 from the
technical-validation phase)."""

import asyncio
import time

from mineru_llama_cpp import SamplingParams

_PROMPTS = [
    "List three fruits, one per line.",
    "List three colors, one per line.",
    "List three animals, one per line.",
    "List three cities, one per line.",
]


async def test_concurrent_agenerate_routing_matches_serial(engine):
    sp = SamplingParams(temperature=0.0, top_k=1, n_predict=32)

    def body(prompt: str) -> list[dict]:
        return [{"role": "user", "content": prompt}]

    serial = []
    for p in _PROMPTS:
        serial.append(await engine.agenerate(body(p), sp))

    concurrent = await asyncio.gather(*[engine.agenerate(body(p), sp) for p in _PROMPTS])

    for i in range(len(_PROMPTS)):
        assert concurrent[i].content == serial[i].content, f"routing mismatch at prompt index {i}"


async def test_concurrent_agenerate_faster_than_serial(engine):
    sp = SamplingParams(temperature=0.0, top_k=1, n_predict=32)

    def body(prompt: str) -> list[dict]:
        return [{"role": "user", "content": prompt}]

    t0 = time.monotonic()
    for p in _PROMPTS:
        await engine.agenerate(body(p), sp)
    serial_dt = time.monotonic() - t0

    t0 = time.monotonic()
    await asyncio.gather(*[engine.agenerate(body(p), sp) for p in _PROMPTS])
    concurrent_dt = time.monotonic() - t0

    assert concurrent_dt < serial_dt, f"concurrent ({concurrent_dt:.2f}s) not faster than serial ({serial_dt:.2f}s)"


async def test_concurrent_agenerate_beyond_slot_count_no_empty_content(engine):
    """Regression test for a llama.cpp bug (patched in patches/llama.cpp/
    0001-fix-cli-task-double-tokenize.patch): requests that have to wait for
    a slot to free up (more concurrent requests than n_parallel slots) used
    to come back with empty content and stop_type="none", because the
    deferred task's cli_prompt got tokenized twice -- the second time
    against an already-cleared (empty) prompt. The `engine` fixture uses
    n_parallel=4, so 8 concurrent requests guarantees at least 4 of them
    have to queue for a slot."""
    sp = SamplingParams(temperature=0.0, top_k=1, n_predict=32)
    prompts = [f"Say the number {i} in words." for i in range(8)]

    results = await asyncio.gather(*[
        engine.agenerate([{"role": "user", "content": p}], sp) for p in prompts
    ])

    for i, r in enumerate(results):
        assert r.content.strip(), f"request {i} (queued past slot count) returned empty content"
        assert r.finish_reason == "stop", f"request {i} unexpected finish_reason: {r.finish_reason!r}"
