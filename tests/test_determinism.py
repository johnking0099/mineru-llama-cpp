"""Covers temp=0 reproducibility (Tier2 #7 from the technical-validation
phase): same prompt + seed + temp=0 must produce byte-identical output,
both within one Engine and across separate Engine instances."""

import asyncio

from mineru_llama_cpp import Engine, SamplingParams

MODEL = "/Users/jinzhenj/.mineru/models/MinerU2.5-Pro-2605-1.2B-Q8_0.gguf"
MMPROJ = "/Users/jinzhenj/.mineru/models/mmproj-MinerU2.5-Pro-2605-1.2B-Q8_0.gguf"

_MESSAGES = [{"role": "user", "content": "List three fruits, one per line."}]
_SP = SamplingParams(temperature=0.0, top_k=1, seed=42, n_predict=64)


def test_repeated_calls_are_byte_identical(engine):
    outputs = {engine.generate(_MESSAGES, _SP).content for _ in range(5)}
    assert len(outputs) == 1, f"non-deterministic output across repeated calls: {outputs}"


async def test_concurrent_slots_produce_identical_output(engine):
    results = await asyncio.gather(*[engine.agenerate(_MESSAGES, _SP) for _ in range(4)])
    outputs = {r.content for r in results}
    assert len(outputs) == 1, f"non-deterministic output across concurrent slots: {outputs}"


def test_determinism_holds_across_separate_engine_instances():
    with Engine(MODEL, MMPROJ, n_ctx=4096) as e1:
        out1 = e1.generate(_MESSAGES, _SP).content
    with Engine(MODEL, MMPROJ, n_ctx=4096) as e2:
        out2 = e2.generate(_MESSAGES, _SP).content
    assert out1 == out2
