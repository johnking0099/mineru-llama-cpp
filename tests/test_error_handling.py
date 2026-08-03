"""Covers the exception hierarchy (design spec §5.3) and error isolation
(Tier2 #5 from the technical-validation phase: a bad request must not take
down the Engine)."""

import pytest

from mineru_llama_cpp import ContextExceededError, Engine, InvalidRequestError

MODEL = "/Users/jinzhenj/.mineru/models/MinerU2.5-Pro-2605-1.2B-Q8_0.gguf"
MMPROJ = "/Users/jinzhenj/.mineru/models/mmproj-MinerU2.5-Pro-2605-1.2B-Q8_0.gguf"

_BAD_IMAGE_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!notvalidbase64!!!"}},
            {"type": "text", "text": "describe"},
        ],
    }
]


def test_malformed_json_raises_invalid_request(engine):
    # Engine.generate() always builds valid JSON itself, so malformed JSON
    # can only be exercised by going through the private _core directly.
    with pytest.raises(InvalidRequestError):
        engine._core.generate("{not valid json")


def test_bad_image_raises_invalid_request(engine):
    with pytest.raises(InvalidRequestError):
        engine.generate(_BAD_IMAGE_MESSAGES)


def test_context_exceeded_raises_context_exceeded_error():
    # A dedicated small-n_ctx Engine (not the shared session fixture), so
    # n_ctx=512 here doesn't affect any other test.
    with Engine(MODEL, MMPROJ, n_ctx=512) as small_engine:
        huge_prompt = "x " * 2000  # far more than 512 tokens
        with pytest.raises(ContextExceededError):
            small_engine.generate([{"role": "user", "content": huge_prompt}])


def test_context_exceeded_is_also_an_invalid_request_error():
    with Engine(MODEL, MMPROJ, n_ctx=512) as small_engine:
        huge_prompt = "x " * 2000
        with pytest.raises(InvalidRequestError):  # the base-class catch must also work
            small_engine.generate([{"role": "user", "content": huge_prompt}])


def test_engine_survives_bad_requests_and_serves_good_ones_after(engine):
    good = [{"role": "user", "content": "hi"}]
    assert engine.generate(good).content

    with pytest.raises(InvalidRequestError):
        engine._core.generate("{not valid json")
    assert engine.generate(good).content, "engine unusable after malformed JSON"

    with pytest.raises(InvalidRequestError):
        engine.generate(_BAD_IMAGE_MESSAGES)
    assert engine.generate(good).content, "engine unusable after bad image"
