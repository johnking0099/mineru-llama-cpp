"""Covers design spec's "路 A（JSON/oaicompat 路径）+ BF16/Metal 崩溃回避" test
coverage goal, using non-streaming generate()."""

import base64
import json
from pathlib import Path

from mineru_llama_cpp import Engine, GenerateResult, SamplingParams


def _image_data_uri(path: Path) -> str:
    data = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


def test_engine_preserves_explicit_grammar():
    """The wildcard UTF-8 grammar is a default, not an override: a caller
    who supplies their own "grammar" key (via a SamplingParams-like object
    whose to_json_fields() returns one) must win over it. See engine.py's
    _VALID_UNICODE_GRAMMAR comment for why this matters."""
    assert "grammar" not in SamplingParams.__dataclass_fields__
    engine = object.__new__(Engine)
    engine._eos_token_str = ""

    class CustomSamplingParams:
        @staticmethod
        def to_json_fields():
            return {"grammar": 'root ::= "overridden"'}

    body = json.loads(
        engine._build_body([{"role": "user", "content": "test"}], CustomSamplingParams(), False)  # type: ignore[arg-type]
    )
    assert body["grammar"] == 'root ::= "overridden"'


def test_generate_text_only(engine):
    result = engine.generate([{"role": "user", "content": "Say hello in one word."}])
    assert isinstance(result, GenerateResult)
    assert result.content
    assert result.finish_reason in ("stop", "length")
    assert result.tokens_predicted > 0


def test_generate_with_image_layout_detection(engine, layout_image_path):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _image_data_uri(layout_image_path)}},
                {"type": "text", "text": "\nLayout Detection:"},
            ],
        },
    ]
    sp = SamplingParams(temperature=0.0, top_p=0.01, top_k=1, repeat_penalty=1.0, n_predict=512)
    result = engine.generate(messages, sp)
    assert "<|box_start|>" in result.content
    # 1369 image tokens (1036x1036 image) + ~25 text tokens; confirms the
    # image was actually routed through mtmd, not silently dropped.
    assert result.tokens_evaluated > 1000


def test_generate_strips_eos_token_by_default(engine):
    """Engine is constructed with special=true (see engine_core.cpp), so the
    model's chat-template EOS token would otherwise leak into `content` as
    literal text (e.g. trailing "...<|im_end|>") whenever generation stops
    naturally instead of being cut short by n_predict. Engine auto-fills
    SamplingParams.stop with the model's EOS token string when the caller
    doesn't set one, which strips it -- verify that actually happens."""
    sp = SamplingParams(temperature=0.0, top_k=1, n_predict=32)
    result = engine.generate([{"role": "user", "content": "Say hello in one word."}], sp)
    assert result.finish_reason == "stop", "test assumes natural EOS stop, not a length cutoff"
    assert "<|im_end|>" not in result.content


def test_generate_explicit_empty_stop_keeps_eos_token(engine):
    """Passing stop=[] is the caller opting out of Engine's default
    EOS-stripping behavior -- must not be silently overridden."""
    sp = SamplingParams(temperature=0.0, top_k=1, n_predict=32, stop=[])
    result = engine.generate([{"role": "user", "content": "Say hello in one word."}], sp)
    assert result.finish_reason == "stop"
    assert result.content.endswith("<|im_end|>")
