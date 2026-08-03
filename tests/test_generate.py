"""Covers design spec's "路 A（JSON/oaicompat 路径）+ BF16/Metal 崩溃回避" test
coverage goal, using non-streaming generate()."""

import base64
from pathlib import Path

from mineru_llama_cpp import GenerateResult, SamplingParams


def _image_data_uri(path: Path) -> str:
    data = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


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
