# mineru-llama-cpp

In-process llama.cpp VLM inference engine for MinerU, exposing a single
`Engine` class with synchronous and asynchronous generate/stream methods.
Wraps a pinned build of [llama.cpp](https://github.com/ggml-org/llama.cpp)
(no HTTP layer, no subprocess) via pybind11.

## Install (development)

```bash
git clone --recurse-submodules <this-repo-url>
cd mineru-llama-cpp
pip install --no-build-isolation -e .
pip install -e ".[test]"
```

The first build compiles llama.cpp from source (a few minutes); subsequent
builds only recompile this library's own C++ files.

## Usage

```python
from mineru_llama_cpp import Engine, SamplingParams

with Engine("/path/to/model.gguf", "/path/to/mmproj.gguf") as engine:
    result = engine.generate([{"role": "user", "content": "hello"}])
    print(result.content)

    for chunk in engine.stream([{"role": "user", "content": "hello"}]):
        print(chunk.delta, end="", flush=True)
```

Async:

```python
async with Engine("/path/to/model.gguf", "/path/to/mmproj.gguf") as engine:
    result = await engine.agenerate([{"role": "user", "content": "hello"}])
    async for chunk in engine.astream([{"role": "user", "content": "hello"}]):
        print(chunk.delta, end="", flush=True)
```

Images go in `content` as `{"type": "image_url", "image_url": {"url": "data:image/png;base64,...."}}`
— **must** be a pre-encoded base64 data URI; this library does not accept
`PIL.Image` objects, file paths, or HTTP URLs (see
`docs/superpowers/specs/2026-08-03-mineru-llama-cpp-engine-design.md` §6 for
why).

## Known issues

See `docs/known-issues.md` — notably: **use Q8_0 models, not BF16**, on
Metal.

## Status

v0.1.0 — local development package only. No wheel/CI packaging, no
HuggingFace auto-download, no batch-generate API, no CUDA/Vulkan testing.
See the design spec's §1 "非目标" for the full non-goals list.
