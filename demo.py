"""End-to-end usage demo for mineru_llama_cpp.

Walks through the library's public API in one script: sync generate/stream,
async agenerate/astream, image input, sampling parameters, and error
handling. Run after an editable install (see README.md):

    .venv/bin/python demo.py --model /path/to/model.gguf --mmproj /path/to/mmproj.gguf
    .venv/bin/python demo.py --model ... --mmproj ... --image /path/to/page.png

With no --image, the image-input section is skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from pathlib import Path

from mineru_llama_cpp import (
    ContextExceededError,
    Engine,
    GenerateResult,
    Messages,
    SamplingParams,
)


def _image_data_uri(path: Path) -> str:
    """Encode an image file as the base64 data URI the Engine expects.

    mineru_llama_cpp never accepts file paths, HTTP URLs, or PIL.Image
    objects in message content -- encoding is entirely the caller's
    responsibility (see README.md's "Images go in `content`" note).
    """
    data = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


def demo_sync_generate(engine: Engine) -> None:
    print("\n=== 1. Synchronous generate() ===")
    # MinerU2.5 is a document-VLM tuned for OCR/layout tasks, not general
    # chat -- a plain conversational prompt with no image can occasionally
    # run all the way to n_ctx generating degenerate output instead of
    # stopping. n_predict caps that so the demo stays fast and readable;
    # temperature=0/top_k=1 keep it deterministic.
    sp = SamplingParams(n_predict=640, temperature=0.0, top_k=1)
    result = engine.generate([{"role": "user", "content": "Say hello in exactly one word."}], sp)
    assert isinstance(result, GenerateResult)
    print(f"content:          {result.content!r}")
    print(f"finish_reason:    {result.finish_reason}")
    print(f"tokens_predicted: {result.tokens_predicted}")
    if result.timings is not None:
        print(f"predicted_per_second: {result.timings.predicted_per_second:.1f} tok/s")


def demo_sync_stream(engine: Engine) -> None:
    print("\n=== 2. Synchronous stream() ===")
    print("delta chunks: ", end="", flush=True)
    for chunk in engine.stream([{"role": "user", "content": "Count from one to five."}]):
        print(chunk.delta, end="", flush=True)
        if chunk.finish_reason is not None:
            print(f"\n(finished: {chunk.finish_reason}, {chunk.tokens_predicted} tokens)")


def demo_sampling_params(engine: Engine) -> None:
    print("\n=== 3. SamplingParams (deterministic output via temperature=0) ===")
    sp = SamplingParams(temperature=0.0, top_k=1, n_predict=16, seed=42)
    result = engine.generate([{"role": "user", "content": "What is 2 + 2?"}], sp)
    print(f"content: {result.content!r}")


def demo_image_input(engine: Engine, image_path: Path) -> None:
    print("\n=== 4. Image input (layout detection) ===")
    messages: Messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _image_data_uri(image_path)}},
                {"type": "text", "text": "\nLayout Detection:"},
            ],
        },
    ]
    sp = SamplingParams(temperature=0.0, top_p=0.01, top_k=1, repeat_penalty=1.0, n_predict=None)
    result = engine.generate(messages, sp)
    print(f"tokens_evaluated: {result.tokens_evaluated}  (includes image tokens)")
    print(f"content: {result.content!r}")


def demo_error_handling(engine: Engine) -> None:
    print("\n=== 5. Error handling ===")
    huge_prompt = "word " * 100_000  # far beyond n_ctx
    try:
        engine.generate([{"role": "user", "content": huge_prompt}])
    except ContextExceededError as exc:
        print(f"caught ContextExceededError as expected: {exc}")
    else:
        print("(prompt did not exceed context -- try a larger --n-ctx or a bigger prompt)")


async def demo_async(engine: Engine) -> None:
    print("\n=== 6. Async agenerate()/astream() ===")
    result = await engine.agenerate([{"role": "user", "content": "Say goodbye in one word."}])
    print(f"agenerate content: {result.content!r}")

    print("astream delta chunks: ", end="", flush=True)
    async for chunk in engine.astream([{"role": "user", "content": "Count from one to three."}]):
        print(chunk.delta, end="", flush=True)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Path to the main .gguf model")
    parser.add_argument("--mmproj", required=True, type=Path, help="Path to the mmproj .gguf file")
    parser.add_argument("--image", type=Path, default=None, help="Optional image to run layout detection on")
    parser.add_argument("--n-ctx-seq", type=int, default=8192, help="Per-slot context length (default: 8192)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Engine is also usable as a plain (non-context-manager) object -- just
    # remember to call engine.close() yourself in that case.
    with Engine(args.model, args.mmproj, n_ctx_seq=args.n_ctx_seq) as engine:
        # demo_sync_generate(engine)
        # demo_sync_stream(engine)
        # demo_sampling_params(engine)
        if args.image is not None:
            demo_image_input(engine, args.image)
        # else:
        #     print("\n=== 4. Image input === (skipped: pass --image to run this section)")
        # demo_error_handling(engine)
        # asyncio.run(demo_async(engine))

    print("\nEngine closed cleanly. Done.")


if __name__ == "__main__":
    main()
