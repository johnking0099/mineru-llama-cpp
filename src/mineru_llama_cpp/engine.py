"""The Engine class. See design spec §5.4."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import TracebackType
from typing import AsyncIterator, Iterator

from ._mineru_llama_cpp import _EngineCore
from .sampling import SamplingParams
from .types import GenerateChunk, GenerateResult, GenerationTimings, Messages


def _timings_from_dict(d: dict | None) -> GenerationTimings | None:
    if d is None:
        return None
    return GenerationTimings(**d)


class Engine:
    def __init__(
        self,
        model: str | Path,
        mmproj: str | Path,
        *,
        n_ctx: int = 8192,
        n_gpu_layers: int = 99,
        n_parallel: int = 1,
    ) -> None:
        """Loads the model and starts the background decode loop thread.

        model/mmproj must be local file paths (no HuggingFace repo id
        auto-download — see design spec §1 "非目标").
        """
        self._core = _EngineCore(str(model), str(mmproj), n_ctx, n_gpu_layers, n_parallel)
        self._closed = False

    def _build_body(self, messages: Messages, sampling_params: SamplingParams | None, stream: bool) -> str:
        body: dict = {"model": "mineru-llama-cpp", "messages": messages, "stream": stream}
        if sampling_params is not None:
            body.update(sampling_params.to_json_fields())
        return json.dumps(body)

    # --- non-streaming ---

    def generate(
        self,
        messages: Messages,
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        """Blocking call. Raises InvalidRequestError/ContextExceededError/
        EngineError on failure (see exceptions.py)."""
        body = self._build_body(messages, sampling_params, stream=False)
        d = self._core.generate(body)
        return GenerateResult(
            content=d["content"],
            finish_reason=d["finish_reason"],
            tokens_evaluated=d["tokens_evaluated"],
            tokens_predicted=d["tokens_predicted"],
            timings=_timings_from_dict(d["timings"]),
        )

    async def agenerate(
        self,
        messages: Messages,
        sampling_params: SamplingParams | None = None,
    ) -> GenerateResult:
        """Async version of generate(), bridged via run_in_executor — does
        not block the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, messages, sampling_params)

    # --- streaming ---

    def stream(
        self,
        messages: Messages,
        sampling_params: SamplingParams | None = None,
    ) -> Iterator[GenerateChunk]:
        """Synchronous generator, yields one GenerateChunk per token. The
        final chunk has finish_reason set (all others have it as None) —
        check `chunk.finish_reason is not None` to detect the end."""
        body = self._build_body(messages, sampling_params, stream=True)
        core_iter = self._core.generate_stream(body)
        for c in core_iter:
            yield GenerateChunk(
                delta=c["delta"],
                finish_reason=c["finish_reason"],
                tokens_evaluated=c["tokens_evaluated"],
                tokens_predicted=c["tokens_predicted"],
                timings=_timings_from_dict(c["timings"]),
            )

    async def astream(
        self,
        messages: Messages,
        sampling_params: SamplingParams | None = None,
    ) -> AsyncIterator[GenerateChunk]:
        """Async version of stream(): a real generator-to-generator bridge.

        run_in_executor() alone can't drive a synchronous *generator* (it's
        built for "call once, get one result"), so this runs stream() to
        completion on a background thread and relays each chunk back to the
        event loop through an asyncio.Queue via call_soon_threadsafe — the
        standard sync-iterator-to-async-iterator bridge pattern.

        Note: breaking out of the consuming loop early (or otherwise
        abandoning this generator before it's exhausted) does not cancel
        the underlying generation — the background thread runs to
        completion regardless, occupying a slot until the model finishes.
        """
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()

        def _run() -> None:
            try:
                for chunk in self.stream(messages, sampling_params):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:  # noqa: BLE001 - relayed to the consumer, not swallowed
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item
