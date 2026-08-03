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
