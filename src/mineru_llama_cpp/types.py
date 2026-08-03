"""Input/output types for mineru_llama_cpp.Engine. See design spec §5.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict


# --- Input: messages ---

class ImageURL(TypedDict):
    url: str
    """A base64 data URI string, e.g. "data:image/png;base64,....".

    Does NOT accept local paths, HTTP URLs, or bare base64 without the
    "data:" prefix — encoding is entirely the caller's responsibility
    (design spec §6)."""


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImagePart(TypedDict):
    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = TextPart | ImagePart


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]


Messages = list[Message]


# --- Output: Engine.generate()/agenerate() and Engine.stream()/astream() ---

@dataclass(frozen=True)
class GenerationTimings:
    prompt_n: int
    prompt_ms: float
    prompt_per_second: float
    predicted_n: int
    predicted_ms: float
    predicted_per_second: float


@dataclass(frozen=True)
class GenerateResult:
    content: str
    finish_reason: Literal["stop", "length"]
    tokens_evaluated: int
    tokens_predicted: int
    timings: GenerationTimings


@dataclass(frozen=True)
class GenerateChunk:
    delta: str
    finish_reason: Literal["stop", "length"] | None = None
    tokens_evaluated: int | None = None
    tokens_predicted: int | None = None
    timings: GenerationTimings | None = None
