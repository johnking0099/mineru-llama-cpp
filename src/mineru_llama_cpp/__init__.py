import os
from pathlib import Path

if os.name == "nt":
    # Windows has no RPATH ($ORIGIN / @loader_path). The .pyd's DLL
    # dependencies (libllama.dll, libggml*.dll, etc.) live in a sibling
    # bin/ directory -- either in the wheel (mineru_llama_cpp/bin/) or, for
    # editable inplace builds, in the build tree (<repo>/bin/). Add those
    # directories to the DLL search path BEFORE importing _mineru_llama_cpp,
    # which triggers the .pyd load and its DT_NEEDED resolution. The three
    # candidates mirror load_packaged_backends() in engine_core.cpp.
    _pkg = Path(__file__).resolve().parent
    for _d in [_pkg / "bin", _pkg.parent / "bin", _pkg.parent.parent / "bin"]:
        if _d.is_dir():
            os.add_dll_directory(str(_d))

from .engine import Engine
from .exceptions import (
    ContextExceededError,
    EngineError,
    InvalidRequestError,
    MineruLlamaCppError,
)
from .sampling import SamplingParams
from .types import (
    ContentPart,
    GenerateChunk,
    GenerateResult,
    GenerationTimings,
    ImagePart,
    ImageURL,
    Message,
    Messages,
    TextPart,
)
from .verbosity import (
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_INFO,
    LOG_LEVEL_OUTPUT,
    LOG_LEVEL_TRACE,
    LOG_LEVEL_WARN,
)

__all__ = [
    "Engine",
    "SamplingParams",
    "MineruLlamaCppError",
    "InvalidRequestError",
    "ContextExceededError",
    "EngineError",
    "Message",
    "Messages",
    "ContentPart",
    "TextPart",
    "ImagePart",
    "ImageURL",
    "GenerateResult",
    "GenerateChunk",
    "GenerationTimings",
    "LOG_LEVEL_OUTPUT",
    "LOG_LEVEL_ERROR",
    "LOG_LEVEL_WARN",
    "LOG_LEVEL_INFO",
    "LOG_LEVEL_TRACE",
    "LOG_LEVEL_DEBUG",
]
