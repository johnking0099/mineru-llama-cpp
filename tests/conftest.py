"""Shared pytest fixtures. See Phase E header note on fixture image sizing."""

from pathlib import Path

import pytest

from mineru_llama_cpp import Engine

MODEL = "/Users/jinzhenj/.mineru/models/MinerU2.5-Pro-2605-1.2B-Q8_0.gguf"
MMPROJ = "/Users/jinzhenj/.mineru/models/mmproj-MinerU2.5-Pro-2605-1.2B-Q8_0.gguf"

FIXTURES_DIR = Path(__file__).parent / "fixtures"
_LAYOUT_IMAGE_SOURCE = Path(
    "/Users/jinzhenj/Downloads/OmniDocBench/v1_2_0/magazine_TheEconomist.2023.12.23_page_052.png"
)
_LAYOUT_IMAGE_1036 = FIXTURES_DIR / "layout_1036.png"


@pytest.fixture(scope="session")
def layout_image_path() -> Path:
    """A 1036x1036 test image, generated once per test session (not checked
    into git -- see .gitignore). Resized the same way mineru-vl-utils
    resizes for its own layout-detection step; see the Phase E header note
    for why the exact size matters."""
    FIXTURES_DIR.mkdir(exist_ok=True)
    if not _LAYOUT_IMAGE_1036.exists():
        if not _LAYOUT_IMAGE_SOURCE.exists():
            pytest.skip(f"source image not found: {_LAYOUT_IMAGE_SOURCE}")
        from PIL import Image

        img = Image.open(_LAYOUT_IMAGE_SOURCE).convert("RGB")
        img.resize((1036, 1036), Image.Resampling.BICUBIC).save(_LAYOUT_IMAGE_1036)
    return _LAYOUT_IMAGE_1036


@pytest.fixture(scope="session")
def engine():
    """One Engine instance shared across the whole test session (loading
    the ~1.2GB model repeatedly per-test would make the suite impractically
    slow). n_parallel=4 so concurrency tests (Task 24) have multiple slots
    to work with; this is harmless for tests that only ever issue one
    request at a time."""
    eng = Engine(MODEL, MMPROJ, n_ctx_seq=8192, n_gpu_layers=99, n_parallel=4)
    yield eng
    eng.close()
