# mineru-llama-cpp

In-process llama.cpp VLM inference engine for MinerU, exposing a single
`Engine` class with synchronous and asynchronous generate/stream methods.
Wraps a pinned build of [llama.cpp](https://github.com/ggml-org/llama.cpp)
(no HTTP layer, no subprocess) via pybind11.

## Status

**Verified on 4 platforms** (build + import + text generate + two-step
document extraction):

| Platform | Backend | Two-step extract | Notes |
|---|---|---|---|
| macOS arm64 (Apple Silicon) | Metal | 7.3s / page | 6.4x faster than CPU |
| macOS arm64 | CPU | 47.1s / page | Fallback |
| Linux x86_64 (NVIDIA GPU) | Vulkan | ~16s / page | OpenMP ON + libgomp bundled |
| Windows x86_64 | CPU | 241.5s / page | OpenMP OFF |
| Windows x86_64 | Vulkan (Intel UHD) | 378.3s / page | Weak iGPU — CPU faster |
| Windows arm64 | CPU (clang-cl) | 208.8s / page | MSVC rejects ARM; clang-cl required |

Key build decisions (all in top-level `CMakeLists.txt`):
- `GGML_BACKEND_DL=ON` — backends are dlopen'd MODULEs, not hard-linked;
  missing GPU loaders (e.g. no `libvulkan.so.1`) gracefully fall back to CPU
- `LLAMA_OPENSSL=OFF` — drops OpenSSL/libssl/libcrypto dependency
- `GGML_NATIVE=OFF` — portable binaries (no `-march=native`); required for
  `GGML_BACKEND_DL` on x86 anyway
- `GGML_OPENMP` — ON on Linux (14% faster, libgomp bundled in wheel),
  OFF on macOS/Windows (zero benefit, removes libomp/libgomp dependency)
- `BUILD_SHARED_LIBS=ON` — prerequisite for `GGML_BACKEND_DL`

## CI wheels

Push to `main` (or a `v*` tag) triggers
[`build-wheels.yml`](.github/workflows/build-wheels.yml), which builds
28 wheels via cibuildwheel — 4 Python versions × 7 platform/arch combos:

| Platform | Wheel tag | Backend | Runner |
|---|---|---|---|
| macOS arm64 | `macosx_11_0_arm64` | Metal + CPU | macos-latest (Apple Silicon) |
| macOS x86_64 | `macosx_10_15_x86_64` | CPU only | macos-14 + Rosetta 2 |
| Linux x86_64 (glibc) | `manylinux_2_34_x86_64` | Vulkan + CPU | ubuntu-latest |
| Linux aarch64 (glibc) | `manylinux_2_34_aarch64` | Vulkan + CPU | ubuntu-24.04-arm |
| Linux x86_64 (musl) | `musllinux_1_2_x86_64` | CPU only | ubuntu-latest |
| Linux aarch64 (musl) | `musllinux_1_2_aarch64` | CPU only | ubuntu-24.04-arm |
| Windows x86_64 | `win_amd64` | Vulkan + CPU | windows-latest (MSVC) |
| Windows arm64 | `win_arm64` | CPU only | windows-11-arm (clang-cl) |

Each Linux wheel bundles `libgomp.so.1` (OpenMP runtime) and the Vulkan
loader, so users don't need to preinstall them. The Linux x86_64 glibc
wheel requires an x86_64-v3 CPU (AVX2/FMA/BMI2/F16C) — ggml-cpu enables
these by default at compile time. `auditwheel repair --disable-isa-ext-check`
bypasses the v1-baseline ISA audit; the wheel tag stays plain
`manylinux_2_34_x86_64` for pip compatibility, matching how numpy/scipy
ship v3-requiring wheels.

musllinux (Alpine) wheels are CPU-only because Alpine's shaderc 2024.4
fails to optimize ggml-vulkan's SPIR-V shaders (VUID-StandaloneSpirv-
None-10684). glibc-based Linux wheels include the Vulkan backend.

macOS x86_64 wheels are CPU-only because Apple deprecated Metal on
Intel Macs; the wheel ships no `libggml-metal.dylib`. macOS arm64
wheels include Metal by default.

Windows x86_64 uses MSVC (`ilammy/msvc-dev-cmd` GHA action + QtIF
silent install for LunarG Vulkan SDK). Windows arm64 uses `clang-cl`
on a native `windows-11-arm` runner — llama.cpp's ggml-cpu rejects
MSVC on ARM (CMakeLists.txt:106), so the override forces clang-cl
which VS's LLVM component provides.

## Install (development)

```bash
git clone --recurse-submodules https://github.com/johnking0099/mineru-llama-cpp.git
cd mineru-llama-cpp
pip install --no-build-isolation -e .
pip install -e ".[test]"
```

The first build compiles llama.cpp from source (a few minutes); subsequent
builds only recompile this library's own C++ files.

## Preparing GGUF models

`Engine` takes two local `.gguf` files: the main model and its multimodal
projector (mmproj). MinerU2.5 ships as HuggingFace safetensors, so you
convert it once with llama.cpp's own tooling. Both tools come with the
llama.cpp submodule; `llama-quantize` is also shipped in this package's
`bin/` directory after a build.

Pre-built Q8_0 GGUF files are available on
[ModelScope](https://www.modelscope.cn/models/jinzhenj/MinerU2.5-Pro-2605-1.2B-GGUF):
`MinerU2.5-Pro-2605-1.2B-Q8_0.gguf` (main model, 506MB) and
`mmproj-MinerU2.5-Pro-2605-1.2B-Q8_0.gguf` (mmproj, 677MB).

To convert from safetensors yourself:

**Main model** — convert to a BF16 GGUF, then quantize from that BF16 file
with `llama-quantize`:

```bash
# safetensors dir -> BF16 GGUF
python third_party/llama.cpp/convert_hf_to_gguf.py \
    /path/to/MinerU2.5-Pro-2605-1.2B \
    --outfile MinerU2.5-Pro-2605-1.2B.gguf \
    --outtype bf16

# BF16 GGUF -> quantized variants (pick what you need)
bin/llama-quantize MinerU2.5-Pro-2605-1.2B.gguf MinerU2.5-Pro-2605-1.2B-Q8_0.gguf   Q8_0
bin/llama-quantize MinerU2.5-Pro-2605-1.2B.gguf MinerU2.5-Pro-2605-1.2B-Q5_K_M.gguf Q5_K_M
bin/llama-quantize MinerU2.5-Pro-2605-1.2B.gguf MinerU2.5-Pro-2605-1.2B-Q4_K_M.gguf Q4_K_M
```

**mmproj (vision projector)** — convert directly from safetensors with
`--mmproj` at the desired precision (no separate `llama-quantize` step;
`--outtype` does the quantization here). The tool auto-prepends the
`mmproj-` prefix to the output filename:

```bash
# BF16 mmproj
python third_party/llama.cpp/convert_hf_to_gguf.py \
    /path/to/MinerU2.5-Pro-2605-1.2B \
    --mmproj \
    --outfile mmproj-MinerU2.5-Pro-2605-1.2B.gguf \
    --outtype bf16

# Q8_0 mmproj
python third_party/llama.cpp/convert_hf_to_gguf.py \
    /path/to/MinerU2.5-Pro-2605-1.2B \
    --mmproj \
    --outfile mmproj-MinerU2.5-Pro-2605-1.2B-Q8_0.gguf \
    --outtype q8_0
```

On macOS/Metal, use the **Q8_0** main model — BF16 crashes on Metal (see
Known issues). The mmproj can stay BF16 or Q8_0.

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

### Engine parameters

| Parameter | Default | Description |
|---|---|---|
| `n_ctx_seq` | 0 (= model training context) | Per-slot context length. Total KV = `n_ctx_seq × n_parallel`. |
| `n_gpu_layers` | 99 | Layers to offload to GPU. Set 0 to force CPU. |
| `n_parallel` | 4 | Concurrent slots. Tune to available GPU memory / CPU cores. |
| `verbosity` | LOG_LEVEL_WARN | Log threshold (LOG_LEVEL_INFO for backend info). |
| `n_threads` | -1 (= auto) | CPU threads. |

### Batch inference

```python
from mineru_llama_cpp import Engine
from mineru_vl_utils import MinerUClient

with Engine(model_path, mmproj_path, n_parallel=8) as engine:
    client = MinerUClient(backend="llama-cpp-engine", llama_cpp_engine=engine)
    results = client.batch_two_step_extract(images)
```

`batch_two_step_extract` dispatches requests concurrently across the
engine's slots (`batching_mode="concurrent"`).

## Cross-platform deployment

### Linux

```bash
# Standard build (CPU + Vulkan via GGML_BACKEND_DL, OpenMP ON)
uv pip install --no-build-isolation -e .

# Or with Vulkan explicitly:
SKBUILD_CMAKE_ARGS="-DGGML_VULKAN=ON" uv pip install --no-build-isolation -e .

# CUDA:
SKBUILD_CMAKE_ARGS="-DGGML_CUDA=ON" uv pip install --no-build-isolation -e .
```

With `GGML_BACKEND_DL=ON`, the Vulkan backend is a dlopen'd MODULE — if
`libvulkan.so.1` is missing, the engine gracefully falls back to CPU
without crashing. The `libgomp.so.1` OpenMP runtime is bundled into the
wheel (under `mineru_llama_cpp/lib/`), so users don't need it
preinstalled.

### Windows

```bash
# Requires MSVC Build Tools + vcvars64 in PATH
set CMAKE_GENERATOR=Ninja
uv pip install --no-build-isolation -e .
```

On Windows x86_64, MSVC compiles ggml-cpu directly. On Windows ARM64,
`ggml-cpu/CMakeLists.txt` rejects MSVC — use `clang-cl` instead:

```bash
set SKBUILD_CMAKE_ARGS=-DCMAKE_C_COMPILER=clang-cl;-DCMAKE_CXX_COMPILER=clang-cl
```

The `__init__.py` adds `bin/` to the DLL search path via
`os.add_dll_directory()` — Windows has no RPATH (`$ORIGIN`), so this is
needed for the `.pyd` to find its sibling DLLs.

On older Intel UHD iGPUs (pre-2020, 24-32 EU), Vulkan may be **slower**
than CPU for small models. Set `n_gpu_layers=0` to force CPU.

### macOS

No special flags needed — Metal is auto-detected. Use Q8_0 models (BF16
crashes on Metal). OpenMP is OFF (Metal GPU is the primary path).

## Build configuration

All build decisions are in the top-level `CMakeLists.txt` as forced cache
variables. The `patches/llama.cpp/*.patch` files are auto-applied during
CMake configure via `git apply` (requires the submodule to be a real git
checkout).

### OpenMP (libgomp) — platform-differential

- **Linux**: ON (default). ~14% faster than pthread fallback. `libgomp.so.1`
  bundled in the wheel as a real file (symlink resolved via `file(REAL_PATH)`).
  Verified in a `python:3.12-slim` container with no system libgomp.
- **macOS**: OFF. Metal is primary path; no libgomp on macOS.
- **Windows**: OFF. Zero perf delta on ARM64; removes `libomp140` dependency.

## Verifying the build

```bash
python -c "
from mineru_llama_cpp import Engine
with Engine('/path/to/model.gguf', '/path/to/mmproj.gguf') as engine:
    print(engine.generate([{'role': 'user', 'content': 'hello'}]).content)
"
```

## Known issues

See `docs/known-issues.md` — notably: **use Q8_0 models, not BF16**, on
Metal.
