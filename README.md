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

## Preparing GGUF models

`Engine` takes two local `.gguf` files: the main model and its multimodal
projector (mmproj). MinerU2.5 ships as HuggingFace safetensors, so you
convert it once with llama.cpp's own tooling. Both tools come with the
llama.cpp submodule; `llama-quantize` is also shipped in this package's
`bin/` directory after a build.

> The commands below were reconstructed from the conversion method used
> during development and verified against each tool's actual `--help`
> flags — they are not a verbatim transcript of the original run. Paths use
> `MinerU2.5-Pro-2605-1.2B` as the example; adjust to your model.

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

## Linux deployment

The project has been developed and tested on macOS (Metal backend). It has
no hosted git remote — moving it to a Linux machine means copying the
working tree directly rather than `git clone`-ing from a URL. This section
covers what changes on Linux and how to bring the project over.

### Copying the project to the target machine

There's no remote to `git clone` from, so bring the directory over as-is
(tar, rsync, scp — anything that preserves the tree). The one thing that
must survive the copy is git metadata: the top-level `CMakeLists.txt`
auto-applies `patches/llama.cpp/*.patch` via `git apply` before building
(see the "Apply patches" block in that file), and that requires
`third_party/llama.cpp` to still be a real git checkout with its `.git`
metadata intact — not just the working-tree files. A plain `.git`-inclusive
copy handles this correctly:

```bash
# from the machine that has the working tree
tar --exclude=build --exclude='*.so' -czf mineru-llama-cpp.tar.gz mineru-llama-cpp/

# on the target Linux machine
tar -xzf mineru-llama-cpp.tar.gz
cd mineru-llama-cpp
```

(`tar` includes `.git` and `third_party/llama.cpp/.git` by default — don't
add `--exclude=.git`.) The target machine needs `git` installed for the
patch-application step to run during CMake configure, even though nothing
is being cloned from a remote.

If you'd rather clone properly: push this repo to any git host you control,
then on the target machine `git clone --recurse-submodules <url>` (recurse
is required so `third_party/llama.cpp` comes down populated, not as an
empty directory).

### Building with CUDA vs. CPU-only

The build supports both without any code changes — `GGML_CUDA` is a
standard llama.cpp CMake option that defaults to `OFF`. Toggle it through
`scikit-build-core`'s standard `SKBUILD_CMAKE_ARGS` environment variable
(no `pyproject.toml` changes needed):

```bash
# GPU build (requires the CUDA Toolkit installed on the target machine;
# first build takes noticeably longer than CPU-only, since llama.cpp's CUDA
# kernels get compiled)
SKBUILD_CMAKE_ARGS="-DGGML_CUDA=ON" uv pip install --no-build-isolation -e .

# CPU-only build (no flag needed — this is the default)
uv pip install --no-build-isolation -e .
```

Both produce the same `Engine` API; `n_gpu_layers` (default 99, meaning
"put everything on the GPU") is honored by whichever backend actually got
compiled in. On a CPU-only build llama.cpp just logs a warning and runs on
CPU — no code branches on which backend is active.

### Verifying the build

Before running anything at scale, confirm the engine loads and generates:

```bash
.venv/bin/python -c "
from mineru_llama_cpp import Engine
with Engine('/path/to/model.gguf', '/path/to/mmproj.gguf') as engine:
    print(engine.generate([{'role': 'user', 'content': 'hello'}]).content)
"
```

or run the existing test suite (`pytest`, after pointing `tests/conftest.py`'s
`MODEL`/`MMPROJ` constants at paths that exist on the target machine — they
default to this project's development machine's paths).

### Batch inference starting point

For running a large image set (e.g. an OmniDocBench-style accuracy pass)
through this engine, the pieces to combine are:

```python
from mineru_llama_cpp import Engine
from mineru_vl_utils import MinerUClient

# n_parallel: tune to available GPU memory / CPU cores -- concurrent
# requests share one unified KV cache pool across all slots, not a fixed
# per-slot split (see Engine's n_ctx/n_parallel docstring)
with Engine(model_path, mmproj_path, n_parallel=8) as engine:
    client = MinerUClient(backend="llama-cpp-engine", llama_cpp_engine=engine)
    results = client.batch_two_step_extract(images)  # list[Image.Image] -> list[ExtractResult]
```

`batch_two_step_extract` dispatches requests concurrently across the
engine's slots (see `mineru_vl_utils`'s `LlamaCppEngineVlmClient`, backend
`"llama-cpp-engine"`, `batching_mode="concurrent"`). Wiring this up against
a specific dataset layout (e.g. OmniDocBench's directory structure) and
scoring against ground truth is outside this library's scope — that's
where an existing OmniDocBench evaluation script takes over.

### Linux validation checklist (first-time bring-up)

This project has only ever been built and run on macOS (Metal). Everything
Linux-related below is **unverified on real hardware** — the code paths
exist and pass static review, but no one has run them on a Linux box yet.
Work through this checklist on the target Linux server (has an NVIDIA GPU,
so the Vulkan loader is present) and record the outcome of each step. Do
them in order; a later step assuming an earlier one passed.

Prerequisites on the target machine: `git`, CMake, a C++17 compiler, Python
3.10–3.13, the Vulkan SDK/headers (build-time) — plus the NVIDIA driver's
Vulkan loader (already there on a GPU box). You also need the GGUF model +
mmproj files present locally (see "Preparing GGUF models" above); note their
paths, referred to below as `$MODEL` and `$MMPROJ`.

**Step 1 — CPU-only build succeeds and the patch auto-applies.**
```bash
uv pip install --no-build-isolation -e . 2>&1 | tee /tmp/build-cpu.log
grep -i "llama.cpp patch" /tmp/build-cpu.log   # optional: confirm patch step ran
python -c "from mineru_llama_cpp import Engine; print('import OK')"
```
Pass criteria: build exits 0; `import` prints `import OK`. The top-level
`CMakeLists.txt` applies `patches/llama.cpp/*.patch` via `git apply` during
configure — if it fails with "patch does not apply", the submodule isn't a
real git checkout (see "Copying the project" above about keeping `.git`).

**Step 2 — RPATH is correct on Linux (this is the main cross-platform fix
to confirm).** The `.so` and its sibling `libllama.so`/`libggml*.so` must be
found at runtime via `$ORIGIN` (Linux), not the macOS `@loader_path`:
```bash
python - <<'PY'
import mineru_llama_cpp, pathlib
so = pathlib.Path(mineru_llama_cpp.__file__).parent / "_mineru_llama_cpp.cpython-*-linux-gnu.so"
import glob; so = glob.glob(str(so))[0]
print("so:", so)
PY
# inspect the RPATH baked into the built .so:
readelf -d $(python -c "import glob,mineru_llama_cpp,pathlib,os; d=pathlib.Path(mineru_llama_cpp.__file__).parent; print(glob.glob(os.path.join(str(d),'_mineru_llama_cpp*.so'))[0])") | grep -iE "RPATH|RUNPATH"
```
Pass criteria: `RUNPATH`/`RPATH` contains `$ORIGIN` (for an editable build
it may instead point at the build tree via an absolute path — that's the
`BUILD_RPATH`, also fine for editable installs; the `$ORIGIN` form is what
matters for a wheel, verified in Step 6). A real end-to-end generate in
Step 3 is the ultimate proof it resolves.

**Step 3 — CPU inference actually runs end to end.**
```bash
python - <<PY
from mineru_llama_cpp import Engine
with Engine("$MODEL", "$MMPROJ") as e:
    print(e.generate([{"role":"user","content":"hi"}]).content)
PY
```
Pass criteria: prints generated text, no `cannot open shared object file`,
no crash.

**Step 4 — Vulkan build succeeds and the GPU is actually used.**
```bash
SKBUILD_CMAKE_ARGS="-DGGML_VULKAN=ON" uv pip install --no-build-isolation -e . 2>&1 | tee /tmp/build-vk.log
python - <<PY
from mineru_llama_cpp import Engine, LOG_LEVEL_INFO
with Engine("$MODEL", "$MMPROJ", verbosity=LOG_LEVEL_INFO) as e:
    print(e.generate([{"role":"user","content":"hi"}]).content)
PY
```
Pass criteria: build exits 0; stderr shows a Vulkan device being enumerated
/ selected (look for `Vulkan`/`ggml_vulkan` lines); generate produces text.
(Default `verbosity` is `WARN` and hides this — that's why `LOG_LEVEL_INFO`
is passed here.)

**Step 5 — A1: the same Vulkan build gracefully falls back to CPU when no
Vulkan runtime is available (THE critical assumption — a single "CPU+Vulkan"
wheel must serve both GPU and no-GPU machines).** Source review says yes,
but this is the step that proves it on hardware. Simulate "no Vulkan" two
ways without touching the system:
```bash
# way A: ggml's own env var, skips Vulkan registration entirely
GGML_DISABLE_VULKAN=1 python - <<PY
from mineru_llama_cpp import Engine, LOG_LEVEL_INFO
with Engine("$MODEL", "$MMPROJ", verbosity=LOG_LEVEL_INFO) as e:
    print(e.generate([{"role":"user","content":"hi"}]).content)
PY

# way B: point the Vulkan loader at a nonexistent ICD (simulates missing driver)
VK_ICD_FILENAMES=/nonexistent.json python - <<PY
from mineru_llama_cpp import Engine, LOG_LEVEL_INFO
with Engine("$MODEL", "$MMPROJ", verbosity=LOG_LEVEL_INFO) as e:
    print(e.generate([{"role":"user","content":"hi"}]).content)
PY
```
Pass criteria: BOTH ways produce text, do not crash, and the log shows CPU
being used (no Vulkan device). If either crashes/aborts instead of falling
back, A1 is FALSE and the single-wheel-covers-both plan needs rethinking —
flag this loudly.

**Step 6 — wheel builds and packages the right RPATH.** Confirms the
distributable artifact (not just the editable install) is self-contained:
```bash
mkdir -p /tmp/wheel-out
SKBUILD_CMAKE_ARGS="-DGGML_VULKAN=ON" uv build --wheel --no-build-isolation -o /tmp/wheel-out
cd /tmp/wheel-out && python -m zipfile -e mineru_llama_cpp-*.whl extracted/
readelf -d extracted/mineru_llama_cpp/_mineru_llama_cpp*.so | grep -iE "RPATH|RUNPATH"
```
Pass criteria: wheel builds; the packaged `.so`'s `RUNPATH`/`RPATH` is
`$ORIGIN/../lib` (Linux form). Bundled `lib/*.so` and `bin/llama-server`
should be present in the extracted tree; `include/`, `lib/cmake/`,
`lib/pkgconfig/` should be absent (excluded via `wheel.exclude`).

**Step 7 — full test suite (optional but recommended).** The suite is
GPU/model-heavy (~6 min on Metal) and hardcodes model paths in
`tests/conftest.py` (`MODEL`/`MMPROJ` point at the dev machine's paths).
Edit those to the target machine's `$MODEL`/`$MMPROJ` first, then:
```bash
python -m pytest --tb=short
```
Pass criteria: all tests pass (the `test_concurrency.py` queueing regression
test in particular exercises the multi-slot path).

**Report back**: for each step, whether it passed and any stderr worth
noting — especially Step 5 (A1) and Steps 2/6 (RPATH), since those are the
Linux-specific unknowns this checklist exists to close.

## Known issues

See `docs/known-issues.md` — notably: **use Q8_0 models, not BF16**, on
Metal.

## Status

v0.1.0 — local development package only. No wheel/CI packaging, no
HuggingFace auto-download, no batch-generate API, no Vulkan testing. CUDA
support relies on llama.cpp's own standard `GGML_CUDA` CMake option (see
"Linux deployment" above) but has not yet been built or run on real GPU
hardware — this project has only been built/tested on macOS (Metal) so far.
See the design spec's §1 "非目标" for the full non-goals list.
