# mineru-llama-cpp Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `mineru-llama-cpp` Python library — a pybind11-wrapped, in-process llama.cpp VLM inference engine exposing a single `Engine` class with sync+async generate/stream methods, per the approved design at `docs/superpowers/specs/2026-08-03-mineru-llama-cpp-engine-design.md`.

**Architecture:** New independent git repo with llama.cpp as a pinned submodule. Three layers: `EngineCore` (pure C++, no pybind11 dependency, reuses the proven Path-A/oaicompat logic from the technical-validation spike) → `binding.cpp` (thin pybind11 layer: GIL release/acquire, exception mapping, dict marshalling) → `engine.py` (Python-facing `Engine` class with sync+async methods, dataclass results). Built via scikit-build-core + CMake.

**Tech Stack:** C++17, pybind11, llama.cpp (commit `9a3bf2b84`, Metal backend), Python 3.10+, scikit-build-core, pytest.

**Reference material (read these first if resuming mid-plan):**
- Design spec: `docs/superpowers/specs/2026-08-03-mineru-llama-cpp-engine-design.md`
- Proven spike code (source of truth for the C++ logic being ported): `cpp/mineru_engine_spike.cpp`, `cpp/CMakeLists.txt` (in the current `mineru-vl-engine` directory, to be migrated)
- Model files used by all tests: `/Users/jinzhenj/.mineru/models/MinerU2.5-Pro-2605-1.2B-Q8_0.gguf` + `/Users/jinzhenj/.mineru/models/mmproj-MinerU2.5-Pro-2605-1.2B-Q8_0.gguf` (Q8_0 — **never use the BF16 variants, they SIGSEGV under Metal in a .so context, see `docs/known-issues.md` created in Task 3**)

---

## Phase A: Repository & Build Skeleton (Tasks 1-6)

### Task 1: Initialize the new repository and directory skeleton

**Files:**
- Create (directory): `/Users/jinzhenj/MinerU-Repo/mineru-llama-cpp/` (new git repo root)

- [ ] **Step 1: Create the directory and initialize git**

```bash
mkdir -p /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git init
```

Expected: `Initialized empty Git repository in /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp/.git/`

- [ ] **Step 2: Create the directory skeleton**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
mkdir -p src/mineru_llama_cpp src/cpp tests/legacy_spike tests/fixtures docs
```

- [ ] **Step 3: Create a `.gitignore`**

```bash
cat > .gitignore <<'EOF'
build/
*.so
*.dylib
__pycache__/
*.egg-info/
.venv/
*.pyc
.pytest_cache/
EOF
```

- [ ] **Step 4: Commit the skeleton**

```bash
git add .gitignore
git commit -m "chore: initialize repository skeleton"
```

Expected: commit succeeds (there may be nothing else to add yet since empty dirs aren't tracked by git — that's fine, `.gitignore` alone is a valid first commit).

---

### Task 2: Add llama.cpp as a pinned git submodule

**Files:**
- Create: `.gitmodules`
- Create (via submodule): `third_party/llama.cpp/` (full clone, checked out at commit `9a3bf2b84`)

- [ ] **Step 1: Add the submodule**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git submodule add https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
```

Expected: clones the full llama.cpp repo into `third_party/llama.cpp/`, creates `.gitmodules`. This will take a few minutes (large repo).

- [ ] **Step 2: Pin to the validated commit**

```bash
cd third_party/llama.cpp
git checkout 9a3bf2b84
cd ../..
```

Expected: `HEAD is now at 9a3bf2b84 server : add extra trace log for prompt similarity (#26218)` (or similar — this is the exact commit already validated in the technical-validation phase; do not use a newer commit without re-running the validation suite).

- [ ] **Step 3: Verify the pin and commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git -C third_party/llama.cpp rev-parse HEAD
```

Expected output starts with `9a3bf2b84`.

```bash
git add .gitmodules third_party/llama.cpp
git commit -m "build: add llama.cpp submodule pinned to 9a3bf2b84"
```

---

### Task 3: Migrate technical-validation spike code into `tests/legacy_spike/`

**Files:**
- Create: `tests/legacy_spike/mineru_engine_spike.cpp`
- Create: `tests/legacy_spike/metal_probe.mm`
- Create: `tests/legacy_spike/test_async_spike.py`, `test_async_text.py`, `test_concurrency.py`, `test_determinism.py`, `test_error_isolation.py`, `test_lifecycle.py`, `test_streaming.py`
- Create: `tests/legacy_spike/README.md`
- Create: `docs/known-issues.md`

- [ ] **Step 1: Copy the spike C++ and Python files verbatim**

```bash
cd /Users/jinzhenj/MinerU-Repo
cp mineru-vl-engine/cpp/mineru_engine_spike.cpp mineru-llama-cpp/tests/legacy_spike/
cp mineru-vl-engine/cpp/metal_probe.mm mineru-llama-cpp/tests/legacy_spike/
cp mineru-vl-engine/cpp/test_async_spike.py mineru-llama-cpp/tests/legacy_spike/
cp mineru-vl-engine/cpp/test_async_text.py mineru-llama-cpp/tests/legacy_spike/
cp mineru-vl-engine/cpp/test_concurrency.py mineru-llama-cpp/tests/legacy_spike/
cp mineru-vl-engine/cpp/test_determinism.py mineru-llama-cpp/tests/legacy_spike/
cp mineru-vl-engine/cpp/test_error_isolation.py mineru-llama-cpp/tests/legacy_spike/
cp mineru-vl-engine/cpp/test_lifecycle.py mineru-llama-cpp/tests/legacy_spike/
cp mineru-vl-engine/cpp/test_streaming.py mineru-llama-cpp/tests/legacy_spike/
```

- [ ] **Step 2: Write a README explaining the archive's purpose**

Create `tests/legacy_spike/README.md`:

```markdown
# Legacy technical-validation spike

This directory archives the C++/Python code written during the technical
validation phase (before `mineru-llama-cpp` existed as a proper package).
It proved: lib-mode feasibility, the JSON/oaicompat input path ("Path A"),
the async/GIL bridge model, multi-slot concurrency, streaming, error
isolation, long-lived lifecycle, and temp=0 determinism.

**Do not build or run these files as part of the normal test suite.** They
reference a different module name (`mineru_engine_spike`, not
`mineru_llama_cpp`) and a different CMake setup (linking prebuilt libraries
from `../../../llama.cpp-build/llama.cpp/build` rather than the
`third_party/llama.cpp` submodule). They are kept read-only for reference
and as a regression baseline if the real implementation's behavior is ever
in doubt — re-run the corresponding spike script and compare.

See `docs/superpowers/specs/2026-08-03-mineru-llama-cpp-engine-design.md`
(in the original `mineru-vl-engine` project) for the full validation history.
```

- [ ] **Step 3: Write `docs/known-issues.md`**

```markdown
# Known Issues

## BF16 models SIGSEGV under Metal when loaded from a Python extension (.so)

**Symptom:** Loading a BF16 GGUF model (main model or mmproj) with the Metal
backend enabled, from inside a pybind11-built `.so` (i.e. via
`import mineru_llama_cpp`), crashes with SIGSEGV. The same model loads fine
from a standalone C++ executable on the same machine.

**Root cause (confirmed via direct Metal API probing):** Metal's shader
compiler does not instantiate certain MSL `[[host_name(...)]]` template
specializations (the BF16 matmul kernels, e.g.
`kernel_mul_mv_ext_bf16_f32_r1_2`) when the compiling process is a `dlopen`ed
extension module rather than the main executable. This is a Metal framework
bug, not a bug in llama.cpp or in this library. It matches upstream report
[ggml-org/llama.cpp#21381](https://github.com/ggml-org/llama.cpp/issues/21381)
(closed as not planned).

**Workaround:** Use Q8_0 (or any non-BF16 quantization) for both the main
model and the mmproj. Q8_0 kernels do not use MSL template specialization and
load correctly in every context tested. Q8_0 is also the quantization
recommended on quality/performance grounds independent of this bug (see the
technical-validation phase's `QUANT-BENCHMARK.md`).

**Status:** Not planned to be worked around in this library — see
design spec §1 "非目标". Not applicable on non-Metal backends (CPU/CUDA/Vulkan
are untested but the failure mode is Metal-specific by construction).
```

- [ ] **Step 4: Commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add tests/legacy_spike docs/known-issues.md
git commit -m "docs: archive technical-validation spike code and known issues"
```

---

### Task 4: Write the top-level `CMakeLists.txt`

**Files:**
- Create: `CMakeLists.txt`

- [ ] **Step 1: Write the file**

```cmake
cmake_minimum_required(VERSION 3.15)
project(mineru_llama_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- llama.cpp submodule: force-enable only what we need ---
#
# LLAMA_STANDALONE becomes OFF when llama.cpp is included via
# add_subdirectory() (its CMAKE_SOURCE_DIR check fails, since we are the
# top-level project). That flips ALL of the following option() defaults to
# OFF (they each default to ${LLAMA_STANDALONE}). We must force them ON as
# CACHE variables *before* add_subdirectory() runs llama.cpp's own
# CMakeLists.txt, so its option() calls see pre-existing cache values and
# leave them alone (standard CMake option()-override pattern).
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)   # server-context depends on the common utils lib
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)    # gates add_subdirectory(tools) in llama.cpp's CMakeLists.txt
set(LLAMA_BUILD_SERVER ON CACHE BOOL "" FORCE)   # gates add_subdirectory(tools/server) -> defines the server-context target
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)      # skip web UI asset fetch; we never expose HTTP

add_subdirectory(third_party/llama.cpp)
add_subdirectory(src/cpp)
```

- [ ] **Step 2: Commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add CMakeLists.txt
git commit -m "build: add top-level CMakeLists.txt wiring the llama.cpp submodule"
```

---

### Task 5: Write `pyproject.toml`

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write the file**

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "pybind11"]
build-backend = "scikit_build_core.build"

[project]
name = "mineru-llama-cpp"
version = "0.1.0"
description = "In-process llama.cpp VLM inference engine for MinerU"
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
test = ["pytest>=7.0", "pillow"]

[tool.scikit-build]
cmake.build-type = "Release"
wheel.packages = ["src/mineru_llama_cpp"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
```

Note: `pillow` is a **test-only** dependency (used solely by `tests/conftest.py` in Task 21 to generate the fixture image at test-collection time). It is not a runtime dependency of `mineru_llama_cpp` itself — the `[project] dependencies = []` line, matching design spec §6's dependency-baseline decision, is unaffected.

- [ ] **Step 2: Commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add pyproject.toml
git commit -m "build: add pyproject.toml (scikit-build-core + pybind11)"
```

---

### Task 6: Build-wiring smoke test (minimal pybind11 module, before real logic)

**Goal of this task:** catch any CMake/scikit-build-core wiring mistakes now, with a trivial module, before Phase B adds the real (much harder to debug) engine logic on top.

**Files:**
- Create: `src/cpp/CMakeLists.txt`
- Create: `src/cpp/binding.cpp` (placeholder — will be replaced wholesale in Task 12)
- Create: `src/mineru_llama_cpp/__init__.py` (placeholder — will be replaced wholesale in Task 20)

- [ ] **Step 1: Write a placeholder `src/cpp/binding.cpp`**

```cpp
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_mineru_llama_cpp, m) {
    m.def("ping", []() { return "pong"; });
}
```

- [ ] **Step 2: Write `src/cpp/CMakeLists.txt`**

```cmake
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_mineru_llama_cpp binding.cpp)

install(TARGETS _mineru_llama_cpp DESTINATION mineru_llama_cpp)
```

- [ ] **Step 3: Write a placeholder `src/mineru_llama_cpp/__init__.py`**

```python
from ._mineru_llama_cpp import ping

__all__ = ["ping"]
```

- [ ] **Step 4: Build via editable install**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
/Users/jinzhenj/MinerU-Repo/.venv/bin/pip install --no-build-isolation -e . -v
```

Note: `--no-build-isolation` is used because `scikit-build-core` and `pybind11` are already available in the shared `.venv` (installed during the technical-validation phase); this avoids re-downloading/re-resolving a fresh build environment every iteration. Expected: build succeeds, ends with `Successfully installed mineru-llama-cpp-0.1.0`. This first build will take several minutes (llama.cpp compiles from source, including Metal shaders) — this is expected, not a hang.

- [ ] **Step 5: Verify the smoke test**

```bash
/Users/jinzhenj/MinerU-Repo/.venv/bin/python -c "import mineru_llama_cpp; print(mineru_llama_cpp.ping())"
```

Expected: `pong`

- [ ] **Step 6: Commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add src/cpp/CMakeLists.txt src/cpp/binding.cpp src/mineru_llama_cpp/__init__.py
git commit -m "build: verify pybind11 + scikit-build-core wiring with a smoke-test module"
```

**If this task fails:** do not proceed to Phase B until `ping()` works. Common failure modes and fixes:
- `Could not find pybind11` → confirm `pybind11` is installed in the venv used by pip (`/Users/jinzhenj/MinerU-Repo/.venv/bin/pip show pybind11`); if missing, `/Users/jinzhenj/MinerU-Repo/.venv/bin/pip install pybind11` first, then retry with `--no-build-isolation`.
- CMake configure errors about `LLAMA_BUILD_*` — re-check Task 4's CMakeLists.txt content matches exactly (the CACHE FORCE lines must run before `add_subdirectory(third_party/llama.cpp)`).
- Long hang during build — expected for the first build (full llama.cpp compile); only investigate if it exceeds ~15 minutes on this machine.

---

## Phase B: C++ Core Layer (Tasks 7-10)

**Design principle for this phase (design spec §4.1):** `engine_core.h`/`.cpp` must have **zero pybind11 dependency**. No `#include <pybind11/...>`, no `py::` anywhere in these two files. GIL handling lives entirely in `binding.cpp` (Phase C). This is a deliberate improvement over the spike (which mixed GIL calls into the `Engine` class body) — the whole point of splitting core/binding is that `EngineCore` could theoretically be reused by a non-Python binding later.

**Correctness note carried forward from design analysis (do not skip):** `server_response_reader` (declared in `third_party/llama.cpp/tools/server/server-queue.h`) has a user-declared destructor that calls `stop()`, which **cancels any in-flight task** associated with its `id_tasks` if the task hasn't finished yet. Because of the user-declared destructor, the class has no implicit move constructor. This makes it dangerous to construct a `server_response_reader` in a local variable, post a task on it, and then try to relocate it into a returned object — the original local's destructor fires when the function returns, sees the task still in flight, and cancels it out from under the caller. Task 10 uses `std::unique_ptr<server_response_reader>` specifically to avoid ever creating a second copy of a *populated* reader; the single heap-allocated instance is mutated in place via pointer method calls and only the pointer itself (not the object) is ever moved. Do not "simplify" this to a by-value `server_response_reader` member — it will compile but break streaming at runtime (the generation gets silently cancelled immediately after `generate_stream()` returns).

### Task 7: Write `engine_core.h`

**Files:**
- Create: `src/cpp/engine_core.h`

- [ ] **Step 1: Write the file**

```cpp
// engine_core.h — pure C++ engine core. NO pybind11 dependency here; all
// GIL handling happens in binding.cpp. See design spec §4.1/§4.2.
#pragma once

#include "server-context.h"
#include "server-queue.h"
#include "server-task.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

class EngineCore {
public:
    EngineCore(const std::string & model_path, const std::string & mmproj_path,
               int n_ctx, int n_gpu_layers, int n_parallel);
    ~EngineCore();

    EngineCore(const EngineCore &) = delete;
    EngineCore & operator=(const EngineCore &) = delete;

    struct Timings {
        int32_t prompt_n            = 0;
        double  prompt_ms           = 0.0;
        double  prompt_per_second   = 0.0;
        int32_t predicted_n         = 0;
        double  predicted_ms        = 0.0;
        double  predicted_per_second = 0.0;
    };

    // Result of a single non-streaming generate() call.
    //
    // Error handling contract (design spec §4.2 point 3):
    //   - Errors detected *before* the task is posted (malformed body JSON,
    //     bad image_url, invalid sampling field) are raised as regular C++
    //     exceptions (std::exception) that propagate OUT of generate()/
    //     generate_stream() uncaught. The binding layer catches these.
    //   - Errors detected *after* posting (server-side send_error, e.g.
    //     image decode failure discovered on the loop thread, or
    //     context-size-exceeded discovered during slot processing) arrive
    //     asynchronously via the normal result channel and are surfaced via
    //     is_error/error_json on this struct — there is no C++ call stack to
    //     throw up at that point, so a raw exception is not possible.
    struct GenerateResult {
        bool        is_error = false;
        std::string error_json;   // valid only if is_error; JSON string, see server-common.cpp format_error_response()

        std::string content;
        std::string finish_reason;   // "stop" | "length"; valid only if !is_error
        int32_t     tokens_evaluated = 0;
        int32_t     tokens_predicted = 0;
        Timings     timings;
    };

    // Blocks the calling thread until the completion is fully generated.
    // May throw std::exception for pre-post errors (see contract above).
    GenerateResult generate(const std::string & body_json);

    // One chunk of a streaming generation.
    struct Chunk {
        bool        is_error = false;
        std::string error_json;

        std::string delta;
        bool        is_final = false;
        // valid only when is_final (and !is_error):
        std::string finish_reason;
        int32_t     tokens_evaluated = 0;
        int32_t     tokens_predicted = 0;
        Timings     timings;
    };

    // Owns the one server_response_reader for this stream's entire
    // lifetime. See the correctness note above this class — do NOT change
    // rd_ to a by-value server_response_reader member.
    class StreamHandle {
    public:
        explicit StreamHandle(std::unique_ptr<server_response_reader> rd);
        // Blocks until the next token (or the final chunk, or an error) is
        // ready. May throw std::exception for pre-post errors, but in
        // practice those all happen inside generate_stream() before a
        // StreamHandle is ever constructed, so next_chunk() itself should
        // only ever report async errors via Chunk::is_error.
        Chunk next_chunk();

    private:
        std::unique_ptr<server_response_reader> rd_;
        bool done_ = false;
    };

    // May throw std::exception for pre-post errors (see contract above).
    StreamHandle generate_stream(const std::string & body_json);

private:
    server_context                        ctx_;
    common_params                          params_;
    std::unique_ptr<server_context_meta>  meta_;
    std::thread                           loop_thread_;
    bool                                  started_ = false;

    // Guards oaicompat_chat_params_parse() + eval_llama_cmpl_schema() +
    // post_task() for both generate() and generate_stream(). Fixes the
    // "concurrent generate() calls corrupt each other's prompt" bug found
    // during technical validation (Tier2 #3). Deliberately does NOT guard
    // the blocking wait for a result — multiple slots must be able to decode
    // in parallel. See design spec §4.2 point 1.
    std::mutex parse_mu_;
};
```

- [ ] **Step 2: Commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add src/cpp/engine_core.h
git commit -m "feat: add EngineCore class declaration"
```

---

### Task 8: Write `engine_core.cpp` — constructor, destructor, shared helpers

**Files:**
- Create: `src/cpp/engine_core.cpp`

- [ ] **Step 1: Write the file (part 1 — includes, helpers, constructor, destructor)**

```cpp
// engine_core.cpp
#include "engine_core.h"

#include "server-common.h"
#include "server-schema.h"
#include "common.h"
#include "llama.h"

#include <stdexcept>
#include <utility>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

// Maps llama-server's stop_type string (from server-task.cpp
// stop_type_to_str(): "eos"/"word"/"limit"/"none") onto the two
// OpenAI-style finish_reason values this library exposes. "eos"/"word" mean
// the model stopped itself (matches oaicompat's own to_json_oaicompat()
// mapping); "limit" means max_tokens was hit. Any other/unrecognized value
// (including "none", which should not occur for a genuinely-finished
// result) defaults to "stop" rather than an empty string, so the Python-side
// Literal["stop", "length"] contract is never violated.
std::string map_finish_reason(const std::string & stop_type_str) {
    if (stop_type_str == "limit") return "length";
    return "stop";
}

EngineCore::Timings extract_timings(const json & j) {
    EngineCore::Timings t;
    if (!j.contains("timings")) return t;
    const auto & tj = j.at("timings");
    t.prompt_n             = tj.value("prompt_n", 0);
    t.prompt_ms            = tj.value("prompt_ms", 0.0);
    t.prompt_per_second    = tj.value("prompt_per_second", 0.0);
    t.predicted_n          = tj.value("predicted_n", 0);
    t.predicted_ms         = tj.value("predicted_ms", 0.0);
    t.predicted_per_second = tj.value("predicted_per_second", 0.0);
    return t;
}

} // namespace

EngineCore::EngineCore(const std::string & model_path, const std::string & mmproj_path,
                        int n_ctx, int n_gpu_layers, int n_parallel) {
    params_.model.path         = model_path;
    params_.mmproj.path        = mmproj_path;
    params_.n_gpu_layers       = n_gpu_layers;
    params_.mmproj_use_gpu     = (n_gpu_layers > 0);
    params_.n_parallel         = n_parallel;
    params_.n_ctx              = n_ctx;
    params_.cont_batching      = true;
    params_.special            = true;
    params_.sleep_idle_seconds = -1;  // never sleep (else model gets unloaded mid-use)
    params_.chat_template      = "";  // use the model's own built-in template
    params_.use_jinja          = true;

    llama_backend_init();
    llama_numa_init(params_.numa);

    if (!ctx_.load_model(params_)) {
        llama_backend_free();
        throw std::runtime_error("load_model failed");
    }
    meta_ = std::make_unique<server_context_meta>(ctx_.get_meta());
    loop_thread_ = std::thread([this] { ctx_.start_loop(); });
    started_ = true;
}

EngineCore::~EngineCore() {
    if (started_) {
        ctx_.terminate();
        if (loop_thread_.joinable()) loop_thread_.join();
    }
    llama_backend_free();
}
```

Note vs. the spike: added `llama_backend_free()` on the `load_model` failure path (the spike didn't need this because it never tested load failure; a clean throw here avoids leaking the backend if construction fails).

- [ ] **Step 2: Commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add src/cpp/engine_core.cpp
git commit -m "feat: EngineCore constructor/destructor + finish_reason/timings helpers"
```

Note: this file is not yet complete (missing `generate()`/`generate_stream()`) and will not compile standalone yet — that's expected, Tasks 9-10 add the rest before the next build attempt in Task 14.

---

### Task 9: Write `engine_core.cpp` — `generate()`

**Files:**
- Modify: `src/cpp/engine_core.cpp` (append)

- [ ] **Step 1: Append the `generate()` implementation**

Add to the end of `src/cpp/engine_core.cpp`:

```cpp
EngineCore::GenerateResult EngineCore::generate(const std::string & body_json) {
    json body = json::parse(body_json);  // throws nlohmann::json::parse_error on malformed input;
                                          // propagates uncaught to the binding layer (design spec §4.2 point 3)

    server_response_reader rd = ctx_.get_response_reader();
    server_task task(SERVER_TASK_TYPE_COMPLETION);

    {
        std::lock_guard<std::mutex> lk(parse_mu_);
        task.id = rd.get_new_id();

        std::vector<raw_buffer> files;
        // Reuses llama-server's own OpenAI-compatible chat parsing ("Path A"):
        // renders the jinja chat template, extracts image_url data URIs into
        // files, and normalizes sampling params into `parsed`. Throws
        // std::runtime_error/std::invalid_argument on bad image_url etc.
        json parsed = oaicompat_chat_params_parse(body, meta_->chat_params, files);
        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_.get_llama_context()));
        task.params = server_schema::eval_llama_cmpl_schema(
            vocab, params_, meta_->slot_n_ctx, meta_->logit_bias_eog, parsed);
        task.params.stream       = false;
        task.params.res_type     = TASK_RESPONSE_TYPE_NONE;
        task.params.cache_prompt = false;  // don't reuse KV across requests (fixes Tier2 #3's "empty prompt" bug)
        task.cli        = true;
        task.cli_prompt = parsed.at("prompt").get<std::string>();
        task.cli_files  = std::move(files);
    }

    rd.post_task(std::move(task));

    GenerateResult result;
    auto r = rd.next([]() { return false; });  // blocks until the result arrives; no GIL concept at this layer
    if (!r) {
        result.is_error   = true;
        result.error_json = R"({"type":"server_error","message":"no result (stopped)"})";
        return result;
    }
    if (r->is_error()) {
        result.is_error   = true;
        result.error_json = r->to_json().dump();
        return result;
    }

    json j = r->to_json();
    result.content         = j.value("content", std::string());
    result.finish_reason    = map_finish_reason(j.value("stop_type", std::string()));
    result.tokens_evaluated = j.value("tokens_evaluated", 0);
    result.tokens_predicted = j.value("tokens_predicted", 0);
    result.timings          = extract_timings(j);
    return result;
}
```

- [ ] **Step 2: Commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add src/cpp/engine_core.cpp
git commit -m "feat: EngineCore::generate()"
```

---

### Task 10: Write `engine_core.cpp` — `generate_stream()` and `StreamHandle`

**Files:**
- Modify: `src/cpp/engine_core.cpp` (append)

- [ ] **Step 1: Append the `StreamHandle` and `generate_stream()` implementation**

Add to the end of `src/cpp/engine_core.cpp`:

```cpp
EngineCore::StreamHandle::StreamHandle(std::unique_ptr<server_response_reader> rd)
    : rd_(std::move(rd)) {}

EngineCore::Chunk EngineCore::StreamHandle::next_chunk() {
    Chunk chunk;
    if (done_) {
        // Caller kept calling after the final chunk; report as an immediate
        // final (empty-delta) chunk rather than blocking on a reader that
        // has nothing left to give.
        chunk.is_final = true;
        chunk.finish_reason = "stop";
        return chunk;
    }

    auto r = rd_->next([]() { return false; });
    if (!r) {
        chunk.is_error   = true;
        chunk.error_json = R"({"type":"server_error","message":"no result (stopped)"})";
        done_ = true;
        return chunk;
    }
    if (r->is_error()) {
        chunk.is_error   = true;
        chunk.error_json = r->to_json().dump();
        done_ = true;
        return chunk;
    }

    // The first partial chunk (is_begin, an SSE-header signal with no
    // content) serializes to a null json, not an object — see server-task.cpp
    // server_task_result_cmpl_partial::to_json(). Treat it as an empty delta.
    json j = r->to_json();
    if (!j.is_null()) {
        chunk.delta = j.value("content", std::string());
    }

    if (r->is_stop()) {
        chunk.is_final          = true;
        chunk.finish_reason     = map_finish_reason(j.is_null() ? std::string() : j.value("stop_type", std::string()));
        chunk.tokens_evaluated  = j.is_null() ? 0 : j.value("tokens_evaluated", 0);
        chunk.tokens_predicted  = j.is_null() ? 0 : j.value("tokens_predicted", 0);
        chunk.timings           = j.is_null() ? Timings{} : extract_timings(j);
        done_ = true;
    }
    return chunk;
}

EngineCore::StreamHandle EngineCore::generate_stream(const std::string & body_json) {
    json body = json::parse(body_json);  // throws on malformed input; propagates to binding layer

    // IMPORTANT: construct the ONE server_response_reader that will live for
    // this stream's entire lifetime directly on the heap, via guaranteed
    // copy elision from get_response_reader()'s return value. From this
    // point on it is mutated in place through the pointer (get_new_id(),
    // post_task()) and never copied again — see the correctness note at the
    // top of Phase B for why that matters (a second copy of a *populated*
    // reader would cancel the task when the copy is destroyed).
    auto reader = std::make_unique<server_response_reader>(ctx_.get_response_reader());

    server_task task(SERVER_TASK_TYPE_COMPLETION);
    {
        std::lock_guard<std::mutex> lk(parse_mu_);
        task.id = reader->get_new_id();

        std::vector<raw_buffer> files;
        json parsed = oaicompat_chat_params_parse(body, meta_->chat_params, files);
        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_.get_llama_context()));
        task.params = server_schema::eval_llama_cmpl_schema(
            vocab, params_, meta_->slot_n_ctx, meta_->logit_bias_eog, parsed);
        task.params.stream       = true;
        task.params.res_type     = TASK_RESPONSE_TYPE_NONE;
        task.params.cache_prompt = false;
        task.cli        = true;
        task.cli_prompt = parsed.at("prompt").get<std::string>();
        task.cli_files  = std::move(files);
    }

    reader->post_task(std::move(task));
    return StreamHandle(std::move(reader));  // only the unique_ptr moves; the reader's identity never changes
}
```

- [ ] **Step 2: Commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add src/cpp/engine_core.cpp
git commit -m "feat: EngineCore::generate_stream() and StreamHandle"
```

**Do not attempt to build yet** — `src/cpp/CMakeLists.txt` (Task 6) only compiles `binding.cpp`, and `binding.cpp` is still the Task-6 placeholder. Phase C rewrites both.

---

## Phase C: pybind11 Binding Layer (Tasks 11-14)

**Design principle for this phase (design spec §4.1):** `binding.cpp` only does GIL release/acquire, exception-type mapping, and type conversion (`EngineCore` structs ↔ Python dicts). No business logic — a future CUDA/Vulkan variant should never need to touch this file.

### Task 11: Write `exceptions.py`

Written now (before `binding.cpp`) because `binding.cpp` imports this module by name at runtime to raise typed exceptions.

**Files:**
- Create: `src/mineru_llama_cpp/exceptions.py`

- [ ] **Step 1: Write the file**

```python
"""Exception hierarchy for mineru_llama_cpp.

See design spec §5.3. The C++ binding layer (src/cpp/binding.cpp) imports
this module and raises these exact classes when llama-server reports an
error — do not rename these classes without updating binding.cpp's
raise_mapped_error() to match.
"""


class MineruLlamaCppError(Exception):
    """Base class for all exceptions raised by this library."""


class InvalidRequestError(MineruLlamaCppError):
    """The request itself was invalid: malformed JSON, bad image data, etc.

    Corresponds to llama-server's error_type::ERROR_TYPE_INVALID_REQUEST
    (JSON field "type": "invalid_request_error"), or to a raw C++ exception
    raised before the request was ever posted to the engine (e.g. malformed
    body JSON).
    """


class ContextExceededError(InvalidRequestError):
    """The prompt (plus requested completion length) exceeds n_ctx.

    Corresponds to llama-server's error_type::ERROR_TYPE_EXCEED_CONTEXT_SIZE
    (JSON field "type": "exceed_context_size_error"). Subclasses
    InvalidRequestError because it is, semantically, a client-supplied input
    that was too large for the configured Engine — callers that only care
    about "was my request bad" can catch InvalidRequestError and this is
    included; callers that specifically want to detect and react to
    context-size overflow (e.g. by truncating and retrying) can catch this
    subclass directly.
    """


class EngineError(MineruLlamaCppError):
    """An internal engine error occurred (not the caller's fault).

    Corresponds to every other llama-server error_type (server_error,
    not_found_error, permission_error, authentication_error,
    not_supported_error, unavailable_error) or to an unrecognized/missing
    "type" field.
    """
```

- [ ] **Step 2: Commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add src/mineru_llama_cpp/exceptions.py
git commit -m "feat: exception hierarchy"
```

---

### Task 12: Rewrite `binding.cpp` — `_EngineCore` construction and `generate()`

**Files:**
- Modify: `src/cpp/binding.cpp` (full rewrite, replacing the Task-6 placeholder)

- [ ] **Step 1: Rewrite the file**

```cpp
// binding.cpp — thin pybind11 layer. GIL release/acquire, exception-type
// mapping, and dict marshalling only. No business logic (design spec §4.1).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <nlohmann/json.hpp>

#include "engine_core.h"

#include <string>

namespace py = pybind11;
using json = nlohmann::ordered_json;

namespace {

// Raises the appropriate Python exception subclass (imported from
// mineru_llama_cpp.exceptions) and never returns. `type` is one of
// llama-server's error_type strings (see server-common.cpp
// format_error_response()), or "invalid_request_error" for pre-post C++
// exceptions caught in the binding layer.
[[noreturn]] void raise_mapped_error(const std::string & type, const std::string & message) {
    py::object exceptions_mod = py::module_::import("mineru_llama_cpp.exceptions");
    py::object exc_class;
    if (type == "exceed_context_size_error") {
        exc_class = exceptions_mod.attr("ContextExceededError");
    } else if (type == "invalid_request_error") {
        exc_class = exceptions_mod.attr("InvalidRequestError");
    } else {
        exc_class = exceptions_mod.attr("EngineError");
    }
    PyErr_SetString(exc_class.ptr(), message.c_str());
    throw py::error_already_set();
}

// Parses an EngineCore error_json string (see server-common.cpp
// format_error_response(): {"code":int,"message":str,"type":str}) and
// raises the mapped exception. Falls back to EngineError with the raw
// string as the message if error_json isn't valid JSON (defensive; should
// not happen in practice since it always originates from
// format_error_response()).
[[noreturn]] void raise_from_error_json(const std::string & error_json_str) {
    std::string type    = "server_error";
    std::string message = error_json_str;
    try {
        json ej = json::parse(error_json_str);
        type    = ej.value("type", type);
        message = ej.value("message", error_json_str);
    } catch (...) {
        // leave type/message at their fallback values
    }
    raise_mapped_error(type, message);
}

py::dict timings_to_dict(const EngineCore::Timings & t) {
    py::dict d;
    d["prompt_n"]             = t.prompt_n;
    d["prompt_ms"]            = t.prompt_ms;
    d["prompt_per_second"]    = t.prompt_per_second;
    d["predicted_n"]          = t.predicted_n;
    d["predicted_ms"]         = t.predicted_ms;
    d["predicted_per_second"] = t.predicted_per_second;
    return d;
}

py::dict generate_impl(EngineCore & self, const std::string & body) {
    EngineCore::GenerateResult r;
    try {
        py::gil_scoped_release release;
        r = self.generate(body);
    } catch (const std::exception & e) {
        raise_mapped_error("invalid_request_error", e.what());
    }
    if (r.is_error) {
        raise_from_error_json(r.error_json);
    }
    py::dict out;
    out["content"]          = r.content;
    out["finish_reason"]    = r.finish_reason;
    out["tokens_evaluated"] = r.tokens_evaluated;
    out["tokens_predicted"] = r.tokens_predicted;
    out["timings"]          = timings_to_dict(r.timings);
    return out;
}

} // namespace

PYBIND11_MODULE(_mineru_llama_cpp, m) {
    py::class_<EngineCore>(m, "_EngineCore")
        .def(py::init<const std::string &, const std::string &, int, int, int>(),
             py::arg("model_path"), py::arg("mmproj_path"), py::arg("n_ctx"),
             py::arg("n_gpu_layers"), py::arg("n_parallel"))
        .def("generate", &generate_impl, py::arg("body"));
}
```

- [ ] **Step 2: Commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add src/cpp/binding.cpp
git commit -m "feat: bind _EngineCore construction and generate()"
```

Note: do not build yet — `src/cpp/CMakeLists.txt` still only lists `binding.cpp` as a source and doesn't link against `server-context`/`mtmd`/etc yet. Task 14 fixes the CMakeLists.txt after Task 13 finishes `binding.cpp`.

---

### Task 13: Extend `binding.cpp` — `generate_stream()` as a Python iterator

**Files:**
- Modify: `src/cpp/binding.cpp`

- [ ] **Step 1: Add the `PyStreamIterator` wrapper class and its bindings**

Insert this into the anonymous namespace in `src/cpp/binding.cpp`, after `generate_impl` and before the closing `}` of the namespace:

```cpp
// Wraps EngineCore::StreamHandle (a move-only C++ type) so pybind11 can
// hold it inside a Python-iterable object. Implements the Python iterator
// protocol: __next__ returns a dict per chunk (including the final one,
// which carries finish_reason/tokens_*/timings) and raises StopIteration on
// the call *after* the final chunk was returned.
class PyStreamIterator {
public:
    explicit PyStreamIterator(EngineCore::StreamHandle handle) : handle_(std::move(handle)) {}

    py::dict next() {
        if (finished_) {
            throw py::stop_iteration();
        }
        EngineCore::Chunk c;
        {
            py::gil_scoped_release release;
            c = handle_.next_chunk();
        }
        if (c.is_error) {
            raise_from_error_json(c.error_json);
        }
        py::dict out;
        out["delta"] = c.delta;
        if (c.is_final) {
            finished_ = true;
            out["finish_reason"]    = c.finish_reason;
            out["tokens_evaluated"] = c.tokens_evaluated;
            out["tokens_predicted"] = c.tokens_predicted;
            out["timings"]          = timings_to_dict(c.timings);
        } else {
            out["finish_reason"]    = py::none();
            out["tokens_evaluated"] = py::none();
            out["tokens_predicted"] = py::none();
            out["timings"]          = py::none();
        }
        return out;
    }

private:
    EngineCore::StreamHandle handle_;
    bool finished_ = false;
};

PyStreamIterator generate_stream_impl(EngineCore & self, const std::string & body) {
    // No GIL release here: this only does the fast parse+post phase (a few
    // ms at most for jinja rendering), not the blocking wait — matches the
    // parse_mu_-guarded critical section in EngineCore. next_chunk() (above)
    // is what releases the GIL for the actual blocking wait.
    try {
        return PyStreamIterator(self.generate_stream(body));
    } catch (const std::exception & e) {
        raise_mapped_error("invalid_request_error", e.what());
    }
}
```

- [ ] **Step 2: Register `PyStreamIterator` and the `generate_stream` method in `PYBIND11_MODULE`**

Replace the `PYBIND11_MODULE` block at the end of `src/cpp/binding.cpp` with:

```cpp
PYBIND11_MODULE(_mineru_llama_cpp, m) {
    py::class_<PyStreamIterator>(m, "_StreamIterator")
        .def("__iter__", [](PyStreamIterator & self) -> PyStreamIterator & { return self; })
        .def("__next__", &PyStreamIterator::next);

    py::class_<EngineCore>(m, "_EngineCore")
        .def(py::init<const std::string &, const std::string &, int, int, int>(),
             py::arg("model_path"), py::arg("mmproj_path"), py::arg("n_ctx"),
             py::arg("n_gpu_layers"), py::arg("n_parallel"))
        .def("generate", &generate_impl, py::arg("body"))
        .def("generate_stream", &generate_stream_impl, py::arg("body"));
}
```

- [ ] **Step 3: Commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add src/cpp/binding.cpp
git commit -m "feat: bind generate_stream() as a Python iterator"
```

---

### Task 14: Rewrite `src/cpp/CMakeLists.txt` for the real engine, and build

**Files:**
- Modify: `src/cpp/CMakeLists.txt`

- [ ] **Step 1: Rewrite the file**

```cmake
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_mineru_llama_cpp
    binding.cpp
    engine_core.cpp
)

# These includes mirror the ones proven to work in the technical-validation
# spike's CMakeLists.txt, just re-pathed to the submodule. We add them
# explicitly rather than relying on transitive include propagation from the
# server-context/mtmd/etc. CMake targets, since that propagation was never
# verified for this llama.cpp version -- this explicit form is known-good.
target_include_directories(_mineru_llama_cpp PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/common
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/tools/server
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/tools/mtmd
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor
)

target_link_libraries(_mineru_llama_cpp PRIVATE
    server-context
    llama-common-base
    llama-common
    mtmd
    llama
    ggml
    ggml-base
    ggml-cpu
)

# GGML_METAL is a CACHE variable set by ggml's own CMakeLists.txt (via
# add_subdirectory(third_party/llama.cpp) in the top-level CMakeLists.txt),
# auto-detected per-platform. We deliberately do not set it ourselves (design
# spec §7) — we only react to whatever it resolved to.
if (GGML_METAL)
    target_link_libraries(_mineru_llama_cpp PRIVATE ggml-metal)
endif()

if (APPLE)
    target_link_libraries(_mineru_llama_cpp PRIVATE
        "-framework Metal"
        "-framework Foundation"
        "-framework CoreGraphics"
    )
endif()

# The shared libraries (libllama, libmtmd, libggml-*) built by
# add_subdirectory(third_party/llama.cpp) land wherever llama.cpp's own
# CMAKE_RUNTIME_OUTPUT_DIRECTORY points (shared across the whole build tree
# since CMAKE_BINARY_DIR is top-level-wide). Use a generator expression
# rather than hardcoding that path, so this keeps working if llama.cpp's
# internal output-directory layout ever changes.
set_target_properties(_mineru_llama_cpp PROPERTIES
    BUILD_RPATH "$<TARGET_FILE_DIR:llama>"
    INSTALL_RPATH "$<TARGET_FILE_DIR:llama>"
)

install(TARGETS _mineru_llama_cpp DESTINATION mineru_llama_cpp)
```

- [ ] **Step 2: Rebuild**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
/Users/jinzhenj/MinerU-Repo/.venv/bin/pip install --no-build-isolation -e . -v
```

Expected: succeeds. This rebuild only recompiles `binding.cpp`/`engine_core.cpp` and re-links (llama.cpp itself was already built in Task 6) — should take well under a minute, not several minutes.

- [ ] **Step 3: Verify with a manual smoke test (real model, real prompt, no pytest yet)**

```bash
/Users/jinzhenj/MinerU-Repo/.venv/bin/python -c "
from mineru_llama_cpp._mineru_llama_cpp import _EngineCore
import json

core = _EngineCore(
    '/Users/jinzhenj/.mineru/models/MinerU2.5-Pro-2605-1.2B-Q8_0.gguf',
    '/Users/jinzhenj/.mineru/models/mmproj-MinerU2.5-Pro-2605-1.2B-Q8_0.gguf',
    4096, 99, 1,
)
body = json.dumps({
    'model': 'm',
    'messages': [{'role': 'user', 'content': 'Say hello in one word.'}],
    'temperature': 0.0, 'top_k': 1, 'n_predict': 8, 'stream': False,
})
result = core.generate(body)
print('RESULT:', result)
assert result['content']
assert result['finish_reason'] in ('stop', 'length')
print('OK')
"
```

Expected: prints a `RESULT: {...}` dict with a non-empty `content` and ends with `OK`. This is the FIRST real end-to-end check of the ported (not spike) C++ logic — if it fails, the bug is almost certainly in Tasks 9/12 (the `generate()` port), since Task 6's wiring and Tasks 7-8's constructor were already exercised by `ping()`/no-op respectively; check field name/type mismatches between `engine_core.cpp`'s JSON handling and `binding.cpp`'s dict construction first.

- [ ] **Step 4: Commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add src/cpp/CMakeLists.txt
git commit -m "build: link EngineCore against llama.cpp submodule targets"
```

---

## Phase D: Python Layer (Tasks 15-20)

### Task 15: Write `types.py`

**Files:**
- Create: `src/mineru_llama_cpp/types.py`

- [ ] **Step 1: Write the file**

```python
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
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
/Users/jinzhenj/MinerU-Repo/.venv/bin/python -c "
from mineru_llama_cpp.types import Message, Messages, GenerateResult, GenerateChunk, GenerationTimings
m: Message = {'role': 'user', 'content': 'hi'}
r = GenerateResult(content='x', finish_reason='stop', tokens_evaluated=1, tokens_predicted=1,
                    timings=GenerationTimings(1, 1.0, 1.0, 1, 1.0, 1.0))
c = GenerateChunk(delta='x')
print('OK', m, r, c)
"
```

Expected: `OK {...} GenerateResult(...) GenerateChunk(...)`. (This works even though `mineru_llama_cpp/__init__.py` is still the Task-6 placeholder and doesn't export these yet — we're importing directly from the submodule path, which is always valid.)

- [ ] **Step 3: Commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add src/mineru_llama_cpp/types.py
git commit -m "feat: input/output types (Message, GenerateResult, GenerateChunk)"
```

---

### Task 16: Write `sampling.py`

**Files:**
- Create: `src/mineru_llama_cpp/sampling.py`

- [ ] **Step 1: Write the file**

```python
"""Sampling parameters for mineru_llama_cpp.Engine. See design spec §5.2.

Field names are copied verbatim from the JSON keys registered in
llama.cpp's tools/server/server-schema.cpp (field_num/field_json calls) —
NOT from vLLM/HuggingFace naming conventions. Notably: `repeat_penalty`
(not `repetition_penalty`), `n_predict` (not `max_tokens`, though
`max_tokens`/`max_completion_tokens` are llama.cpp's own OpenAI-compatible
aliases for n_predict and would also work if sent directly as JSON, but the
canonical field here is the native name).

Deliberately excluded (not oversights — see design spec §5.2): grammar,
json_schema, logit_bias, samplers, adaptive_target, adaptive_decay,
backend_sampling, post_sampling_probs. These are advanced/structured-output
features orthogonal to basic sampling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class SamplingParams:
    # Generation length
    n_predict: int | None = None  # alias: max_tokens, max_completion_tokens

    # Core sampling
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    typical_p: float | None = None
    top_n_sigma: float | None = None
    xtc_probability: float | None = None
    xtc_threshold: float | None = None
    dynatemp_range: float | None = None
    dynatemp_exponent: float | None = None

    # Repetition penalties
    repeat_last_n: int | None = None
    repeat_penalty: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    # DRY
    dry_multiplier: float | None = None
    dry_base: float | None = None
    dry_allowed_length: int | None = None
    dry_penalty_last_n: int | None = None
    dry_sequence_breakers: list[str] | None = None

    # Mirostat
    mirostat: int | None = None
    mirostat_tau: float | None = None
    mirostat_eta: float | None = None

    # Other
    seed: int | None = None
    stop: list[str] | None = None
    n_probs: int | None = None
    min_keep: int | None = None
    ignore_eos: bool | None = None

    def to_json_fields(self) -> dict:
        """Returns a dict of only the explicitly-set (non-None) fields,
        suitable for merging into an OpenAI-style chat completion body."""
        return {k: v for k, v in asdict(self).items() if v is not None}
```

- [ ] **Step 2: Write a quick verification**

```bash
/Users/jinzhenj/MinerU-Repo/.venv/bin/python -c "
from mineru_llama_cpp.sampling import SamplingParams
sp = SamplingParams(temperature=0.0, top_k=1, n_predict=8)
fields = sp.to_json_fields()
assert fields == {'temperature': 0.0, 'top_k': 1, 'n_predict': 8}, fields
print('OK', fields)
"
```

Expected: `OK {'temperature': 0.0, 'top_k': 1, 'n_predict': 8}`

- [ ] **Step 3: Commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add src/mineru_llama_cpp/sampling.py
git commit -m "feat: SamplingParams dataclass"
```

---

### Task 17: Write `engine.py` — `__init__`, `generate`, `agenerate`

**Files:**
- Create: `src/mineru_llama_cpp/engine.py`

- [ ] **Step 1: Write the file (part 1 — imports, `__init__`, `_build_body`, `generate`, `agenerate`)**

```python
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
```

- [ ] **Step 2: Commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add src/mineru_llama_cpp/engine.py
git commit -m "feat: Engine.__init__, generate(), agenerate()"
```

---

### Task 18: Extend `engine.py` — `stream`, `astream`

**Files:**
- Modify: `src/mineru_llama_cpp/engine.py`

- [ ] **Step 1: Append `stream()` and `astream()` to the `Engine` class**

Insert these methods into the `Engine` class in `src/mineru_llama_cpp/engine.py`, after `agenerate()`:

```python
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
```

- [ ] **Step 2: Commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add src/mineru_llama_cpp/engine.py
git commit -m "feat: Engine.stream(), astream() (sync-to-async generator bridge)"
```

---

### Task 19: Extend `engine.py` — `close`, `aclose`, context managers, `__del__`

**Files:**
- Modify: `src/mineru_llama_cpp/engine.py`

- [ ] **Step 1: Append lifecycle methods to the `Engine` class**

Insert these methods into the `Engine` class in `src/mineru_llama_cpp/engine.py`, after `astream()`:

```python
    # --- lifecycle ---

    def close(self) -> None:
        """Explicit shutdown: terminate + join the background loop thread,
        free the llama backend. Idempotent (safe to call more than once).

        v1 assumes no in-flight generate()/stream() calls when close() is
        called — see design spec §5.4's note on close()/in-flight requests
        for why this isn't handled defensively yet."""
        if self._closed:
            return
        self._closed = True
        del self._core  # drops the last reference -> EngineCore's C++ destructor runs now

    async def aclose(self) -> None:
        """Async version of close(), offloaded to a thread so it doesn't
        block the event loop while the loop thread joins."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.close)

    def __enter__(self) -> Engine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> Engine:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def __del__(self) -> None:
        """Safety-net fallback only — normal code should call close()/
        aclose() explicitly (or use `with`/`async with`)."""
        try:
            self.close()
        except Exception:
            pass
```

- [ ] **Step 2: Verify the full `Engine` class end-to-end (real model, sync path)**

```bash
/Users/jinzhenj/MinerU-Repo/.venv/bin/python -c "
from mineru_llama_cpp.engine import Engine

with Engine(
    '/Users/jinzhenj/.mineru/models/MinerU2.5-Pro-2605-1.2B-Q8_0.gguf',
    '/Users/jinzhenj/.mineru/models/mmproj-MinerU2.5-Pro-2605-1.2B-Q8_0.gguf',
    n_ctx=4096,
) as engine:
    result = engine.generate([{'role': 'user', 'content': 'Say hello in one word.'}])
    print('generate:', result)
    assert result.content
    chunks = list(engine.stream([{'role': 'user', 'content': 'Say hello in one word.'}]))
    print('stream chunks:', len(chunks))
    assert len(chunks) > 1
    assert chunks[-1].finish_reason is not None
print('OK')
"
```

Expected: prints `generate: GenerateResult(...)`, `stream chunks: N` (N > 1), then `OK`.

- [ ] **Step 3: Commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add src/mineru_llama_cpp/engine.py
git commit -m "feat: Engine.close(), aclose(), context managers, __del__"
```

---

### Task 20: Rewrite `__init__.py` to export the full public API

**Files:**
- Modify: `src/mineru_llama_cpp/__init__.py` (full rewrite, replacing the Task-6 placeholder)

- [ ] **Step 1: Rewrite the file**

```python
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
]
```

- [ ] **Step 2: Verify the public import surface**

```bash
/Users/jinzhenj/MinerU-Repo/.venv/bin/python -c "
import mineru_llama_cpp as m
assert hasattr(m, 'Engine')
assert hasattr(m, 'SamplingParams')
assert hasattr(m, 'InvalidRequestError')
assert hasattr(m, 'ContextExceededError')
assert hasattr(m, 'EngineError')
assert issubclass(m.ContextExceededError, m.InvalidRequestError)
assert issubclass(m.InvalidRequestError, m.MineruLlamaCppError)
print('OK')
"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add src/mineru_llama_cpp/__init__.py
git commit -m "feat: export full public API from mineru_llama_cpp"
```

---

## Phase E: Tests (Tasks 21-28)

**Note on fixture image sizing:** the technical-validation phase found that feeding a non-1036x1036 image for layout detection causes severe model output degradation (the model was trained on mineru-vl-utils' `layout_image_size=(1036,1036)` preprocessing). `conftest.py`'s `layout_image_path` fixture always produces a 1036x1036 image — do not use a differently-sized image in any test that checks output *content* (quality, not just "did it run").

### Task 21: Write `conftest.py`

**Files:**
- Create: `tests/conftest.py`
- Create (generated at test time, not checked in): `tests/fixtures/layout_1036.png`

- [ ] **Step 1: Add `pytest-asyncio` as a test dependency and enable auto mode**

Modify `pyproject.toml`: change the `[project.optional-dependencies]` `test` list and add `asyncio_mode` to `[tool.pytest.ini_options]`:

```toml
[project.optional-dependencies]
test = ["pytest>=7.0", "pytest-asyncio>=0.23", "pillow"]
```

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
asyncio_mode = "auto"
```

(`asyncio_mode = "auto"` means `async def test_*` functions are automatically run as asyncio tests — no `@pytest.mark.asyncio` decorator needed anywhere in this test suite.)

Install the test deps:

```bash
/Users/jinzhenj/MinerU-Repo/.venv/bin/pip install pytest pytest-asyncio pillow psutil
```

(`psutil` is used by `test_lifecycle.py` in Task 26 for RSS measurement, with a `resource`-module fallback if unavailable — not declared in `pyproject.toml` since it's optional/best-effort there too.)

- [ ] **Step 2: Write `tests/conftest.py`**

```python
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
    eng = Engine(MODEL, MMPROJ, n_ctx=8192, n_gpu_layers=99, n_parallel=4)
    yield eng
    eng.close()
```

- [ ] **Step 3: Add `.gitignore` entry for the generated fixture**

Append to `.gitignore`:

```bash
echo "tests/fixtures/*.png" >> /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp/.gitignore
```

- [ ] **Step 4: Commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add pyproject.toml tests/conftest.py .gitignore
git commit -m "test: add conftest.py with session-scoped engine + layout image fixtures"
```

---

### Task 22: Write `tests/test_generate.py`

**Files:**
- Create: `tests/test_generate.py`

- [ ] **Step 1: Write the file**

```python
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
```

- [ ] **Step 2: Run it**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
/Users/jinzhenj/MinerU-Repo/.venv/bin/python -m pytest tests/test_generate.py -v
```

Expected: `2 passed`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_generate.py
git commit -m "test: generate() text-only and image layout detection"
```

---

### Task 23: Write `tests/test_streaming.py`

**Files:**
- Create: `tests/test_streaming.py`

- [ ] **Step 1: Write the file**

```python
from mineru_llama_cpp import GenerateChunk, SamplingParams


def test_stream_yields_multiple_chunks_with_final_metadata(engine):
    chunks = list(engine.stream([{"role": "user", "content": "List three fruits, one per line."}]))
    assert len(chunks) > 1
    assert all(isinstance(c, GenerateChunk) for c in chunks)
    assert all(c.finish_reason is None for c in chunks[:-1])
    assert chunks[-1].finish_reason is not None
    assert chunks[-1].timings is not None


def test_stream_concatenation_matches_generate(engine):
    messages = [{"role": "user", "content": "List three fruits, one per line."}]
    streamed = "".join(c.delta for c in engine.stream(messages))
    non_streamed = engine.generate(messages).content
    assert streamed == non_streamed


def test_stream_chunk_count_grows_with_n_predict(engine):
    messages = [{"role": "user", "content": "List three fruits, one per line."}]
    short = list(engine.stream(messages, SamplingParams(temperature=0.0, top_k=1, n_predict=32)))
    long_ = list(engine.stream(messages, SamplingParams(temperature=0.0, top_k=1, n_predict=96)))
    assert len(long_) >= len(short)


async def test_astream_yields_same_content_as_stream(engine):
    messages = [{"role": "user", "content": "List three fruits, one per line."}]
    sp = SamplingParams(temperature=0.0, top_k=1, n_predict=32)

    async_chunks = [c async for c in engine.astream(messages, sp)]
    sync_chunks = list(engine.stream(messages, sp))

    assert "".join(c.delta for c in async_chunks) == "".join(c.delta for c in sync_chunks)
    assert async_chunks[-1].finish_reason is not None
```

- [ ] **Step 2: Run it**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
/Users/jinzhenj/MinerU-Repo/.venv/bin/python -m pytest tests/test_streaming.py -v
```

Expected: `4 passed`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_streaming.py
git commit -m "test: stream()/astream() chunk semantics and sync/async parity"
```

---

### Task 24: Write `tests/test_concurrency.py`

**Files:**
- Create: `tests/test_concurrency.py`

- [ ] **Step 1: Write the file**

```python
"""Covers multi-slot concurrency + result routing (Tier2 #3 from the
technical-validation phase)."""

import asyncio
import time

from mineru_llama_cpp import SamplingParams

_PROMPTS = [
    "List three fruits, one per line.",
    "List three colors, one per line.",
    "List three animals, one per line.",
    "List three cities, one per line.",
]


async def test_concurrent_agenerate_routing_matches_serial(engine):
    sp = SamplingParams(temperature=0.0, top_k=1, n_predict=32)

    def body(prompt: str) -> list[dict]:
        return [{"role": "user", "content": prompt}]

    serial = []
    for p in _PROMPTS:
        serial.append(await engine.agenerate(body(p), sp))

    concurrent = await asyncio.gather(*[engine.agenerate(body(p), sp) for p in _PROMPTS])

    for i in range(len(_PROMPTS)):
        assert concurrent[i].content == serial[i].content, f"routing mismatch at prompt index {i}"


async def test_concurrent_agenerate_faster_than_serial(engine):
    sp = SamplingParams(temperature=0.0, top_k=1, n_predict=32)

    def body(prompt: str) -> list[dict]:
        return [{"role": "user", "content": prompt}]

    t0 = time.monotonic()
    for p in _PROMPTS:
        await engine.agenerate(body(p), sp)
    serial_dt = time.monotonic() - t0

    t0 = time.monotonic()
    await asyncio.gather(*[engine.agenerate(body(p), sp) for p in _PROMPTS])
    concurrent_dt = time.monotonic() - t0

    assert concurrent_dt < serial_dt, f"concurrent ({concurrent_dt:.2f}s) not faster than serial ({serial_dt:.2f}s)"
```

- [ ] **Step 2: Run it**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
/Users/jinzhenj/MinerU-Repo/.venv/bin/python -m pytest tests/test_concurrency.py -v
```

Expected: `2 passed`. (If `test_concurrent_agenerate_faster_than_serial` is flaky on a loaded machine, it's measuring wall-clock speedup, not correctness — rerun before treating it as a real regression.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_concurrency.py
git commit -m "test: multi-slot concurrency routing and speedup"
```

---

### Task 25: Write `tests/test_error_handling.py`

**Files:**
- Create: `tests/test_error_handling.py`

- [ ] **Step 1: Write the file**

```python
"""Covers the exception hierarchy (design spec §5.3) and error isolation
(Tier2 #5 from the technical-validation phase: a bad request must not take
down the Engine)."""

import pytest

from mineru_llama_cpp import ContextExceededError, Engine, InvalidRequestError

MODEL = "/Users/jinzhenj/.mineru/models/MinerU2.5-Pro-2605-1.2B-Q8_0.gguf"
MMPROJ = "/Users/jinzhenj/.mineru/models/mmproj-MinerU2.5-Pro-2605-1.2B-Q8_0.gguf"

_BAD_IMAGE_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!notvalidbase64!!!"}},
            {"type": "text", "text": "describe"},
        ],
    }
]


def test_malformed_json_raises_invalid_request(engine):
    # Engine.generate() always builds valid JSON itself, so malformed JSON
    # can only be exercised by going through the private _core directly.
    with pytest.raises(InvalidRequestError):
        engine._core.generate("{not valid json")


def test_bad_image_raises_invalid_request(engine):
    with pytest.raises(InvalidRequestError):
        engine.generate(_BAD_IMAGE_MESSAGES)


def test_context_exceeded_raises_context_exceeded_error():
    # A dedicated small-n_ctx Engine (not the shared session fixture), so
    # n_ctx=512 here doesn't affect any other test.
    with Engine(MODEL, MMPROJ, n_ctx=512) as small_engine:
        huge_prompt = "x " * 2000  # far more than 512 tokens
        with pytest.raises(ContextExceededError):
            small_engine.generate([{"role": "user", "content": huge_prompt}])


def test_context_exceeded_is_also_an_invalid_request_error():
    with Engine(MODEL, MMPROJ, n_ctx=512) as small_engine:
        huge_prompt = "x " * 2000
        with pytest.raises(InvalidRequestError):  # the base-class catch must also work
            small_engine.generate([{"role": "user", "content": huge_prompt}])


def test_engine_survives_bad_requests_and_serves_good_ones_after(engine):
    good = [{"role": "user", "content": "hi"}]
    assert engine.generate(good).content

    with pytest.raises(InvalidRequestError):
        engine._core.generate("{not valid json")
    assert engine.generate(good).content, "engine unusable after malformed JSON"

    with pytest.raises(InvalidRequestError):
        engine.generate(_BAD_IMAGE_MESSAGES)
    assert engine.generate(good).content, "engine unusable after bad image"
```

- [ ] **Step 2: Run it**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
/Users/jinzhenj/MinerU-Repo/.venv/bin/python -m pytest tests/test_error_handling.py -v
```

Expected: `5 passed`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_error_handling.py
git commit -m "test: exception hierarchy and error isolation"
```

---

### Task 26: Write `tests/test_lifecycle.py`

**Files:**
- Create: `tests/test_lifecycle.py`

- [ ] **Step 1: Write the file**

```python
"""Covers long-lived-service memory stability and graceful shutdown
(Tier2 #6 from the technical-validation phase)."""

import gc
import os

from mineru_llama_cpp import Engine

MODEL = "/Users/jinzhenj/.mineru/models/MinerU2.5-Pro-2605-1.2B-Q8_0.gguf"
MMPROJ = "/Users/jinzhenj/.mineru/models/mmproj-MinerU2.5-Pro-2605-1.2B-Q8_0.gguf"


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB->MB on macOS


def test_many_requests_no_memory_growth(engine):
    n = 50
    rss_before = _rss_mb()
    for i in range(n):
        result = engine.generate([{"role": "user", "content": f"Say something {i}"}])
        assert result.content
    rss_after = _rss_mb()
    growth = rss_after - rss_before
    assert growth < 100, f"RSS grew {growth:.1f} MB over {n} requests (threshold 100 MB)"


def test_close_is_idempotent_and_engine_is_reconstructable():
    eng = Engine(MODEL, MMPROJ, n_ctx=4096)
    assert eng.generate([{"role": "user", "content": "hi"}]).content
    eng.close()
    gc.collect()

    eng.close()  # must not raise on a second call

    # A fresh Engine must still load and work after the first one closed --
    # proves the C++ backend was actually released, not just Python-detached.
    eng2 = Engine(MODEL, MMPROJ, n_ctx=4096)
    assert eng2.generate([{"role": "user", "content": "hi"}]).content
    eng2.close()


def test_context_manager_closes_on_exit():
    with Engine(MODEL, MMPROJ, n_ctx=4096) as eng:
        assert eng.generate([{"role": "user", "content": "hi"}]).content
    assert eng._closed
```

- [ ] **Step 2: Run it**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
/Users/jinzhenj/MinerU-Repo/.venv/bin/python -m pytest tests/test_lifecycle.py -v
```

Expected: `3 passed`. (This test constructs 2 extra Engines beyond the shared fixture, so it's the slowest file in the suite — several extra model loads, tens of seconds.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_lifecycle.py
git commit -m "test: long-lived memory stability and graceful shutdown"
```

---

### Task 27: Write `tests/test_determinism.py`

**Files:**
- Create: `tests/test_determinism.py`

- [ ] **Step 1: Write the file**

```python
"""Covers temp=0 reproducibility, within a run and across slots (Tier3 #9
from the technical-validation phase)."""

import asyncio

from mineru_llama_cpp import SamplingParams

_MESSAGES = [{"role": "user", "content": "List three fruits, one per line."}]
_SP = SamplingParams(temperature=0.0, top_p=0.01, top_k=1, repeat_penalty=1.0, n_predict=32, seed=42)


def test_same_seed_repeated_calls_are_identical(engine):
    outputs = [engine.generate(_MESSAGES, _SP).content for _ in range(3)]
    assert len(set(outputs)) == 1, f"non-deterministic outputs: {outputs}"


async def test_same_seed_identical_across_concurrent_slots(engine):
    outputs = await asyncio.gather(*[engine.agenerate(_MESSAGES, _SP) for _ in range(4)])
    contents = [o.content for o in outputs]
    assert len(set(contents)) == 1, f"non-deterministic across slots: {contents}"
```

- [ ] **Step 2: Run it**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
/Users/jinzhenj/MinerU-Repo/.venv/bin/python -m pytest tests/test_determinism.py -v
```

Expected: `2 passed`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_determinism.py
git commit -m "test: temp=0 determinism within-run and cross-slot"
```

---

### Task 28: Run the full suite, write a top-level README, final commit

**Files:**
- Create: `README.md`

- [ ] **Step 1: Run the entire test suite**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
/Users/jinzhenj/MinerU-Repo/.venv/bin/python -m pytest tests/ -v --ignore=tests/legacy_spike
```

Expected: all tests pass (`test_generate.py` 2, `test_streaming.py` 4, `test_concurrency.py` 2, `test_error_handling.py` 5, `test_lifecycle.py` 3, `test_determinism.py` 2 — 18 total). `--ignore=tests/legacy_spike` is required since that directory's files are not meant to be collected as part of this suite (see Task 3's README there) and would fail to even import under the new package name.

If anything fails here that passed individually in its own task, suspect fixture-scope interaction (the `engine` fixture is session-scoped and shared across all files — a test that mutates engine-global state, like `test_error_handling.py`'s malformed-JSON test, could in principle leave something behind; re-run the specific failing file alone first to confirm whether it's an isolation issue or a real regression).

- [ ] **Step 2: Write `README.md`**

```markdown
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
```

- [ ] **Step 3: Final commit**

```bash
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add README.md
git commit -m "docs: add top-level README"
```

- [ ] **Step 4: Copy the design spec and this plan into the new repo for posterity**

```bash
mkdir -p /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp/docs/superpowers/specs
mkdir -p /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp/docs/superpowers/plans
cp /Users/jinzhenj/MinerU-Repo/mineru-vl-engine/docs/superpowers/specs/2026-08-03-mineru-llama-cpp-engine-design.md \
   /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp/docs/superpowers/specs/
cp /Users/jinzhenj/MinerU-Repo/mineru-vl-engine/docs/superpowers/plans/2026-08-03-mineru-llama-cpp-engine-implementation.md \
   /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp/docs/superpowers/plans/
cd /Users/jinzhenj/MinerU-Repo/mineru-llama-cpp
git add docs/superpowers
git commit -m "docs: copy design spec and implementation plan into the new repo"
```

**End state after this task:** a working `mineru-llama-cpp` package, installed editable in the shared venv, with 18 passing pytest tests covering every capability proven during the technical-validation phase (JSON/oaicompat input, async/GIL bridging, multi-slot concurrency + routing, real streaming, layered exceptions + error isolation, long-lived memory stability + graceful shutdown, temp=0 determinism), ready for `mineru-vl-utils` integration work (out of scope for this plan — see design spec §9).
