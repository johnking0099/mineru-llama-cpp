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
