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
