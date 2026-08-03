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
