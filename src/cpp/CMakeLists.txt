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

# BUILD_RPATH points at the build tree so the .so can load its sibling
# ggml/llama/mtmd dylibs straight out of wherever llama.cpp's own
# CMAKE_RUNTIME_OUTPUT_DIRECTORY put them (shared across the whole build tree
# since CMAKE_BINARY_DIR is top-level-wide) -- this is what editable
# "inplace" installs run against, since there's no install step to relocate
# anything. A generator expression is used rather than a hardcoded path, so
# this keeps working if llama.cpp's internal output-directory layout changes.
#
# INSTALL_RPATH is relative (@loader_path/../lib on macOS, $ORIGIN/../lib on
# Linux) instead, since a wheel's lib/ sits one level up from
# mineru_llama_cpp/_mineru_llama_cpp.so -- baking in the build-tree absolute
# path here would make every wheel unusable on any machine other than the
# one it was built on. This overrides the directory-scoped
# CMAKE_INSTALL_RPATH set in the top-level CMakeLists.txt, since explicit
# target properties take precedence -- kept the same value for consistency,
# but set explicitly since this target's needs are the reason that variable
# exists.
set_target_properties(_mineru_llama_cpp PROPERTIES
    BUILD_RPATH "$<TARGET_FILE_DIR:llama>"
    INSTALL_RPATH "$<IF:$<BOOL:${APPLE}>,@loader_path/../lib,$ORIGIN/../lib>"
)

# Build the extension module directly into the source tree so that
# scikit-build-core's editable "inplace" mode (which runs only `cmake
# --build`, never `cmake --install`) produces a .so where Python's import
# machinery actually looks for it. Regular wheel/redirect builds are
# unaffected: the install(TARGETS) step below copies the .so from here into
# the wheel exactly as before.
set_target_properties(_mineru_llama_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/mineru_llama_cpp"
)

install(TARGETS _mineru_llama_cpp DESTINATION mineru_llama_cpp)

# llama-server links against this at runtime (@rpath/libllama-server-impl.dylib),
# but llama.cpp's own tools/server/CMakeLists.txt only installs it when
# LLAMA_TOOLS_INSTALL is ON, which defaults to LLAMA_STANDALONE (OFF here,
# since we add_subdirectory() llama.cpp instead of building it standalone).
# Installing it here directly avoids flipping LLAMA_TOOLS_INSTALL globally,
# which would also install a dozen unrelated llama.cpp CLI tools we don't
# ship (llama-tts, llama-mtmd-cli, llama-fit-params, etc).
install(TARGETS llama-server-impl LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
