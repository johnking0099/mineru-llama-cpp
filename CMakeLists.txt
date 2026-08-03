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
set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "" FORCE)  # LLAMA_BUILD_UI alone doesn't skip the HF asset download fallback

add_subdirectory(third_party/llama.cpp)
add_subdirectory(src/cpp)
