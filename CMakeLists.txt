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

# Every binary and shared library under third_party/llama.cpp links its
# sibling ggml/llama/mtmd libs via @rpath/$ORIGIN (bare LC_LOAD_DYLIB/DT_NEEDED
# entries with no baked-in directory). CMAKE_INSTALL_RPATH is a
# directory-scoped (non-CACHE) variable, so setting it here before
# add_subdirectory() applies it to every target llama.cpp defines, without
# patching its CMakeLists.txt.
#
# The wheel layout is flat: bin/, lib/, and mineru_llama_cpp/ are siblings
# under the wheel's platlib root. @loader_path (macOS)/$ORIGIN (Linux) covers
# dylib-to-dylib lookups within lib/; the /../lib variant covers
# bin/llama-server and mineru_llama_cpp/_mineru_llama_cpp.so reaching into
# lib/ from one level down. Both are added so RPATH resolution works
# regardless of which directory a given consumer sits in.
if (APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../lib")
else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib")
endif()

# --- Apply patches to the llama.cpp submodule before building it ---
#
# patches/llama.cpp/*.patch fix bugs in llama.cpp itself that only surface
# through this project's usage (see each patch's header comment for the
# specific bug). Applied here, before add_subdirectory(), so every build
# (fresh checkout or CI) picks them up automatically -- no manual step, and
# no risk of building against an unpatched tree.
#
# Idempotent and safe to re-run: `git apply --check` tells us whether the
# patch is already applied (via --reverse --check) before touching anything,
# so repeated configure runs don't fail or double-apply.
file(GLOB LLAMA_CPP_PATCHES "${CMAKE_SOURCE_DIR}/patches/llama.cpp/*.patch")
list(SORT LLAMA_CPP_PATCHES)
foreach(patch_file ${LLAMA_CPP_PATCHES})
    get_filename_component(patch_name ${patch_file} NAME)

    execute_process(
        COMMAND git apply --reverse --check ${patch_file}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/llama.cpp
        RESULT_VARIABLE already_applied
        OUTPUT_QUIET ERROR_QUIET
    )
    if(already_applied EQUAL 0)
        message(STATUS "llama.cpp patch already applied: ${patch_name}")
        continue()
    endif()

    execute_process(
        COMMAND git apply --check ${patch_file}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/llama.cpp
        RESULT_VARIABLE applies_cleanly
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT applies_cleanly EQUAL 0)
        message(FATAL_ERROR
            "llama.cpp patch ${patch_name} does not apply and is not already "
            "applied -- the submodule tree is in an unexpected state (wrong "
            "commit checked out, or the patch needs updating for a new "
            "llama.cpp version). Inspect third_party/llama.cpp manually.")
    endif()

    message(STATUS "Applying llama.cpp patch: ${patch_name}")
    execute_process(
        COMMAND git apply ${patch_file}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/llama.cpp
        RESULT_VARIABLE apply_result
    )
    if(NOT apply_result EQUAL 0)
        message(FATAL_ERROR "Failed to apply llama.cpp patch: ${patch_name}")
    endif()
endforeach()

add_subdirectory(third_party/llama.cpp)
add_subdirectory(src/cpp)
