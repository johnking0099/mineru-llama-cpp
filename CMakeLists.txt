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

# Drop the OpenSSL dependency. LLAMA_OPENSSL (default ON) makes vendored
# cpp-httplib link OpenSSL::SSL/Crypto for HTTPS, which propagates through
# libllama-common all the way to our _mineru_llama_cpp extension. On the build
# machine that resolves to a Homebrew-absolute path
# (/opt/homebrew/opt/openssl@3/...) that won't exist on a user's machine ->
# "Library not loaded" at import. We only ever load local GGUF paths (no
# --hf-repo download, see README), so HTTPS is dead weight -- turning it off
# removes libssl/libcrypto entirely and leaves the wheel depending only on
# system libraries.
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)

# Load ggml backends (CPU, Vulkan, Metal, CUDA, ...) as separate dlopen'd
# MODULE libraries at runtime instead of hard-linking them into
# _mineru_llama_cpp/llama-server. Without this, a backend's own shared
# library dependency (e.g. libvulkan.so.1) becomes a *hard* DT_NEEDED of the
# whole extension: on a machine that lacks that dependency, the extension
# fails to even import, because the dynamic linker resolves every DT_NEEDED
# before any of our code -- including ggml's own graceful backend-selection
# fallback in its registry -- gets a chance to run (confirmed via a Linux
# container lacking libvulkan.so.1: ImportError at import time, not a clean
# CPU fallback). With GGML_BACKEND_DL=ON, a missing backend dependency just
# means that one MODULE fails to dlopen and never registers; ggml's CPU
# backend (always statically linked/registered, see ggml-backend-reg.cpp's
# #ifdef GGML_USE_CPU block) still works, so CPU fallback actually happens.
# See engine_core.cpp's load_packaged_backends() for how we locate these
# MODULEs at runtime without setting GGML_BACKEND_PATH.
set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)

# BUILD_SHARED_LIBS is a hard prerequisite of GGML_BACKEND_DL=ON (ggml's own
# CMakeLists.txt errors out otherwise). llama.cpp's default is already ON for
# our target platforms (Linux/macOS -- see BUILD_SHARED_LIBS_DEFAULT in
# third_party/llama.cpp/CMakeLists.txt: only EMSCRIPTEN/MINGW flip it OFF), so
# this is functionally redundant on those targets today. Force-setting it
# anyway makes the DL prerequisite explicit at our level and defends against
# a future MinGW/Emscripten target or an externally-supplied
# -DBUILD_SHARED_LIBS=OFF quietly breaking the DL build.
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

# GGML_NATIVE (llama.cpp's default ON) is incompatible with GGML_BACKEND_DL:
# on x86_64, ggml/src/ggml-cpu/CMakeLists.txt:374 raises a hard
# `FATAL_ERROR "GGML_NATIVE is not compatible with GGML_BACKEND_DL"` during
# configure (the feature check depends on ARCH_DEFINITIONS, which
# GGML_NATIVE mode never sets). arm64 happens to skip that check and builds
# anyway, but the underlying incompatibility is the same -- the right fix is
# to turn it off everywhere, not rely on arm64's missing guard. Two further
# reasons this is the right default for a published wheel even aside from
# the DL conflict: (1) -march=native bakes in the *build* machine's CPU
# instruction set, so a wheel built on, say, Apple M4 or x86 zen4 can SIGILL
# on an older host; (2) llama.cpp's own reproducible-build path already
# forces GGML_NATIVE off via SOURCE_DATE_EPOCH/CMAKE_CROSSCOMPILING for the
# same reason. llama.cpp recommends GGML_CPU_ALL_VARIANTS for per-host CPU
# autoselection under DL, but we don't need that complexity yet -- a single
# portable CPU backend is fine for this library's workload.
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)

# OpenMP: platform-differential.
#
# Linux: keep ON (default). Empirically verified on a Linux x86_64 server
# (pure CPU, two_step_extract, 10 rounds × 30 images, strict ON/OFF
# alternation): OMP_ON avg 487s vs OMP_OFF avg 557s -- turning it off costs
# ~14% throughput. ggml-cpu has its own pthread threadpool (ggml_threadpool)
# that takes over when OpenMP is off, but on Linux libgomp's OpenMP
# implementation is measurably faster than the pthread fallback (likely
# better-tuned sync primitives). The libgomp.so.1 runtime it pulls in is
# bundled into the wheel (see install logic below) so users don't need it
# preinstalled.
#
# macOS / Windows: force OFF. macOS uses Metal (GPU) as its primary path, so
# CPU OpenMP isn't worth the libgomp.dylib dependency (which would need
# Homebrew gcc at build time and bake in a /opt/homebrew path). Windows
# ARM64 showed zero perf delta (206.3s vs 206.0s, 3 runs each), and dropping
# OpenMP removes the libomp140.aarch64.dll dependency that otherwise breaks
# import on machines without VS BuildTools installed.
if (APPLE OR WIN32)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
endif()
# On Linux, GGML_OPENMP defaults to ON (llama.cpp's own default) -- we
# deliberately do NOT force it, so a builder can still override with
# -DGGML_OPENMP=OFF if they want a libgomp-free build.

# Keep native runtime artifacts scoped inside the Python package directory
# instead of at the wheel root, so the backend MODULEs (discovered via
# load_packaged_backends()) and the SONAME libs (found via RPATH) both
# travel inside mineru_llama_cpp/ as one relocatable unit. llama-server and
# the backend MODULEs share bin/ (both installed via CMAKE_INSTALL_BINDIR --
# see ggml/src/CMakeLists.txt's ggml_add_backend_library()), so llama.cpp's
# own executable-relative backend search also works when running the
# packaged llama-server standalone, outside of Python.
#
# These only affect install()-time layout (regular/wheel builds); editable
# "inplace" builds never run `cmake --install` (see src/cpp/CMakeLists.txt's
# comment on LIBRARY_OUTPUT_DIRECTORY) and instead place build-tree
# artifacts at the repo root's bin/ -- scikit-build-core's inplace editable
# mode sets CMAKE_BINARY_DIR to CMAKE_SOURCE_DIR itself, ignoring pyproject
# .toml's build-dir setting entirely (see get_build_dir() in
# scikit_build_core/build/common_wheel_helpers.py). load_packaged_backends()
# accounts for both layouts.
set(CMAKE_INSTALL_BINDIR "mineru_llama_cpp/bin" CACHE PATH "" FORCE)
set(CMAKE_INSTALL_LIBDIR "mineru_llama_cpp/lib" CACHE PATH "" FORCE)

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
#
# Windows (else branch below) has no $ORIGIN equivalent -- DLL resolution
# there uses the default search order (loaded-module's directory, then system
# dirs, then PATH). Leaving CMAKE_INSTALL_RPATH empty on Windows is
# intentional: our .pyd and the SONAME DLLs in lib/ resolve via the
# loaded-module-directory rule, and backend MODULEs are dlopen'd by
# load_packaged_backends() with an explicit path (no search needed). This is
# the part most likely to need adjustment after a real Windows build --
# verify with `dumpbin /dependents` on the installed .pyd.
if (APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../lib")
elseif (WIN32)
    # Windows: no RPATH mechanism; rely on default DLL search + load_packaged_backends().
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
# Exclude macOS AppleDouble files (._*.patch) that Finder/certain editors
# create alongside the real patches -- `git apply` would fail on them
# otherwise, breaking configure on macOS dev machines.
list(FILTER LLAMA_CPP_PATCHES EXCLUDE REGEX "/\\._[^/]*\\.patch$")
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

# Bundle libgomp.so.1 into the Linux wheel. Done at top-level scope (after
# add_subdirectory(third_party/llama.cpp)) so ggml's OpenMP state is visible.
#
# Why GGML_OPENMP_ENABLED (not TARGET OpenMP::OpenMP_CXX or OpenMP_FOUND):
# ggml/src/CMakeLists.txt sets GGML_OPENMP_ENABLED as CACHE INTERNAL -- the
# only form that's truly globally visible across CMake's subdirectory scopes.
# OpenMP_FOUND is a plain variable scoped to ggml/src (doesn't propagate up).
# TARGET OpenMP::OpenMP_CXX is an imported target created in ggml/src; its
# default scope is NOT global (IMPORTED_GLOBAL defaults to false in CMake
# 3.x), so it's invisible at top-level -- confirmed empirically: the wheel
# build's isolated configure logged "OpenMP: not enabled... target check
# failed" even though ggml had linked OpenMP. GGML_OPENMP_ENABLED doesn't
# have that problem.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND GGML_OPENMP_ENABLED)
    message(STATUS "OpenMP: ENABLED on Linux (GGML_OPENMP_ENABLED), locating libgomp.so.1 for bundling")
    find_library(LIBGOMP_RUNTIME NAMES libgomp.so.1
                 PATHS /usr/lib /lib /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu
                       /usr/lib/aarch64-linux-gnu /lib/aarch64-linux-gnu
                 NO_DEFAULT_PATH)
    if (NOT LIBGOMP_RUNTIME)
        find_library(LIBGOMP_RUNTIME NAMES libgomp.so.1)
    endif()
    if (LIBGOMP_RUNTIME)
        message(STATUS "Bundling libgomp: ${LIBGOMP_RUNTIME}")
        install(FILES "${LIBGOMP_RUNTIME}" DESTINATION ${CMAKE_INSTALL_LIBDIR})
    else()
        message(WARNING "OpenMP is ON but libgomp.so.1 was not found -- "
                "the wheel will depend on the system's libgomp.so.1 at "
                "runtime. Install libgomp1 / libgomp on the build machine "
                "to bundle it.")
    endif()
else()
    message(STATUS "OpenMP: not enabled (GGML_OPENMP_ENABLED=${GGML_OPENMP_ENABLED}, platform=${CMAKE_SYSTEM_NAME}), skipping libgomp bundling")
endif()

add_subdirectory(src/cpp)
