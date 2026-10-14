cmake_minimum_required(VERSION 3.28)

# Backend selection: CUDA (upstream default) or HIP (gfx1100 port via compat/cuda shim).
set(NINFER_BACKEND "CUDA" CACHE STRING "GPU backend: CUDA or HIP")
set_property(CACHE NINFER_BACKEND PROPERTY STRINGS CUDA HIP)

if(NINFER_BACKEND STREQUAL "HIP")
  # Language, architecture and source extensions must be fixed before project() so CMake
  # never injects a generic host-toolchain default and routes .cu files to the HIP compiler.
  set(CMAKE_HIP_ARCHITECTURES gfx1100 CACHE STRING "HIP architectures to build")
  set(CMAKE_HIP_SOURCE_FILE_EXTENSIONS "cu;hip" CACHE STRING "HIP source file extensions")
  set(CMAKE_HIP_STANDARD 20)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  project(ninfer LANGUAGES C CXX HIP)
  # gfx1100 has no FP8/FP4 tensor cores, so the SM8X compatibility routes (A16
  # dequantizing, INT8 KV) apply unchanged; NINFER_HIP marks the backend in code.
  add_compile_definitions(NINFER_HIP=1 NINFER_SM8X_COMPAT=1 __HIP_PLATFORM_AMD__=1)
  # CMake only auto-adds -x hip for .hip sources; per-source properties (not global
  # options, whose generator expressions leak into dependency-scan commands on 4.3)
  # carry it for .cu files without touching link commands. Without this marking a target that
  # owns .cu sources gets no compile rule at all and links with zero objects.
  function(ninfer_mark_hip_sources target)
    get_target_property(_ninfer_marked_srcs ${target} SOURCES)
    foreach(_ninfer_src ${_ninfer_marked_srcs})
      if(_ninfer_src MATCHES "\.cu$")
        set_source_files_properties(${_ninfer_src} PROPERTIES
          LANGUAGE HIP
          COMPILE_OPTIONS "-x;hip")
      endif()
    endforeach()
  endfunction()
  # compat/cuda must win the <cuda_*.h> name lookup before any system include path.
  include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/compat/cuda)
  find_package(hip REQUIRED)
  set(NINFER_NVTX_TARGET "")
  set(NINFER_CUDART_TARGET hip::host)
else()
# The current implementation supports Ampere sm_86 and early Ada sm_89. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(86|89)$")
  message(FATAL_ERROR
    "NInfer supports CMAKE_CUDA_ARCHITECTURES=86 or 89; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)
endif()

# NINFER_SM8X_COMPAT gates every Ampere/early-Ada compatibility path. ninfer_ops also sets it
# PUBLIC for its own consumers, but that link is PRIVATE through ninfer_engine, so apps and tests
# would not otherwise see it. Defining it globally keeps a single consistent answer for every
# target, including apps/ninfer-perplexity, whose upstream default KV profile is FP8.
# (HIP builds set NINFER_SM8X_COMPAT alongside NINFER_HIP before project above.)
if(NOT NINFER_BACKEND STREQUAL "HIP")
if(CMAKE_CUDA_ARCHITECTURES MATCHES "^(86|89)$")
  add_compile_definitions(NINFER_SM8X_COMPAT=1)
endif()
endif()

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(MSVC)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
  # /Zc:__cplusplus makes MSVC report the standard it is actually compiling. Without it the macro
  # stays at 199711L even under /std:c++20, and CUDA's <vector_types.h> then hides the constexpr
  # dim3 constructors behind its own __cplusplus >= 201103L test, which fails to compile every
  # translation unit that builds a dim3 in a constant expression.
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>)
  # CUDA's static runtime requests LIBCMT even when the application and vcpkg dependencies use
  # the project-wide /MD runtime. Keep one CRT in the final process.
  add_link_options(/NODEFAULTLIB:LIBCMT)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

if(NOT NINFER_BACKEND STREQUAL "HIP")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR
    "NInfer requires CUDA 12.8 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()
endif()

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

if(NOT NINFER_BACKEND STREQUAL "HIP")
find_package(CUDAToolkit REQUIRED)
if(WIN32)
  set(NINFER_CUDART_TARGET CUDA::cudart_static)
else()
  set(NINFER_CUDART_TARGET CUDA::cudart)
endif()
set(NINFER_NVTX_TARGET CUDA::nvtx3)
endif()
if(WIN32 OR DEFINED VCPKG_TARGET_TRIPLET)
  find_package(FFMPEG REQUIRED)
  add_library(ninfer_ffmpeg_dependencies INTERFACE)
  target_include_directories(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_INCLUDE_DIRS})
  target_link_directories(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARY_DIRS})
  target_link_libraries(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARIES})
  set(NINFER_FFMPEG_TARGET ninfer_ffmpeg_dependencies)
  if(NINFER_BUILD_MEDIA_ACQUIRE)
    find_package(CURL 7.85 REQUIRED)
    set(NINFER_CURL_TARGET CURL::libcurl)
  endif()
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
  set(NINFER_FFMPEG_TARGET PkgConfig::FFMPEG)
  if(NINFER_BUILD_MEDIA_ACQUIRE)
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
    set(NINFER_CURL_TARGET PkgConfig::LIBCURL)
  endif()
endif()
find_package(Threads REQUIRED)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
