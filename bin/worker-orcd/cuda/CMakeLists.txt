cmake_minimum_required(VERSION 3.18)

# Set CUDA paths before project() if not already set
# Common CUDA installation locations
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    if(EXISTS "/opt/cuda/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")
    elseif(EXISTS "/usr/local/cuda/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    endif()
endif()

if(NOT DEFINED CUDAToolkit_ROOT)
    if(EXISTS "/opt/cuda")
        set(CUDAToolkit_ROOT "/opt/cuda")
    elseif(EXISTS "/usr/local/cuda")
        set(CUDAToolkit_ROOT "/usr/local/cuda")
    endif()
endif()

project(worker_cuda LANGUAGES CXX CUDA)

# Find CUDA toolkit explicitly
find_package(CUDAToolkit REQUIRED)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures (CUDA 13+ only supports 75+)
# FIX (2025-10-04): Removed compute_70 (Volta) - unsupported in CUDA 13
# 75 = Turing, 80/86 = Ampere, 89 = Ada, 90 = Hopper
# -- Cascade
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

# Source files
set(CUDA_SOURCES
    src/ffi.cpp
    src/context.cpp
    src/model.cpp
    src/inference.cu
    src/health.cpp
    src/errors.cpp
    src/utils.cpp
    src/vram_tracker.cpp
    src/device_memory.cpp
    src/cublas_wrapper.cpp
    src/rng.cpp
    src/kv_cache.cpp
    src/gguf/header_parser.cpp
    src/gguf/llama_metadata.cpp
    src/io/mmap_file.cpp
    src/io/chunked_transfer.cpp
    src/validation/pre_load.cpp
    src/model/arch_detect.cpp
    src/model/gpt_weights.cpp
    src/model/gpt_model.cpp
    src/gpt_transformer_layer.cpp
)

# Kernel files
set(KERNEL_SOURCES
    kernels/attention.cu
    kernels/gemm.cu
    kernels/sampling.cu
    kernels/rope.cu
    kernels/rmsnorm.cu
    kernels/residual.cu
    kernels/gqa_attention.cu
    kernels/swiglu.cu
    kernels/embedding.cu
    kernels/kv_cache.cu
    # GPT-specific kernels
    kernels/layernorm.cu
    kernels/gelu.cu
    kernels/positional_embedding.cu
    kernels/gpt_ffn.cu
    kernels/mha_attention.cu
    kernels/mxfp4_dequant.cu
)

# Build static library
add_library(worker_cuda STATIC
    ${CUDA_SOURCES}
    ${KERNEL_SOURCES}
)

# Include directories
target_include_directories(worker_cuda
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link CUDA runtime and cuBLAS
target_link_libraries(worker_cuda
    PUBLIC
        CUDA::cudart
        CUDA::cublas
)

# Enable position-independent code for static library
set_target_properties(worker_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Testing
if(BUILD_TESTING)
    enable_testing()
    
    # Find Google Test
    find_package(GTest REQUIRED)
    
    # Test sources
    set(TEST_SOURCES
        tests/test_ffi_interface.cpp
        tests/test_errors.cpp
        tests/test_context.cpp
        tests/test_model.cpp
        tests/test_inference.cpp
        tests/test_health.cpp
        tests/test_vram_tracker.cpp
        tests/test_ffi_integration.cpp
        tests/test_device_memory.cpp
        tests/test_embedding.cu
        tests/test_cublas.cu
        tests/test_sampling.cu
        tests/sampling_advanced_test.cu
        tests/test_rng.cpp
        tests/kv_cache_test.cpp
        tests/test_gguf_header_parser.cpp
        tests/test_gguf_security_fuzzing.cpp
        tests/test_llama_metadata.cpp
        tests/test_mmap_file.cpp
        tests/test_chunked_transfer.cpp
        tests/test_pre_load_validation.cpp
        tests/test_arch_detect.cpp
        tests/test_rope_kernel.cpp
        tests/test_rmsnorm_kernel.cpp
        tests/test_residual_kernel.cpp
        tests/test_gqa_attention.cpp
        tests/test_swiglu.cpp
        # GPT Sprint 4 tests (GT-024, GT-025, GT-026) - Disabled: missing kernel implementations
        # tests/test_gpt_weights.cpp
        # tests/test_gpt_model.cpp
        # GPT Sprint 5 tests (GT-030 - MXFP4 Dequantization)
        tests/test_mxfp4_dequant.cu
        # GPT kernel tests - TODO: Convert to GTest format
        # tests/test_gpt_kernels.cu
        # tests/test_layernorm_comprehensive.cu
        # tests/test_gelu_comprehensive.cu
        # tests/test_gpt_ffn.cu
        # tests/test_mha_attention.cu
    )
    
    # Build test executable
    add_executable(cuda_tests ${TEST_SOURCES})
    
    target_include_directories(cuda_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    
    target_link_libraries(cuda_tests
        PRIVATE
            worker_cuda
            GTest::gtest
            GTest::gtest_main
    )
    
    # Register tests
    include(GoogleTest)
    gtest_discover_tests(cuda_tests)
endif()

# Installation (for Cargo integration)
install(TARGETS worker_cuda
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
