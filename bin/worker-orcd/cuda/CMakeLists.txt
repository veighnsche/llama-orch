cmake_minimum_required(VERSION 3.18)
project(worker_cuda LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures (CUDA 13+ only supports 75+)
# FIX (2025-10-04): Removed compute_70 (Volta) - unsupported in CUDA 13
# 75 = Turing, 80/86 = Ampere, 89 = Ada, 90 = Hopper
# -- Cascade
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

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
)

# Kernel files
set(KERNEL_SOURCES
    kernels/attention.cu
    kernels/gemm.cu
    kernels/sampling.cu
    kernels/rope.cu
    kernels/rmsnorm.cu
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
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link CUDA runtime
target_link_libraries(worker_cuda
    PUBLIC
        cudart
)

# Enable position-independent code for static library
set_target_properties(worker_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
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
    )
    
    # Build test executable
    add_executable(cuda_tests ${TEST_SOURCES})
    
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
