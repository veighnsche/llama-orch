# TEAM-006: Checkpoint extraction wrapper tool
# Created by: TEAM-006
# Based on: TEAM-005 comprehensive analysis
# Modified by: TEAM-007 - Fixed build to work with llama.cpp build directory

cmake_minimum_required(VERSION 3.14)
project(llorch-checkpoint-extractor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TEAM-007: Find llama.cpp build directory
if(NOT DEFINED LLAMA_BUILD_DIR)
    set(LLAMA_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../reference/llama.cpp/build")
endif()

set(LLAMA_SRC_DIR "${LLAMA_BUILD_DIR}/..")

# Wrapper tool executable
add_executable(llorch-checkpoint-extractor
    src/main.cpp
    src/checkpoint_callback.cpp
)

target_include_directories(llorch-checkpoint-extractor
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${LLAMA_SRC_DIR}/include
    ${LLAMA_SRC_DIR}/ggml/include
)

target_link_directories(llorch-checkpoint-extractor
    PRIVATE
    ${LLAMA_BUILD_DIR}/bin
)

target_link_libraries(llorch-checkpoint-extractor
    PRIVATE
    llama
    ggml
    ggml-base
)

# Install
install(TARGETS llorch-checkpoint-extractor
    RUNTIME DESTINATION bin
)
