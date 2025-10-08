# TEAM-006: Checkpoint extraction wrapper tool
# Created by: TEAM-006
# Based on: TEAM-005 comprehensive analysis

cmake_minimum_required(VERSION 3.14)
project(llorch-checkpoint-extractor)

# Find llama.cpp library
find_package(llama REQUIRED)

# Wrapper tool executable
add_executable(llorch-checkpoint-extractor
    src/main.cpp
    src/checkpoint_callback.cpp
)

target_link_libraries(llorch-checkpoint-extractor
    PRIVATE
    llama
)

target_include_directories(llorch-checkpoint-extractor
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Install
install(TARGETS llorch-checkpoint-extractor
    RUNTIME DESTINATION bin
)
