cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Environment)

include(${CMAKE_CURRENT_SOURCE_DIR}/../Paths.cmake)

# Define source files for the library
set(ENVIRONMENT_SRCS
    Agent.cpp
    Visualizer.cpp
    Environment.cpp
    RaceTrack.cpp
)

# Compiler flags
set(COMPILER_FLAGS -Wall -std=c++17 -O2)

# Create the library
add_library(Environment STATIC ${ENVIRONMENT_SRCS})

target_include_directories(Environment PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(Environment SYSTEM PRIVATE 
    ${EIGEN_DIR}
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${SHARED_MEM_LIB_DIR}
)

target_link_libraries(Environment PRIVATE raylib dl pthread rt)
target_link_directories(Environment PUBLIC ${RAYLIB_LINK_DIR})
target_compile_options(Environment PRIVATE ${COMPILER_FLAGS})
