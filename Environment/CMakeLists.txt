cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Environment LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

find_package(GLEW REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/../Paths.cmake)

#### CollisionChecker ####
add_library(CollisionChecker STATIC
    CollisionChecker.cpp
    CollisionChecker.cu
)
set_target_properties(CollisionChecker PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(CollisionChecker PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(CollisionChecker PRIVATE 
    "/home/s0001734/Downloads/cuda/targets/x86_64-linux/include"    
)
target_link_libraries(CollisionChecker PRIVATE
    ${CUDA_LIBRARIES}
    GL
)
set_target_properties(CollisionChecker PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "50"
)

#### Main Environment ####
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
    ${RAYLIB_INCLUDE_DIR}
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${SHARED_MEM_LIB_DIR}
)

target_link_libraries(Environment PRIVATE 
    raylib dl pthread rt
    CollisionChecker
)
target_link_directories(Environment PUBLIC ${RAYLIB_LINK_DIR})
target_compile_options(Environment PRIVATE ${COMPILER_FLAGS})
