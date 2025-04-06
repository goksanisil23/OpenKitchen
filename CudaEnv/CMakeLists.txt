cmake_minimum_required(VERSION 3.12)
project(parallel_agents LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

set(COMPILER_FLAGS -Wall -std=c++17 -g -O2)

include(${CMAKE_CURRENT_SOURCE_DIR}/../Paths.cmake)

find_package(GLEW REQUIRED)

#### Environment ####
add_library(Environment STATIC
    Environment.cu
)
set_target_properties(Environment PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(Environment PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RAYLIB_INCLUDE_DIR}
)
target_link_libraries(Environment    
    raylib 
)
####################
#### DummyBrain ####
add_library(DummyBrain STATIC
    DummyBrain.cpp
    DummyBrain.cu
)
set_target_properties(DummyBrain PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(DummyBrain PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    "/home/s0001734/Downloads/cuda/targets/x86_64-linux/include"
)
####################

add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    "/home/s0001734/Downloads/cuda/targets/x86_64-linux/include"
)

target_link_libraries(${PROJECT_NAME} 
    raylib 
    ${CUDA_LIBRARIES}
    GL 
    ${GLEW_LIBRARIES} 
    Environment
    DummyBrain
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_directories(${PROJECT_NAME} PRIVATE ${RAYLIB_LINK_DIR})

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES "50")
set_property(TARGET Environment PROPERTY CUDA_ARCHITECTURES "50")
set_property(TARGET DummyBrain PROPERTY CUDA_ARCHITECTURES "50")
