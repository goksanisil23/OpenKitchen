cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(deep-sarsa-grid)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(COMPILER_FLAGS -Wall -std=c++17 -O2 -ldl -pthread -lrt)
include(${CMAKE_CURRENT_SOURCE_DIR}/../Paths.cmake)

add_executable(deep_sarsa_grid deep_sarsa_grid.cpp)
target_include_directories(deep_sarsa_grid SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
)
target_link_directories(deep_sarsa_grid PRIVATE ${RAYLIB_LINK_DIR})
target_link_libraries(deep_sarsa_grid 
    "${TORCH_LIBRARIES}" pthread raylib dl rt pthread)

target_compile_options(deep_sarsa_grid PUBLIC ${COMPILER_FLAGS})