cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(e2e_supervised_racer)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(COMPILER_FLAGS -Wall -std=c++17 -g -O2)

################# Common Environment Libraries #################
include(${CMAKE_CURRENT_SOURCE_DIR}/../Paths.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Environment ${CMAKE_CURRENT_BINARY_DIR}/Environment)
set(ENVIRONMENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

##################  ################## 
set(BIRDSEYE_RACER_SRCS
    ${ENVIRONMENT_INCLUDE_DIR}/Environment/Agent.cpp
    ${ENVIRONMENT_INCLUDE_DIR}/Environment/Visualizer.cpp
    ${ENVIRONMENT_INCLUDE_DIR}/Environment/Environment.hpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/AttentionAgent.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python_birdseye_infer_agent_main.cpp
)

add_executable(e2e_supervised_racer ${BIRDSEYE_RACER_SRCS})

target_include_directories(e2e_supervised_racer SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${SHARED_MEM_LIB_DIR}
    ${EIGEN_DIR}
)

target_link_libraries(e2e_supervised_racer "${TORCH_LIBRARIES}" pthread raylib dl rt)
target_link_directories(e2e_supervised_racer PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(e2e_supervised_racer PUBLIC
${CMAKE_CURRENT_SOURCE_DIR}
${ENVIRONMENT_INCLUDE_DIR}
)

target_compile_options(e2e_supervised_racer PUBLIC ${COMPILER_FLAGS})