cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(q-learning)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(COMPILER_FLAGS -Wall -std=c++17 -g -O2)

################# Common Environment Libraries #################
include(${CMAKE_CURRENT_SOURCE_DIR}/../../Paths.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../Environment ${CMAKE_CURRENT_BINARY_DIR}/Environment)
set(ENVIRONMENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../)

################## Q-learning ################## 
set(Q_LEARN_SRCS
    ${ENVIRONMENT_INCLUDE_DIR}/Environment/Agent.cpp
    ${ENVIRONMENT_INCLUDE_DIR}/Environment/Visualizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/QAgent.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Environment.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/q_racer_sim.cpp
)

add_executable(q_learn ${Q_LEARN_SRCS})

target_include_directories(q_learn SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${SHARED_MEM_LIB_DIR}
    ${EIGEN_DIR}
)

target_link_libraries(q_learn pthread raylib dl rt)
target_link_directories(q_learn PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(q_learn PUBLIC
${CMAKE_CURRENT_SOURCE_DIR}
${ENVIRONMENT_INCLUDE_DIR}
)

target_compile_options(q_learn PUBLIC ${COMPILER_FLAGS})