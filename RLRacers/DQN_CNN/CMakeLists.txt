cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(deep-q-learning)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(COMPILER_FLAGS -Wall -std=c++17 -g -O2)

################# Common Environment Libraries #################
include(${CMAKE_CURRENT_SOURCE_DIR}/../../Paths.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../Environment ${CMAKE_CURRENT_BINARY_DIR}/Environment)
set(ENVIRONMENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../)

################## Deep-Q-learning ################## 
set(DQ_CNN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/DQCNNAgent.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CNN.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dq_cnn_racer_sim.cpp
)

add_executable(dq_cnn ${DQ_CNN_SRCS})

target_include_directories(dq_cnn SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${SHARED_MEM_LIB_DIR}
    ${EIGEN_DIR}
)

target_link_libraries(dq_cnn "${TORCH_LIBRARIES}" pthread raylib dl rt Environment)
target_link_directories(dq_cnn PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(dq_cnn PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ENVIRONMENT_INCLUDE_DIR}
)

target_compile_options(dq_cnn PUBLIC ${COMPILER_FLAGS})