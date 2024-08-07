cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ppo)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(COMPILER_FLAGS -Wall -std=c++17 -g -O2)

################# Common Environment Libraries #################
include(${CMAKE_CURRENT_SOURCE_DIR}/../../Paths.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../Environment ${CMAKE_CURRENT_BINARY_DIR}/Environment)
set(ENVIRONMENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../)

################## PPO-learning ################## 
set(PPO_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/PPOAgent.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ppo_sim.cpp
)

add_executable(ppo ${PPO_SRCS})

target_include_directories(ppo SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${SHARED_MEM_LIB_DIR}
    ${EIGEN_DIR}
)

target_link_libraries(ppo "${TORCH_LIBRARIES}" pthread raylib dl rt Environment)
target_link_directories(ppo PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(ppo PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ENVIRONMENT_INCLUDE_DIR}
)

target_compile_options(ppo PUBLIC ${COMPILER_FLAGS})