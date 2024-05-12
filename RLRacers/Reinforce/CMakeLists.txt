cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(reinforce)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(COMPILER_FLAGS -Wall -std=c++17 -g -O2)

################# Common Environment Libraries #################
include(${CMAKE_CURRENT_SOURCE_DIR}/../../Paths.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../Environment ${CMAKE_CURRENT_BINARY_DIR}/Environment)
set(ENVIRONMENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../)

################## Q-learning ################## 
set(REINFORCE_SRCS
    ${ENVIRONMENT_INCLUDE_DIR}/Environment/Agent.cpp
    ${ENVIRONMENT_INCLUDE_DIR}/Environment/Visualizer.cpp
    ${ENVIRONMENT_INCLUDE_DIR}/Environment/Environment.hpp    
    ${CMAKE_CURRENT_SOURCE_DIR}/ReinforceAgent.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reinforce_sim.cpp
)

add_executable(reinforce ${REINFORCE_SRCS})

target_include_directories(reinforce SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${SHARED_MEM_LIB_DIR}
    ${EIGEN_DIR}
)

target_link_libraries(reinforce "${TORCH_LIBRARIES}" pthread raylib dl rt)
target_link_directories(reinforce PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(reinforce PUBLIC
${CMAKE_CURRENT_SOURCE_DIR}
${ENVIRONMENT_INCLUDE_DIR}
)

target_compile_options(reinforce PUBLIC ${COMPILER_FLAGS})