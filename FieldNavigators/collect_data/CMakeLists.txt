cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(collect_data_racetrack)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(COMPILER_FLAGS -Wall -std=c++17 -g -O2)

################# Common Environment Libraries #################
include(${CMAKE_CURRENT_SOURCE_DIR}/../../Paths.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../Environment ${CMAKE_CURRENT_BINARY_DIR}/Environment)
set(ENVIRONMENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../)

################## Q-learning ################## 
set(COLLECT_DATA
    ${CMAKE_CURRENT_SOURCE_DIR}/collect_data_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../PotentialFieldAgent.hpp
)

add_executable(collect_data_racetrack ${COLLECT_DATA})

target_include_directories(collect_data_racetrack SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${SHARED_MEM_LIB_DIR}
    ${EIGEN_DIR}
)

target_link_libraries(collect_data_racetrack raylib pthread dl rt Environment)
target_link_directories(collect_data_racetrack PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(collect_data_racetrack PUBLIC
${CMAKE_CURRENT_SOURCE_DIR}
${ENVIRONMENT_INCLUDE_DIR}
)

target_compile_options(collect_data_racetrack PUBLIC ${COMPILER_FLAGS})