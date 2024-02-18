cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(vfh)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(COMPILER_FLAGS -Wall -std=c++17 -g -O2)

################# Common Environment Libraries #################
include(${CMAKE_CURRENT_SOURCE_DIR}/../Paths.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Environment ${CMAKE_CURRENT_BINARY_DIR}/Environment)
set(ENVIRONMENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

################## Q-learning ################## 
set(VFH_SRCS
    ${ENVIRONMENT_INCLUDE_DIR}/Environment/Agent.cpp
    ${ENVIRONMENT_INCLUDE_DIR}/Environment/Visualizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PotentialFieldAgent.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/VFHAgent.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Environment.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

add_executable(vfh ${VFH_SRCS})

target_include_directories(vfh SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${SHARED_MEM_LIB_DIR}
    ${EIGEN_DIR}
)

target_link_libraries(vfh pthread raylib dl rt)
target_link_directories(vfh PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(vfh PUBLIC
${CMAKE_CURRENT_SOURCE_DIR}
${ENVIRONMENT_INCLUDE_DIR}
)

target_compile_options(vfh PUBLIC ${COMPILER_FLAGS})