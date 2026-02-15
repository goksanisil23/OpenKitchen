cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(flow_control)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(COMPILER_FLAGS -Wall -std=c++17 -g -O2)

################# Common Environment Libraries #################
include(${CMAKE_CURRENT_SOURCE_DIR}/../Paths.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Environment ${CMAKE_CURRENT_BINARY_DIR}/Environment)
set(ENVIRONMENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

############ Flow-matching control inference ############
set(FLOW_CTRL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/main_flow_control.cpp
)

add_executable(flow_control ${FLOW_CTRL_SRCS})

target_include_directories(flow_control SYSTEM PRIVATE
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${SHARED_MEM_LIB_DIR}
    ${EIGEN_DIR}
)

target_link_libraries(flow_control "${TORCH_LIBRARIES}" pthread raylib dl rt Environment)
target_link_directories(flow_control PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(flow_control PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ENVIRONMENT_INCLUDE_DIR}
)

target_compile_options(flow_control PUBLIC ${COMPILER_FLAGS})
