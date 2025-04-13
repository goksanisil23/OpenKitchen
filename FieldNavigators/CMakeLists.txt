cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(field_navigators)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(COMPILER_FLAGS -Wall -std=c++17 -g -O2)

################# Common Environment Libraries #################
include(${CMAKE_CURRENT_SOURCE_DIR}/../Paths.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Environment ${CMAKE_CURRENT_BINARY_DIR}/Environment)
set(ENVIRONMENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

################## Field Navigators ################## 
set(FIELD_NAV_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/PotentialFieldAgent.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/VFHAgent.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

add_executable(field_navigator ${FIELD_NAV_SRCS})

target_include_directories(field_navigator SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${SHARED_MEM_LIB_DIR}
    ${EIGEN_DIR}
)

target_link_libraries(field_navigator pthread raylib dl rt Environment)
target_link_directories(field_navigator PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(field_navigator PUBLIC
${CMAKE_CURRENT_SOURCE_DIR}
${ENVIRONMENT_INCLUDE_DIR}
)

target_compile_options(field_navigator PUBLIC ${COMPILER_FLAGS})