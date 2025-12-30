cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(collect_data_racetrack)

find_package(CUDA REQUIRED)
set(COMPILER_FLAGS -Wall -std=c++17 -g -O2)

################# Common Environment Libraries #################
include(${CMAKE_CURRENT_SOURCE_DIR}/../../Paths.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../Environment ${CMAKE_CURRENT_BINARY_DIR}/Environment)
set(ENVIRONMENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../)

################## Data Collection ##################
set(COLLECT_DATA_COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../PotentialFieldAgent.hpp
)

function(add_collect_data_target target_name main_source)
    add_executable(${target_name}
        ${main_source}
        ${COLLECT_DATA_COMMON_SOURCES}
    )

    target_include_directories(${target_name} SYSTEM PRIVATE
        ${RAYLIB_CPP_INCLUDE_DIR}
        ${RAYLIB_INCLUDE_DIR}
        ${SHARED_MEM_LIB_DIR}
        ${EIGEN_DIR}
    )

    target_link_libraries(${target_name} raylib pthread dl rt Environment ${CUDA_LIBRARIES})
    target_link_directories(${target_name} PRIVATE ${RAYLIB_LINK_DIR})

    target_include_directories(${target_name} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ENVIRONMENT_INCLUDE_DIR}
    )

    target_compile_options(${target_name} PUBLIC ${COMPILER_FLAGS})
endfunction()

add_collect_data_target(collect_data_3_lanes ${CMAKE_CURRENT_SOURCE_DIR}/collect_data_3_lanes.cpp)
add_collect_data_target(collect_data_random ${CMAKE_CURRENT_SOURCE_DIR}/collect_data_random.cpp)
