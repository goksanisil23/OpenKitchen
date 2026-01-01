cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(bc_main)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(COMPILER_FLAGS -Wall -std=c++17 -g -O2)

################# Common Environment Libraries #################
include(${CMAKE_CURRENT_SOURCE_DIR}/../Paths.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Environment ${CMAKE_CURRENT_BINARY_DIR}/Environment)
set(ENVIRONMENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

################## Behavior Cloning Inference ##################

set(BC_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

add_executable(bc_main ${BC_SRCS})

target_include_directories(bc_main SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${EIGEN_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(bc_main "${TORCH_LIBRARIES}" pthread raylib dl rt Environment ${OpenCV_LIBS})
target_link_directories(bc_main PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(bc_main PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ENVIRONMENT_INCLUDE_DIR}
)

target_compile_options(bc_main PUBLIC ${COMPILER_FLAGS})

