cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dino_control)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(COMPILER_FLAGS -Wall -std=c++17 -g -O2)

################# Common Environment Libraries #################
include(${CMAKE_CURRENT_SOURCE_DIR}/../Paths.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Environment ${CMAKE_CURRENT_BINARY_DIR}/Environment)
set(ENVIRONMENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

############ Dino control inference ############
set(DINO_CTRL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/main_dino_control.cpp
)

add_executable(dino_control ${DINO_CTRL_SRCS})

target_include_directories(dino_control SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${SHARED_MEM_LIB_DIR}
    ${EIGEN_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(dino_control "${TORCH_LIBRARIES}" pthread raylib dl rt Environment ${OpenCV_LIBS})
target_link_directories(dino_control PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(dino_control PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ENVIRONMENT_INCLUDE_DIR}
)

target_compile_options(dino_control PUBLIC ${COMPILER_FLAGS})