cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(guided_cost_learning)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(COMPILER_FLAGS -Wall -std=c++17 -g -O2)

################# Common Environment Libraries #################
include(${CMAKE_CURRENT_SOURCE_DIR}/../../Paths.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../Environment ${CMAKE_CURRENT_BINARY_DIR}/Environment)
set(ENVIRONMENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../)

###############################
add_executable(${CMAKE_PROJECT_NAME} 
    ReadExpertData.hpp
    Networks.hpp
    GCLAgent.hpp
)

target_include_directories(${CMAKE_PROJECT_NAME} SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${EIGEN_DIR}
)

target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}" pthread raylib dl rt Environment)
target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ENVIRONMENT_INCLUDE_DIR}
)

target_compile_options(${CMAKE_PROJECT_NAME} PUBLIC ${COMPILER_FLAGS})

######## TEST ########
add_executable(test test.cpp)
target_include_directories(test SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${EIGEN_DIR}
)

target_link_libraries(test pthread raylib dl rt Environment)
target_link_directories(test PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(test PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ENVIRONMENT_INCLUDE_DIR}
)

set(COMPILER_FLAGS_TEST -Wall -std=c++17 -g -fno-omit-frame-pointer)
target_compile_options(test PUBLIC ${COMPILER_FLAGS_TEST})