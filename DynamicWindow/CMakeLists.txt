cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dynamic-window-planning)

set(COMPILER_FLAGS -Wall -std=c++17 -O2 -ldl -pthread -lrt)

include(${CMAKE_CURRENT_SOURCE_DIR}/../Paths.cmake)

set(SOURCES 
    dynamic_window_sim.cpp 
    Robot.cpp
    Npc.cpp
)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})
target_include_directories(${CMAKE_PROJECT_NAME} SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
)
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE ${RAYLIB_LINK_DIR})
target_link_libraries(${CMAKE_PROJECT_NAME} pthread raylib dl rt pthread)
target_compile_options(${CMAKE_PROJECT_NAME} PUBLIC ${COMPILER_FLAGS})

