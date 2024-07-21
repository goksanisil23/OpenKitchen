cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(GeneticLearner)
set(BINARY_NAME genetic_learner_sim)

include(${CMAKE_CURRENT_SOURCE_DIR}/../Paths.cmake)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Environment ${CMAKE_CURRENT_BINARY_DIR}/Environment)
set(ENVIRONMENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

set(COMPILER_FLAGS -Wall -std=c++17 -O2 -ldl -pthread -lrt)

# Targets for executables
set(GENETIC_LEARNER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/GeneticAgent.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Mating.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Network.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MiscUtils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/VisUtils.hpp
)

add_executable(${BINARY_NAME} genetic_learner_sim.cpp ${GENETIC_LEARNER_SRCS})

target_link_libraries(${BINARY_NAME} PRIVATE Environment raylib dl pthread rt)

target_include_directories(${BINARY_NAME} SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${SHARED_MEM_LIB_DIR}
    ${EIGEN_DIR}
)

target_include_directories(${BINARY_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ENVIRONMENT_INCLUDE_DIR}
)

target_compile_options(${BINARY_NAME} PUBLIC ${COMPILER_FLAGS})