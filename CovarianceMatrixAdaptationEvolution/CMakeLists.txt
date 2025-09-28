cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(CMA_ES)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(COMPILER_FLAGS -Wall -std=c++17 -g -O2)

################# Common Environment Libraries #################
include(${CMAKE_CURRENT_SOURCE_DIR}/../Paths.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Environment ${CMAKE_CURRENT_BINARY_DIR}/Environment)
set(ENVIRONMENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

################## cma es ################## 
set(CMA_ES_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/CmaEsSolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Controller.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

add_executable(cma_es ${CMA_ES_SRCS})

target_include_directories(cma_es SYSTEM PRIVATE
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${SHARED_MEM_LIB_DIR}
    ${EIGEN_DIR}
)

target_link_libraries(cma_es "${TORCH_LIBRARIES}" pthread raylib dl rt Environment)
target_link_directories(cma_es PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(cma_es PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ENVIRONMENT_INCLUDE_DIR}
)

target_compile_options(cma_es PUBLIC ${COMPILER_FLAGS})