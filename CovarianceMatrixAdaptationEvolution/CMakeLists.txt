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
set(CMA_ES_EIGEN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/CmaEsSolverEigen.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Controller.cpp
)

set(CMA_ES_TORCH_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/CmaEsSolverTorch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Controller.cpp
)


add_executable(cma_es_eigen main_eigen.cpp ${CMA_ES_EIGEN_SRCS})
add_executable(cma_es_torch main_torch.cpp ${CMA_ES_TORCH_SRCS})

target_include_directories(cma_es_eigen SYSTEM PRIVATE
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${SHARED_MEM_LIB_DIR}
    ${EIGEN_DIR}
)

target_include_directories(cma_es_torch SYSTEM PRIVATE
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${SHARED_MEM_LIB_DIR}
)

target_link_libraries(cma_es_eigen "${TORCH_LIBRARIES}" pthread raylib dl rt Environment)
target_link_directories(cma_es_eigen PRIVATE ${RAYLIB_LINK_DIR})

target_link_libraries(cma_es_torch "${TORCH_LIBRARIES}" pthread raylib dl rt Environment)
target_link_directories(cma_es_torch PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(cma_es_eigen PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ENVIRONMENT_INCLUDE_DIR}
)
target_compile_options(cma_es_eigen PUBLIC ${COMPILER_FLAGS})

target_include_directories(cma_es_torch PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ENVIRONMENT_INCLUDE_DIR}
)
target_compile_options(cma_es_torch PUBLIC ${COMPILER_FLAGS})