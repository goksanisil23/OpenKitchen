cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(transformer_racer)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(COMPILER_FLAGS -Wall -std=c++17 -g -O2)

################# Common Environment Libraries #################
include(${CMAKE_CURRENT_SOURCE_DIR}/../Paths.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Environment ${CMAKE_CURRENT_BINARY_DIR}/Environment)
set(ENVIRONMENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

########### SIMPLE TEST WITH TRACED PYTORCH MODEL ###########
add_executable(infer_torch_traced_test.cpp infer_torch_traced_test.cpp)
target_link_libraries(infer_torch_traced_test.cpp "${TORCH_LIBRARIES}")
target_compile_options(infer_torch_traced_test.cpp PUBLIC ${COMPILER_FLAGS})
###############################
add_executable(infer_torch_traced_main infer_torch_traced_main.cpp)

target_include_directories(infer_torch_traced_main SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${EIGEN_DIR}
)

target_link_libraries(infer_torch_traced_main "${TORCH_LIBRARIES}" pthread raylib dl rt Environment)
target_link_libraries(infer_torch_traced_main "${TORCH_LIBRARIES}")
target_link_directories(infer_torch_traced_main PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(infer_torch_traced_main PUBLIC
${CMAKE_CURRENT_SOURCE_DIR}
${ENVIRONMENT_INCLUDE_DIR}
)

target_compile_options(infer_torch_traced_main PUBLIC ${COMPILER_FLAGS})
