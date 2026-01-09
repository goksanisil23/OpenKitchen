cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(world_model_racer)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(COMPILER_FLAGS -Wall -std=c++17 -g -O2)

################# Common Environment Libraries #################
include(${CMAKE_CURRENT_SOURCE_DIR}/../Paths.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Environment ${CMAKE_CURRENT_BINARY_DIR}/Environment)
set(ENVIRONMENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

################## CMA-ES  ##################

add_library(CMA_ES STATIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../CovarianceMatrixAdaptationEvolution/CmaEsSolverTorch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../CovarianceMatrixAdaptationEvolution/Controller.cpp
)
target_include_directories(CMA_ES PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../CovarianceMatrixAdaptationEvolution/
)
target_include_directories(CMA_ES SYSTEM PRIVATE 
    ${EIGEN_DIR}
)
target_link_libraries(CMA_ES PRIVATE 
    "${TORCH_LIBRARIES}"
)


############ World Model ############
set(WORLD_MODEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

add_executable(world_model_racer ${WORLD_MODEL_SRCS})

target_include_directories(world_model_racer SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${SHARED_MEM_LIB_DIR}
    ${EIGEN_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../CovarianceMatrixAdaptationEvolution/
)

target_link_libraries(world_model_racer "${TORCH_LIBRARIES}" pthread raylib dl rt Environment ${OpenCV_LIBS} CMA_ES)
target_link_directories(world_model_racer PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(world_model_racer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ENVIRONMENT_INCLUDE_DIR}
)

target_compile_options(world_model_racer PUBLIC ${COMPILER_FLAGS})

####### TEST ########

add_executable(test_export test_export.cpp)

target_include_directories(test_export SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${SHARED_MEM_LIB_DIR}
    ${EIGEN_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_export "${TORCH_LIBRARIES}" pthread raylib dl rt Environment ${OpenCV_LIBS})
target_link_directories(test_export PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(test_export PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ENVIRONMENT_INCLUDE_DIR}
)

target_compile_options(test_export PUBLIC ${COMPILER_FLAGS})

### PPO

add_executable(ppo ${CMAKE_CURRENT_SOURCE_DIR}/main_ppo.cpp)
target_include_directories(ppo SYSTEM PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ppo "${TORCH_LIBRARIES}" pthread raylib dl rt Environment ${OpenCV_LIBS})
target_link_directories(ppo PRIVATE ${RAYLIB_LINK_DIR})

target_include_directories(ppo PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ENVIRONMENT_INCLUDE_DIR}
)

target_compile_options(ppo PUBLIC ${COMPILER_FLAGS})