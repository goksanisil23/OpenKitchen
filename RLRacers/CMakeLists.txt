cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(deep-sarsa-racer)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(COMPILER_FLAGS -Wall -std=c++17 -g -O2 -ldl -pthread -lrt)

################# Common Environment Libraries #################
include(${CMAKE_CURRENT_SOURCE_DIR}/../Paths.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Environment ${CMAKE_CURRENT_BINARY_DIR}/Environment)
set(ENVIRONMENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

################# Example Torch App #################
add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}" pthread)
target_compile_options(example-app PUBLIC ${COMPILER_FLAGS})

################## Deep Sarsa ################## 
# set(DEEP_SARSA_SRCS
#     ${ENVIRONMENT_INCLUDE_DIR}/Environment/Agent.cpp
#     ${ENVIRONMENT_INCLUDE_DIR}/Environment/Visualizer.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/DeepSarsaAgent.hpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/GreedyAgent.hpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/Environment.hpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/Network.hpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/deep_sarsa_racer_sim.cpp
# )

# add_executable(${CMAKE_PROJECT_NAME} ${DEEP_SARSA_SRCS})

# target_include_directories(${CMAKE_PROJECT_NAME} SYSTEM PRIVATE 
#     ${RAYLIB_CPP_INCLUDE_DIR}
#     ${RAYLIB_INCLUDE_DIR}
#     ${SHARED_MEM_LIB_DIR}
#     ${EIGEN_DIR}
# )

# target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}" pthread raylib dl rt)
# target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE ${RAYLIB_LINK_DIR})

# target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC
# ${CMAKE_CURRENT_SOURCE_DIR}
# ${ENVIRONMENT_INCLUDE_DIR}
# )

# target_compile_options(${CMAKE_PROJECT_NAME} PUBLIC ${COMPILER_FLAGS})



##################

# add_executable(sarsa_example sarsa_example.cpp)
# target_include_directories(sarsa_example SYSTEM PRIVATE 
#     ${RAYLIB_CPP_INCLUDE_DIR}
#     ${RAYLIB_INCLUDE_DIR}
# )
# target_link_directories(sarsa_example PRIVATE ${RAYLIB_LINK_DIR})
# target_link_libraries(sarsa_example pthread raylib dl rt pthread)

