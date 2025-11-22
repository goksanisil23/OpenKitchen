cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Environment LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

find_package(GLEW REQUIRED)
find_package(CUDA REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/../Paths.cmake)

include(FetchContent)
FetchContent_Declare(fpng_repo
  GIT_REPOSITORY https://github.com/richgel999/fpng.git
  GIT_TAG        main)
FetchContent_MakeAvailable(fpng_repo)

add_library(fpng STATIC ${fpng_repo_SOURCE_DIR}/src/fpng.cpp)
target_include_directories(fpng PUBLIC ${fpng_repo_SOURCE_DIR}/src)

# x86/x64: enable SSE4.1 + PCLMUL on GCC/Clang; others: disable SSE in fpng
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(fpng PRIVATE -msse4.1 -mpclmul -fno-strict-aliasing)
  endif()
else()
  target_compile_definitions(fpng PUBLIC FPNG_NO_SSE=1)
endif()

#### CollisionChecker ####
add_library(CollisionChecker STATIC
    ScreenGrabber.cpp
    ScreenGrabber.cu
    CollisionChecker.cpp
    CollisionChecker.cu
)
set_target_properties(CollisionChecker PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(CollisionChecker PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(CollisionChecker PRIVATE 
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/External
)
target_link_libraries(CollisionChecker PRIVATE
    ${CUDA_LIBRARIES}
    GL
    GLEW
    fpng
)
set_target_properties(CollisionChecker PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "89"
)

#### Main Environment ####
set(ENVIRONMENT_SRCS
    Agent.cpp
    Visualizer.cpp
    Environment.cpp
    RaceTrack.cpp
)

# Compiler flags
set(COMPILER_FLAGS -Wall -std=c++17 -O2)

# Create the library
add_library(Environment STATIC ${ENVIRONMENT_SRCS})

target_include_directories(Environment PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(Environment SYSTEM PRIVATE 
    ${EIGEN_DIR}
    ${RAYLIB_INCLUDE_DIR}
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${SHARED_MEM_LIB_DIR}
)

target_link_libraries(Environment PRIVATE 
    raylib dl pthread rt
    CollisionChecker
)
target_link_directories(Environment PUBLIC ${RAYLIB_LINK_DIR})
target_compile_options(Environment PRIVATE ${COMPILER_FLAGS})
