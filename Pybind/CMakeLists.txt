cmake_minimum_required(VERSION 3.18)
project(open_kitchen_pybind LANGUAGES CXX)

set(COMPILER_FLAGS -Wall -std=c++17 -g -O2)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v3.0.1
)
FetchContent_MakeAvailable(pybind11)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(${CMAKE_CURRENT_SOURCE_DIR}/../Paths.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Environment ${CMAKE_CURRENT_BINARY_DIR}/Environment)
set(ENVIRONMENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)


add_library(open_kitchen_pybind MODULE bindings.cpp)

target_link_libraries(open_kitchen_pybind PRIVATE pybind11::module Environment pthread raylib dl rt)
target_include_directories(open_kitchen_pybind PRIVATE 
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}
)
target_link_directories(open_kitchen_pybind PRIVATE ${RAYLIB_LINK_DIR})
target_include_directories(open_kitchen_pybind PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ENVIRONMENT_INCLUDE_DIR}
)

pybind11_extension(open_kitchen_pybind)
pybind11_strip(open_kitchen_pybind)

set_target_properties(open_kitchen_pybind PROPERTIES
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
)