#!/bin/bash

export RAYLIB_CPP_INCLUDE_DIR="/home/s0001734/Downloads/raylib-cpp/include"
export RAYLIB_INCLUDE_DIR="/home/s0001734/Downloads/raylib-cpp/build/_deps/raylib-build/raylib/include"
export RAYLIB_LINK_DIR="/home/s0001734/Downloads/raylib-cpp/build/_deps/raylib-build/raylib"

export OKITCH_UTILS_DIR="/home/s0001734/Downloads/OpenKitchen/Utilities"
export SHARED_MEM_LIB_DIR="/home/s0001734/Downloads/OpenKitchen/SharedMemory"

export EIGEN_DIR="/usr/include/eigen3"

g++ --std=c++17 -o env_sim env_sim.cpp \
-isystem $RAYLIB_CPP_INCLUDE_DIR -I $RAYLIB_INCLUDE_DIR -I $OKITCH_UTILS_DIR -I . -I $SHARED_MEM_LIB_DIR \
-L $RAYLIB_LINK_DIR -lraylib \
-I $EIGEN_DIR \
-Wall -std=c++17 -O2 -ldl -pthread -lrt


g++ -o laser_reader laser_reader.cpp \
-isystem $RAYLIB_CPP_INCLUDE_DIR -I $RAYLIB_INCLUDE_DIR -I $OKITCH_UTILS_DIR -I . -I $SHARED_MEM_LIB_DIR -L $RAYLIB_LINK_DIR -lraylib \
-Wall -std=c++17 -O2 -ldl -pthread -lrt