#!/bin/bash

export RAYLIB_CPP_INCLUDE_DIR="/home/s0001734/Downloads/raylib-cpp/include"
export RAYLIB_INCLUDE_DIR="/home/s0001734/Downloads/raylib-cpp/build/_deps/raylib-build/raylib/include"
export RAYLIB_LINK_DIR="/home/s0001734/Downloads/raylib-cpp/build/_deps/raylib-build/raylib"

g++ -o topo_sort topo_sort.cpp \
-isystem $RAYLIB_CPP_INCLUDE_DIR -I $RAYLIB_INCLUDE_DIR -L $RAYLIB_LINK_DIR -lraylib \
-Wall -std=c++17 -O2 -ldl -pthread
