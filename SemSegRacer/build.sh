#!/bin/bash

export RAYLIB_CPP_INCLUDE_DIR="/home/s0001734/Downloads/raylib-cpp/include"
export RAYLIB_INCLUDE_DIR="/home/s0001734/Downloads/raylib-cpp/build/_deps/raylib-build/raylib/include"
export RAYLIB_LINK_DIR="/home/s0001734/Downloads/raylib-cpp/build/_deps/raylib-build/raylib"
export OKITCH_UTILS_DIR="/home/s0001734/Downloads/OpenKitchen/Utilities"


g++ -o race_track_gen race_track_gen.cpp \
-isystem $RAYLIB_CPP_INCLUDE_DIR -I $RAYLIB_INCLUDE_DIR -I $OKITCH_UTILS_DIR -I . -L $RAYLIB_LINK_DIR -lraylib \
-Wall -std=c++17 -O2 -ldl -pthread
