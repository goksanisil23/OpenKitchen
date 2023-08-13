#!/bin/bash

export RAYLIB_CPP_INCLUDE_DIR="/home/s0001734/Downloads/raylib-cpp/include"
export RAYLIB_INCLUDE_DIR="/home/s0001734/Downloads/raylib-cpp/build/_deps/raylib-build/raylib/include"
export RAYLIB_LINK_DIR="/home/s0001734/Downloads/raylib-cpp/build/_deps/raylib-build/raylib"

export OKITCH_UTILS_DIR="/home/s0001734/Downloads/OpenKitchen/Utilities"
export SHARED_MEM_LIB_DIR="/home/s0001734/Downloads/OpenKitchen/SharedMemory"

g++ -o race_track_training_gen race_track_training_gen.cpp \
-isystem $RAYLIB_CPP_INCLUDE_DIR -I $RAYLIB_INCLUDE_DIR -I $OKITCH_UTILS_DIR -I . -I $SHARED_MEM_LIB_DIR -L $RAYLIB_LINK_DIR -lraylib \
-Wall -std=c++17 -O2 -ldl -pthread -lrt

g++ -o play_race_track play_race_track.cpp \
-isystem $RAYLIB_CPP_INCLUDE_DIR -I $RAYLIB_INCLUDE_DIR -I $OKITCH_UTILS_DIR -I . -I $SHARED_MEM_LIB_DIR -L $RAYLIB_LINK_DIR -lraylib \
-Wall -std=c++17 -O2 -ldl -pthread -lrt