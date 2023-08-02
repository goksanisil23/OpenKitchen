#!/bin/bash

export RAYLIB_CPP_INCLUDE_DIR="/home/s0001734/Downloads/raylib-cpp/include"
export RAYLIB_INCLUDE_DIR="/home/s0001734/Downloads/raylib-cpp/build/_deps/raylib-build/raylib/include"
export RAYLIB_LINK_DIR="/home/s0001734/Downloads/raylib-cpp/build/_deps/raylib-build/raylib"

export OKITCH_UTILS_DIR="/home/s0001734/Downloads/OpenKitchen/Utilities"

export ONNX_INCLUDE_DIR="/home/s0001734/Downloads/onnxruntime-linux-x64-gpu-1.15.1/include"
export ONNX_LIB_DIR="/home/s0001734/Downloads/onnxruntime-linux-x64-gpu-1.15.1/lib"



g++ -o race_track_gen race_track_gen.cpp \
-isystem $RAYLIB_CPP_INCLUDE_DIR -I $RAYLIB_INCLUDE_DIR -I $OKITCH_UTILS_DIR -I . -L $RAYLIB_LINK_DIR -lraylib \
-Wall -std=c++17 -O2 -ldl -pthread

g++ -o infer_onnx infer_onnx.cpp \
-isystem $RAYLIB_CPP_INCLUDE_DIR -I $RAYLIB_INCLUDE_DIR -I $OKITCH_UTILS_DIR -I . -L $RAYLIB_LINK_DIR -lraylib \
-I $ONNX_INCLUDE_DIR -L $ONNX_LIB_DIR -lonnxruntime -Wl,-rpath,$ONNX_LIB_DIR \
-I /usr/include/opencv4 -L /usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs \
-Wall -std=c++17 -O2 -ldl -pthread