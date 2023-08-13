#!/bin/bash

# This script assumes that ONNX and OPENCV are installed as system libraries

export SHARED_MEM_LIB_DIR="/home/SharedMemory"
export OKITCH_UTILS_DIR="/home/Utilities"

g++ -o infer_onnx_gpu_from_file infer_onnx_gpu_from_file.cpp \
-lonnxruntime \
-I /usr/include/opencv4 -L /usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs \
-Wall -std=c++17 -O2 -ldl -pthread

g++ -o infer_onnx_gpu_from_raylib infer_onnx_gpu_from_raylib.cpp \
-lonnxruntime \
-I $SHARED_MEM_LIB_DIR -I $OKITCH_UTILS_DIR \
-I /usr/include/opencv4 -L /usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs \
-Wall -std=c++17 -O2 -ldl -pthread -lrt

g++ -o image_reader image_reader.cpp \
-I $SHARED_MEM_LIB_DIR -I $OKITCH_UTILS_DIR \
-lonnxruntime \
-I /usr/include/opencv4 -L /usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs \
-Wall -std=c++17 -O2 -ldl -pthread -lrt