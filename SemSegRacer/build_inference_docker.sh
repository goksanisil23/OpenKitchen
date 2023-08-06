#!/bin/bash

# This script assumes that ONNX and OPENCV are installed as system libraries

g++ -o infer_onnx_gpu infer_onnx_gpu.cpp \
-lonnxruntime \
-I /usr/include/opencv4 -L /usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs \
-Wall -std=c++17 -O2 -ldl -pthread