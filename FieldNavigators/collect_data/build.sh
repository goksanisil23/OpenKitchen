#!/bin/bash

export PATH=/home/goksan/Downloads/cuda_13.0/bin:$PATH
export LD_LIBRARY_PATH=/home/goksan/Downloads/cuda_13.0/lib64:$LD_LIBRARY_PATH


mkdir -p build
cmake -DCMAKE_PREFIX_PATH=/home/goksan/Downloads/libtorch-shared-with-deps-2.10.0+cu130/libtorch -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build