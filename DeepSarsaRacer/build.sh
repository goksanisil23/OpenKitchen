#!/bin/bash

export PATH=/home/s0001734/Downloads/cuda/bin:$PATH
export LD_LIBRARY_PATH=/home/s0001734/Downloads/cuda/lib64:$LD_LIBRARY_PATH
mkdir -p build
cmake -DCMAKE_PREFIX_PATH=/home/s0001734/Downloads/libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121/libtorch -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build