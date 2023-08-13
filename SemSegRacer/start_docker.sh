#!/bin/bash

xhost +

export HOST_MNT_DIR="/home/s0001734/Downloads/OpenKitchen"
export DOCKER_MNT_DIR="/home"

docker run -it --gpus all -v $HOST_MNT_DIR:$DOCKER_MNT_DIR \
-v /tmp/.X11-unix:/tmp/.X11-unix --ipc="host" --env="DISPLAY" \
--shm-size=1g --name semseg_onnx_cuda_container onnxruntime-cuda-okitch-semseg-user