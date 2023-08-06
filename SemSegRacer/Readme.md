### Onnx GPU runtime
Build the docker image that contains ONNX runtime with CUDA dependencies and OpenCV with GPU support

```sh
docker build -t onnxruntime-cuda-okitch-semseg -f Dockerfile.semseg .
``` 

### Start docker container
```sh
xhost +
export HOST_MNT_DIR="/home/s0001734/Downloads/OpenKitchen/SemSegRacer"
export DOCKER_MNT_DIR="/home"
docker run -it --gpus all -v $HOST_MNT_DIR:$DOCKER_MNT_DIR -v /tmp/.X11-unix:/tmp/.X11-unix --ipc="host" --env="DISPLAY" --shm-size=1g --name semseg_onnx_cuda_container onnxruntime-cuda-okitch-semseg
```

### Build & run inference inside container
```sh
cd /home
./build_docker.sh
./infer_onnx_gpu
```