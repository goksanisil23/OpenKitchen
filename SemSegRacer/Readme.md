# Semantic Segmentation
This is a simple semantic segmentation example where we train a deeplab v3 Resnet 50 model using Pytorch. Images are left and right boundries and drivable areas of  [various race tracks around the world](https://github.com/TUMFTM/racetrack-database) that we render in different colors using raylib, which also stands for semantic classes.

The trained model is converted to onnx format, and the inference runs on a GPU using the Onnx CUDA C++ runtime, inside the docker container.
### Training
Raylib is used to shade track boundaries and the area between them using triangle shading. Each track is successively traversed along the center line and a birds-eye view from a certain distance is saved as training image.
```sh
./build_sim.sh
./race_track_training_gen PATH_TO_RACE_TRACKS_DIR
```
<img src="https://raw.githubusercontent.com/goksanisil23/OpenKitchen/main/SemSegRacer/resources/example_training_image.png" width=30% height=30%>

Since each lane is rendered a different color, we can generate annotation mask on the fly according to rgb value of each pixel. We also modify the final layer of the original deeplabv3 network so that it only produces 4 class probabilities at the end (left/right/drivable area/ none)
```sh
python train_semseg.py
```
When exporting the model to ONNX format, we load the .torch weights resulting from the training and make sure that we apply the same final network layer modification before the call to export.
```sh
python export_semseg.py
``` 

### Onnx GPU runtime
Build the docker image that contains ONNX runtime with CUDA dependencies and OpenCV with GPU support.
Note that we add user and group id as identical to the host in order to allow shared memory with the host.
```sh
docker build -t onnxruntime-cuda-okitch-semseg -f Dockerfile.semseg .
docker build -t onnxruntime-cuda-okitch-semseg-user \
--build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f Dockerfile.semseg.user .
``` 

### Start docker container
Since we're sharing images from the simulator running on host to inference app running inside docker, we need to setup the docker container so that it can access /dev/shm with that runs the simulator.
```sh
./start_docker.sh
```

### Build & run inference inside container
```sh
cd /home
./build_inference_in_docker.sh
./infer_onnx_gpu_from_file
# OR
# Start the simulator on host via:
# ./play_race_track PATH_TO_RACE_TRACKS_DIR/Monza.csv
./infer_onnx_gpu_from_raylib
```

<img src="https://raw.githubusercontent.com/goksanisil23/OpenKitchen/main/SemSegRacer/resources/inference_gpu.gif" width=50% height=50%>