# Installation
We need to install CUDA and libtorch locally. I prefer not having these installed as system libraries so we will use local paths.

1) Download cuda and install locally. If you have NVidia drivers already installed, uncheck that option in interactive shell.
```sh
mkdir ~/Downloads/cuda & cd  ~/Downloads/cuda
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
./cuda_12.0.0_525.60.13_linux.run --toolkit --toolkitpath=/home/s0001734/Downloads/cuda
```

2) Get libtorch c++ library, and extract
```sh
https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
```

3) When compiling the sample app, point CMAKE to libtorch installation, as well as extending the environment
variables for local CUDA installation to be found
```sh
export PATH=/home/s0001734/Downloads/cuda/bin:$PATH
export LD_LIBRARY_PATH=/home/s0001734/Downloads/cuda/lib64:$LD_LIBRARY_PATH
cmake -DCMAKE_PREFIX_PATH=/home/s0001734/Downloads/libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121/libtorch ..
```