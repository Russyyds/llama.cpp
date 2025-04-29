DEVICE_TYPE=$1
BUILD_TYPE=$2
echo build type :${BUILD_TYPE}
BUILD_DIR=build
if [ $1 = "CPU" ]; then
    if [ -d $BUILD_DIR ]; then
        echo "Directory $BUILD_DIR already exists, remove it."
        # rm -rf $BUILD_DIR
    fi
    cmake -B $BUILD_DIR -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
elif [ $1 = "CUDA" ]; then
    BUILD_DIR=build_cuda
    if [ -d $BUILD_DIR ]; then
        echo "Directory $BUILD_DIR already exists, remove it."
        rm -rf $BUILD_DIR
    fi
    cmake -B $BUILD_DIR -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
else
    echo "Default build:CPU"
    if [ -d $BUILD_DIR ]; then
        echo "Directory $BUILD_DIR already exists, remove it."
        rm -rf $BUILD_DIR
    fi
    cmake -B $BUILD_DIR -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
fi
# cmake --build $BUILD_DIR --config Release -j 16
cmake --build $BUILD_DIR -j 16