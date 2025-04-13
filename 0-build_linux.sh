BUILD_TYPE=$1

BUILD_DIR=build
if [ $1 = "CPU" ]; then
    if [ -d $BUILD_DIR ]; then
        echo "Directory $BUILD_DIR already exists, remove it."
        rm -rf $BUILD_DIR
    fi
    cmake -B $BUILD_DIR -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
elif [ $1 = "CUDA" ]; then
    BUILD_DIR=build_cuda
    if [ -d $BUILD_DIR ]; then
        echo "Directory $BUILD_DIR already exists, remove it."
        rm -rf $BUILD_DIR
    fi
    cmake -B $BUILD_DIR -DGGML_CUDA=ON -DLLAMA_CURL=OFF
else
    echo "Default build:CPU"
    if [ -d $BUILD_DIR ]; then
        echo "Directory $BUILD_DIR already exists, remove it."
        rm -rf $BUILD_DIR
    fi
    cmake -B $BUILD_DIR -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
fi
# cmake --build $BUILD_DIR --config Release -j 16
cmake --build $BUILD_DIR -j 16