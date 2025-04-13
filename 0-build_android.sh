PROJECT_DIR=$(pwd)
THIRD_PARTY_DIR=${PROJECT_DIR}/third_party

if [ ! -d $THIRD_PARTY_DIR ]; then
  mkdir $THIRD_PARTY_DIR
fi

cd $THIRD_PARTY_DIR
if [ ! -d OpenCL-Headers ]; then
  echo "Directory OpenCL-Headers already exists, remove it."
  git clone https://github.com/KhronosGroup/OpenCL-Headers
fi
cd OpenCL-Headers && \
cp -r CL $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include

cd ..
if [ ! -d OpenCL-ICD-Loader ]; then
    git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader
fi

cd OpenCL-ICD-Loader
if [ -d build_ndk ]; then
  echo "Directory build_ndk already exists, remove it."
  rm -rf build_ndk
fi

mkdir build_ndk
cd build_ndk && \
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DOPENCL_ICD_LOADER_HEADERS_DIR=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=24 \
  -DANDROID_STL=c++_shared && \
ninja && \
cp libOpenCL.so $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android

cd ${PROJECT_DIR}
if [ -d build_android ]; then
  echo "Directory build already exists, remove it."
  rm -rf build_android
fi
mkdir build_android
cd build_android

cmake .. -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_OPENCL=ON \
  -DGGML_OPENMP=OFF \
  -DLLAMA_CURL=OFF

ninja -j 16