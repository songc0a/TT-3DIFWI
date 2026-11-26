#!/bin/sh

# Build only the 3D scalar propagator. Pass NOCUDA for a CPU-only build.
# Run this script from the deepwave3D directory.
set -eu

CFLAGS="-Wall -Wextra -pedantic -std=c99 -fPIC -fopenmp -Ofast -mavx2"
BUILD_DIR=$(mktemp -d)
trap 'rm -rf "$BUILD_DIR"' EXIT
CPU_OBJECTS=""
for accuracy in 2 4 6 8; do
    for dtype in float double; do
        object="$BUILD_DIR/scalar_cpu_iso_3d_${accuracy}_${dtype}.o"
        gcc $CFLAGS -DDW_ACCURACY=$accuracy -DDW_DTYPE=$dtype \
            -c scalar.c -o "$object"
        CPU_OBJECTS="$CPU_OBJECTS $object"
    done
done

gcc -shared $CPU_OBJECTS -fopenmp -Wl,-rpath,'$ORIGIN' \
    -o "$BUILD_DIR/libdeepwave_cpu_linux_x86_64.so"
cp "$BUILD_DIR/libdeepwave_cpu_linux_x86_64.so" \
    libdeepwave_cpu_linux_x86_64.so
echo "Built libdeepwave_cpu_linux_x86_64.so"

if [ "${1:-}" != "NOCUDA" ]; then
    CUDAFLAGS="--restrict --use_fast_math -O3 \
        -gencode=arch=compute_60,code=sm_60 \
        -gencode=arch=compute_70,code=sm_70 \
        -gencode=arch=compute_75,code=sm_75 \
        -gencode=arch=compute_80,code=sm_80 \
        -gencode=arch=compute_86,code=sm_86 \
        -gencode=arch=compute_86,code=compute_86 \
        --compiler-options -fPIC"
    CUDA_OBJECTS=""
    for accuracy in 2 4 6 8; do
        for dtype in float double; do
            object="$BUILD_DIR/scalar_cuda_iso_3d_${accuracy}_${dtype}.o"
            nvcc $CUDAFLAGS -DDW_ACCURACY=$accuracy -DDW_DTYPE=$dtype \
                -c scalar.cu -o "$object"
            CUDA_OBJECTS="$CUDA_OBJECTS $object"
        done
    done
    nvcc -shared $CUDA_OBJECTS \
        -o "$BUILD_DIR/libdeepwave_cuda_linux_x86_64.so"
    cp "$BUILD_DIR/libdeepwave_cuda_linux_x86_64.so" \
        libdeepwave_cuda_linux_x86_64.so
    echo "Built libdeepwave_cuda_linux_x86_64.so"
fi
