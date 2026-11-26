#!/bin/sh

# Build only the 3D scalar propagator. Run from a Git Bash shell opened from
# an x64 Visual Studio Developer Command Prompt. Pass NOCUDA for a CPU-only
# build; otherwise nvcc must be available.
set -eu

# Prevent Git Bash/MSYS from rewriting MSVC options such as /O2 as paths.
export MSYS2_ARG_CONV_EXCL='*'

CPU_OBJECTS=""
for accuracy in 2 4 6 8; do
    for dtype in float double; do
        object="scalar_cpu_iso_3d_${accuracy}_${dtype}.obj"
        cl /nologo /O2 /fp:fast /arch:AVX2 /openmp /std:c11 /c \
            /DDW_ACCURACY=$accuracy /DDW_DTYPE=$dtype \
            /Fo"$object" scalar.c
        CPU_OBJECTS="$CPU_OBJECTS $object"
    done
done

cl /nologo /LD $CPU_OBJECTS /Fe:libdeepwave_cpu_windows_x86_64.dll
rm -f $CPU_OBJECTS
rm -f libdeepwave_cpu_windows_x86_64.exp libdeepwave_cpu_windows_x86_64.lib
echo "Built libdeepwave_cpu_windows_x86_64.dll"

if [ "${1:-}" != "NOCUDA" ]; then
    CUDAFLAGS="--restrict --use_fast_math -O3 \
        -gencode=arch=compute_60,code=sm_60 \
        -gencode=arch=compute_70,code=sm_70 \
        -gencode=arch=compute_75,code=sm_75 \
        -gencode=arch=compute_80,code=sm_80 \
        -gencode=arch=compute_86,code=sm_86 \
        -gencode=arch=compute_86,code=compute_86"
    CUDA_OBJECTS=""
    for accuracy in 2 4 6 8; do
        for dtype in float double; do
            object="scalar_cuda_iso_3d_${accuracy}_${dtype}.obj"
            nvcc $CUDAFLAGS -DDW_ACCURACY=$accuracy -DDW_DTYPE=$dtype \
                -c scalar.cu -o "$object"
            CUDA_OBJECTS="$CUDA_OBJECTS $object"
        done
    done
    nvcc -shared $CUDA_OBJECTS -o libdeepwave_cuda_windows_x86_64.dll
    rm -f $CUDA_OBJECTS
    echo "Built libdeepwave_cuda_windows_x86_64.dll"
fi
