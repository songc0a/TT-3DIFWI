#!/bin/sh

set -e

DW_OMP_NAME=libgomp.so.1
CFLAGS="-Wall -Wextra -pedantic -fPIC -fopenmp -Ofast -mavx2"
CUDAFLAGS="--restrict --use_fast_math -O3 -gencode=arch=compute_52,code=sm_52, -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=compute_80  --compiler-options -fPIC"
# gcc $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=float -DDW_NDIM=3 -c scalar.c -o scalar_cpu_iso_3d_2_float.o
# gcc $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=float -DDW_NDIM=3 -c scalar.c -o scalar_cpu_iso_3d_4_float.o
# gcc $CFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=float -DDW_NDIM=3 -c scalar.c -o scalar_cpu_iso_3d_6_float.o
# gcc $CFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=float -DDW_NDIM=3 -c scalar.c -o scalar_cpu_iso_3d_8_float.o
# gcc $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=double -DDW_NDIM=3 -c scalar.c -o scalar_cpu_iso_3d_2_double.o
# gcc $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=double -DDW_NDIM=3 -c scalar.c -o scalar_cpu_iso_3d_4_double.o
# gcc $CFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=double -DDW_NDIM=3 -c scalar.c -o scalar_cpu_iso_3d_6_double.o
# gcc $CFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=double -DDW_NDIM=3 -c scalar.c -o scalar_cpu_iso_3d_8_double.o
# gcc $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=float -DDW_NDIM=3 -c scalar_born.c -o scalar_born_cpu_iso_3d_2_float.o
# gcc $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=float -DDW_NDIM=3 -c scalar_born.c -o scalar_born_cpu_iso_3d_4_float.o
# gcc $CFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=float -DDW_NDIM=3 -c scalar_born.c -o scalar_born_cpu_iso_3d_6_float.o
# gcc $CFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=float -DDW_NDIM=3 -c scalar_born.c -o scalar_born_cpu_iso_3d_8_float.o
# gcc $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=double -DDW_NDIM=3 -c scalar_born.c -o scalar_born_cpu_iso_3d_2_double.o
# gcc $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=double -DDW_NDIM=3 -c scalar_born.c -o scalar_born_cpu_iso_3d_4_double.o
# gcc $CFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=double -DDW_NDIM=3 -c scalar_born.c -o scalar_born_cpu_iso_3d_6_double.o
# gcc $CFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=double -DDW_NDIM=3 -c scalar_born.c -o scalar_born_cpu_iso_3d_8_double.o
# gcc $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=float -DDW_NDIM=3 -c elastic.c -o elastic_cpu_iso_3d_2_float.o
# gcc $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=float -DDW_NDIM=3 -c elastic.c -o elastic_cpu_iso_3d_4_float.o
# gcc $CFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=double -DDW_NDIM=3 -c elastic.c -o elastic_cpu_iso_3d_2_double.o
# gcc $CFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=double -DDW_NDIM=3 -c elastic.c -o elastic_cpu_iso_3d_4_double.o
# gcc $CFLAGS -shared scalar_cpu_iso_3d_2_float.o scalar_cpu_iso_3d_4_float.o scalar_cpu_iso_3d_6_float.o scalar_cpu_iso_3d_8_float.o scalar_cpu_iso_3d_2_double.o scalar_cpu_iso_3d_4_double.o scalar_cpu_iso_3d_6_double.o scalar_cpu_iso_3d_8_double.o scalar_born_cpu_iso_3d_2_float.o scalar_born_cpu_iso_3d_4_float.o scalar_born_cpu_iso_3d_6_float.o scalar_born_cpu_iso_3d_8_float.o scalar_born_cpu_iso_3d_2_double.o scalar_born_cpu_iso_3d_4_double.o scalar_born_cpu_iso_3d_6_double.o scalar_born_cpu_iso_3d_8_double.o elastic_cpu_iso_3d_2_float.o elastic_cpu_iso_3d_4_float.o elastic_cpu_iso_3d_2_double.o elastic_cpu_iso_3d_4_double.o -L. -l:$DW_OMP_NAME -Wl,-rpath='$ORIGIN' -o libdeepwave_cpu_linux_x86_64.so
if [ "${1}" != 'NOCUDA' ]
    then
    nvcc $CUDAFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=float -DDW_NDIM=3 -c scalar.cu -o scalar_cuda_iso_3d_2_float.o
    nvcc $CUDAFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=float -DDW_NDIM=3 -c scalar.cu -o scalar_cuda_iso_3d_4_float.o
    nvcc $CUDAFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=float -DDW_NDIM=3 -c scalar.cu -o scalar_cuda_iso_3d_6_float.o
    nvcc $CUDAFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=float -DDW_NDIM=3 -c scalar.cu -o scalar_cuda_iso_3d_8_float.o
    nvcc $CUDAFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=double -DDW_NDIM=3 -c scalar.cu -o scalar_cuda_iso_3d_2_double.o
    nvcc $CUDAFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=double -DDW_NDIM=3 -c scalar.cu -o scalar_cuda_iso_3d_4_double.o
    nvcc $CUDAFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=double -DDW_NDIM=3 -c scalar.cu -o scalar_cuda_iso_3d_6_double.o
    nvcc $CUDAFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=double -DDW_NDIM=3 -c scalar.cu -o scalar_cuda_iso_3d_8_double.o
    # nvcc $CUDAFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=float -DDW_NDIM=3 -c scalar_born.cu -o scalar_born_cuda_iso_3d_2_float.o
    # nvcc $CUDAFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=float -DDW_NDIM=3 -c scalar_born.cu -o scalar_born_cuda_iso_3d_4_float.o
    # nvcc $CUDAFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=float -DDW_NDIM=3 -c scalar_born.cu -o scalar_born_cuda_iso_3d_6_float.o
    # nvcc $CUDAFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=float -DDW_NDIM=3 -c scalar_born.cu -o scalar_born_cuda_iso_3d_8_float.o
    # nvcc $CUDAFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=double -DDW_NDIM=3 -c scalar_born.cu -o scalar_born_cuda_iso_3d_2_double.o
    # nvcc $CUDAFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=double -DDW_NDIM=3 -c scalar_born.cu -o scalar_born_cuda_iso_3d_4_double.o
    # nvcc $CUDAFLAGS -DDW_ACCURACY=6 -DDW_DTYPE=double -DDW_NDIM=3 -c scalar_born.cu -o scalar_born_cuda_iso_3d_6_double.o
    # nvcc $CUDAFLAGS -DDW_ACCURACY=8 -DDW_DTYPE=double -DDW_NDIM=3 -c scalar_born.cu -o scalar_born_cuda_iso_3d_8_double.o
    # nvcc $CUDAFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=float -DDW_NDIM=3 -c elastic.cu -o elastic_cuda_iso_3d_2_float.o
    # nvcc $CUDAFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=float -DDW_NDIM=3 -c elastic.cu -o elastic_cuda_iso_3d_4_float.o
    # nvcc $CUDAFLAGS -DDW_ACCURACY=2 -DDW_DTYPE=double -DDW_NDIM=3 -c elastic.cu -o elastic_cuda_iso_3d_2_double.o
    # nvcc $CUDAFLAGS -DDW_ACCURACY=4 -DDW_DTYPE=double -DDW_NDIM=3 -c elastic.cu -o elastic_cuda_iso_3d_4_double.o
    # nvcc $CUDAFLAGS -shared scalar_cuda_iso_3d_2_float.o scalar_cuda_iso_3d_4_float.o scalar_cuda_iso_3d_6_float.o scalar_cuda_iso_3d_8_float.o scalar_cuda_iso_3d_2_double.o scalar_cuda_iso_3d_4_double.o scalar_cuda_iso_3d_6_double.o scalar_cuda_iso_3d_8_double.o scalar_born_cuda_iso_3d_2_float.o scalar_born_cuda_iso_3d_4_float.o scalar_born_cuda_iso_3d_6_float.o scalar_born_cuda_iso_3d_8_float.o scalar_born_cuda_iso_3d_2_double.o scalar_born_cuda_iso_3d_4_double.o scalar_born_cuda_iso_3d_6_double.o scalar_born_cuda_iso_3d_8_double.o elastic_cuda_iso_3d_2_float.o elastic_cuda_iso_3d_4_float.o elastic_cuda_iso_3d_2_double.o elastic_cuda_iso_3d_4_double.o -o libdeepwave_cuda_linux_x86_64.so
    nvcc $CUDAFLAGS -shared scalar_cuda_iso_3d_2_float.o scalar_cuda_iso_3d_4_float.o scalar_cuda_iso_3d_6_float.o scalar_cuda_iso_3d_8_float.o scalar_cuda_iso_3d_2_double.o scalar_cuda_iso_3d_4_double.o scalar_cuda_iso_3d_6_double.o scalar_cuda_iso_3d_8_double.o -o libdeepwave_cuda_linux_x86_64.so
fi
rm *.o
