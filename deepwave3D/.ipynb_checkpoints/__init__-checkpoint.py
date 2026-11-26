"""Wave propagation modules for PyTorch.

Provides forward modelling and backpropagation of the scalar wave
equation (regular and Born), and elastic wave equation.
It can thus be used to generate synthetic data, and to perform
FWI and LSRTM.

Module and functional interfaces are provided, and allow the propagators
to be easily integrated into a larger computational graph, if desired.

For better computational performance, the code is written in C++ and
CUDA.
"""

import platform
import ctypes
from ctypes import c_void_p, c_int64, c_float, c_double, c_bool
import pathlib
from deepwave3D.scalar import Scalar, scalar
from deepwave3D.scalar_born import ScalarBorn, scalar_born
from deepwave3D.elastic import Elastic, elastic
import deepwave3D.wavelets
import deepwave3D.location_interpolation
if platform.system() == 'Linux':
    dll_cpu = ctypes.CDLL(
        str(
            pathlib.Path(__file__).resolve().parent /
            "libdeepwave_cpu_linux_x86_64.so"))
    dll_cuda = ctypes.CDLL(
        str(
            pathlib.Path(__file__).resolve().parent /
            "libdeepwave_cuda_linux_x86_64.so"))
elif platform.system() == 'Darwin':
    if platform.machine() == 'arm64':
        dll_cpu = ctypes.CDLL(
            str(
                pathlib.Path(__file__).resolve().parent /
                "libdeepwave_cpu_macos_arm64.dylib"))
    else:
        dll_cpu = ctypes.CDLL(
            str(
                pathlib.Path(__file__).resolve().parent /
                "libdeepwave_cpu_macos_x86_64.dylib"))
elif platform.system() == 'Windows':
    dll_cpu = ctypes.CDLL(
        str(
            pathlib.Path(__file__).resolve().parent /
            "libdeepwave_cpu_windows_x86_64.dll"))
    dll_cuda = ctypes.CDLL(
        str(
            pathlib.Path(__file__).resolve().parent /
            "libdeepwave_cuda_windows_x86_64.dll"))
else:
    raise RuntimeError("Unsupported OS or platform type")
# Check if was compiled with OpenMP support
try:
    dll_cpu.omp_get_num_threads
    use_openmp = True
except AttributeError:
    use_openmp = False
# dll_cpu.scalar_iso_3d_2_float_forward.restype = None
# dll_cpu.scalar_iso_3d_4_float_forward.restype = None
# dll_cpu.scalar_iso_3d_6_float_forward.restype = None
# dll_cpu.scalar_iso_3d_8_float_forward.restype = None
# dll_cpu.scalar_iso_3d_2_double_forward.restype = None
# dll_cpu.scalar_iso_3d_4_double_forward.restype = None
# dll_cpu.scalar_iso_3d_6_double_forward.restype = None
# dll_cpu.scalar_iso_3d_8_double_forward.restype = None
# dll_cpu.scalar_iso_3d_2_float_backward.restype = None
# dll_cpu.scalar_iso_3d_4_float_backward.restype = None
# dll_cpu.scalar_iso_3d_6_float_backward.restype = None
# dll_cpu.scalar_iso_3d_8_float_backward.restype = None
# dll_cpu.scalar_iso_3d_2_double_backward.restype = None
# dll_cpu.scalar_iso_3d_4_double_backward.restype = None
# dll_cpu.scalar_iso_3d_6_double_backward.restype = None
# dll_cpu.scalar_iso_3d_8_double_backward.restype = None

if platform.system() != 'Darwin':
    dll_cuda.scalar_iso_3d_2_float_forward.restype = None
    dll_cuda.scalar_iso_3d_4_float_forward.restype = None
    dll_cuda.scalar_iso_3d_6_float_forward.restype = None
    dll_cuda.scalar_iso_3d_8_float_forward.restype = None
    dll_cuda.scalar_iso_3d_2_double_forward.restype = None
    dll_cuda.scalar_iso_3d_4_double_forward.restype = None
    dll_cuda.scalar_iso_3d_6_double_forward.restype = None
    dll_cuda.scalar_iso_3d_8_double_forward.restype = None
    dll_cuda.scalar_iso_3d_2_float_backward.restype = None
    dll_cuda.scalar_iso_3d_4_float_backward.restype = None
    dll_cuda.scalar_iso_3d_6_float_backward.restype = None
    dll_cuda.scalar_iso_3d_8_float_backward.restype = None
    dll_cuda.scalar_iso_3d_2_double_backward.restype = None
    dll_cuda.scalar_iso_3d_4_double_backward.restype = None
    dll_cuda.scalar_iso_3d_6_double_backward.restype = None
    dll_cuda.scalar_iso_3d_8_double_backward.restype = None

scalar_iso_float_forward_argtypes = [
    c_void_p,  # v
    c_void_p,  # f
    c_void_p,  # wfc
    c_void_p,  # wfp
    c_void_p,  # psiy
    c_void_p,  # psix
    c_void_p,  # psiyn
    c_void_p,  # psixn
    c_void_p,  # zetay
    c_void_p,  # zetax
    c_void_p,  # dwdv
    c_void_p,  # r
    c_void_p,  # ay
    c_void_p,  # ax
    c_void_p,  # by
    c_void_p,  # bx
    c_void_p,  # dbydy
    c_void_p,  # dbxdx
    c_void_p,  # bz
    c_void_p,  # az
    c_void_p,  # dbzdz
    c_void_p,  # psiz
    c_void_p,  # psizn
    c_void_p,  # zetaz
    c_void_p,  # sources_i
    c_void_p,  # receivers_i
    c_float,   # rdz_h
    c_float,   # rdy_h
    c_float,   # rdx_h
    c_float,   # rdz2_h
    c_float,   # rdy2_h
    c_float,   # rdx2_h
    c_float,   # dt2_h
    c_int64,   # nt
    c_int64,   # n_shots
    c_int64,   # nz
    c_int64,   # ny
    c_int64,   # nx
    c_int64,   # n_sources_per_shot
    c_int64,   # n_receivers_per_shot
    c_int64,   # step_ratio
    c_bool,    # v_requires_grad
    c_int64,   # pml_z0
    c_int64,   # pml_z1
    c_int64,   # pml_y0
    c_int64,   # pml_y1
    c_int64,   # pml_x0
    c_int64,   # pml_x1
    c_int64    # device
]

scalar_iso_double_forward_argtypes = [
    c_void_p,  # v
    c_void_p,  # f
    c_void_p,  # wfc
    c_void_p,  # wfp
    c_void_p,  # psiy
    c_void_p,  # psix
    c_void_p,  # psiyn
    c_void_p,  # psixn
    c_void_p,  # zetay
    c_void_p,  # zetax
    c_void_p,  # dwdv
    c_void_p,  # r
    c_void_p,  # ay
    c_void_p,  # ax
    c_void_p,  # by
    c_void_p,  # bx
    c_void_p,  # dbydy
    c_void_p,  # dbxdx
    c_void_p,  # bz
    c_void_p,  # az
    c_void_p,  # dbzdz
    c_void_p,  # psiz
    c_void_p,  # psizn
    c_void_p,  # zetaz
    c_void_p,  # sources_i
    c_void_p,  # receivers_i
    c_double,   # rdz_h
    c_double,   # rdy_h
    c_double,   # rdx_h
    c_double,   # rdz2_h
    c_double,   # rdy2_h
    c_double,   # rdx2_h
    c_double,   # dt2_h
    c_int64,   # nt
    c_int64,   # n_shots
    c_int64,   # nz
    c_int64,   # ny
    c_int64,   # nx
    c_int64,   # n_sources_per_shot
    c_int64,   # n_receivers_per_shot
    c_int64,   # step_ratio
    c_bool,    # v_requires_grad
    c_int64,   # pml_z0
    c_int64,   # pml_z1
    c_int64,   # pml_y0
    c_int64,   # pml_y1
    c_int64,   # pml_x0
    c_int64,   # pml_x1
    c_int64    # device
]

scalar_iso_float_backward_argtypes = [
    c_void_p,  # v2dt2
    c_void_p,  # grad_r
    c_void_p,  # wfc
    c_void_p,  # wfp
    c_void_p,  # psiy
    c_void_p,  # psix
    c_void_p,  # psiyn
    c_void_p,  # psixn
    c_void_p,  # zetay
    c_void_p,  # zetax
    c_void_p,  # zetayn
    c_void_p,  # zetaxn
    c_void_p,  # dwdv
    c_void_p,  # grad_f
    c_void_p,  # grad_v
    c_void_p,  # grad_v_shot
    c_void_p,  # ay
    c_void_p,  # ax
    c_void_p,  # by
    c_void_p,  # bx
    c_void_p,  # dbydy
    c_void_p,  # dbxdx
    c_void_p,  # bz
    c_void_p,  # az
    c_void_p,  # dbzdz
    c_void_p,  # psiz
    c_void_p,  # psizn
    c_void_p,  # zetaz
    c_void_p,  # zetazn
    c_void_p,  # sources_i
    c_void_p,  # receivers_i
    c_float,   # rdz_h
    c_float,   # rdy_h
    c_float,   # rdx_h
    c_float,   # rdz2_h
    c_float,   # rdy2_h
    c_float,   # rdx2_h
    c_int64,   # nt
    c_int64,   # n_shots
    c_int64,   # nz
    c_int64,   # ny
    c_int64,   # nx
    c_int64,   # n_sources_per_shot
    c_int64,   # n_receivers_per_shot
    c_int64,   # step_ratio
    c_bool,    # v_requires_grad
    c_int64,   # pml_z0
    c_int64,   # pml_z1
    c_int64,   # pml_y0
    c_int64,   # pml_y1
    c_int64,   # pml_x0
    c_int64,   # pml_x1
    c_int64    # device
]

scalar_iso_double_backward_argtypes = [
    c_void_p,  # v2dt2
    c_void_p,  # grad_r
    c_void_p,  # wfc
    c_void_p,  # wfp
    c_void_p,  # psiy
    c_void_p,  # psix
    c_void_p,  # psiyn
    c_void_p,  # psixn
    c_void_p,  # zetay
    c_void_p,  # zetax
    c_void_p,  # zetayn
    c_void_p,  # zetaxn
    c_void_p,  # dwdv
    c_void_p,  # grad_f
    c_void_p,  # grad_v
    c_void_p,  # grad_v_shot
    c_void_p,  # ay
    c_void_p,  # ax
    c_void_p,  # by
    c_void_p,  # bx
    c_void_p,  # dbydy
    c_void_p,  # dbxdx
    c_void_p,  # bz
    c_void_p,  # az
    c_void_p,  # dbzdz
    c_void_p,  # psiz
    c_void_p,  # psizn
    c_void_p,  # zetaz
    c_void_p,  # zetazn
    c_void_p,  # sources_i
    c_void_p,  # receivers_i
    c_double,   # rdz_h
    c_double,   # rdy_h
    c_double,   # rdx_h
    c_double,   # rdz2_h
    c_double,   # rdy2_h
    c_double,   # rdx2_h
    c_int64,   # nt
    c_int64,   # n_shots
    c_int64,   # nz
    c_int64,   # ny
    c_int64,   # nx
    c_int64,   # n_sources_per_shot
    c_int64,   # n_receivers_per_shot
    c_int64,   # step_ratio
    c_bool,    # v_requires_grad
    c_int64,   # pml_z0
    c_int64,   # pml_z1
    c_int64,   # pml_y0
    c_int64,   # pml_y1
    c_int64,   # pml_x0
    c_int64,   # pml_x1
    c_int64    # device
]



# dll_cpu.scalar_iso_3d_2_float_forward.argtypes = scalar_iso_float_forward_argtypes
# dll_cpu.scalar_iso_3d_4_float_forward.argtypes = scalar_iso_float_forward_argtypes
# dll_cpu.scalar_iso_3d_6_float_forward.argtypes = scalar_iso_float_forward_argtypes
# dll_cpu.scalar_iso_3d_8_float_forward.argtypes = scalar_iso_float_forward_argtypes
# dll_cpu.scalar_iso_3d_2_double_forward.argtypes = scalar_iso_double_forward_argtypes
# dll_cpu.scalar_iso_3d_4_double_forward.argtypes = scalar_iso_double_forward_argtypes
# dll_cpu.scalar_iso_3d_6_double_forward.argtypes = scalar_iso_double_forward_argtypes
# dll_cpu.scalar_iso_3d_8_double_forward.argtypes = scalar_iso_double_forward_argtypes
# dll_cpu.scalar_iso_3d_2_float_backward.argtypes = scalar_iso_float_backward_argtypes
# dll_cpu.scalar_iso_3d_4_float_backward.argtypes = scalar_iso_float_backward_argtypes
# dll_cpu.scalar_iso_3d_6_float_backward.argtypes = scalar_iso_float_backward_argtypes
# dll_cpu.scalar_iso_3d_8_float_backward.argtypes = scalar_iso_float_backward_argtypes
# dll_cpu.scalar_iso_3d_2_double_backward.argtypes = scalar_iso_double_backward_argtypes
# dll_cpu.scalar_iso_3d_4_double_backward.argtypes = scalar_iso_double_backward_argtypes
# dll_cpu.scalar_iso_3d_6_double_backward.argtypes = scalar_iso_double_backward_argtypes
# dll_cpu.scalar_iso_3d_8_double_backward.argtypes = scalar_iso_double_backward_argtypes

if platform.system() != 'Darwin':
    dll_cuda.scalar_iso_3d_2_float_forward.argtypes = scalar_iso_float_forward_argtypes
    dll_cuda.scalar_iso_3d_4_float_forward.argtypes = scalar_iso_float_forward_argtypes
    dll_cuda.scalar_iso_3d_6_float_forward.argtypes = scalar_iso_float_forward_argtypes
    dll_cuda.scalar_iso_3d_8_float_forward.argtypes = scalar_iso_float_forward_argtypes
    dll_cuda.scalar_iso_3d_2_double_forward.argtypes = scalar_iso_double_forward_argtypes
    dll_cuda.scalar_iso_3d_4_double_forward.argtypes = scalar_iso_double_forward_argtypes
    dll_cuda.scalar_iso_3d_6_double_forward.argtypes = scalar_iso_double_forward_argtypes
    dll_cuda.scalar_iso_3d_8_double_forward.argtypes = scalar_iso_double_forward_argtypes
    dll_cuda.scalar_iso_3d_2_float_backward.argtypes = scalar_iso_float_backward_argtypes
    dll_cuda.scalar_iso_3d_4_float_backward.argtypes = scalar_iso_float_backward_argtypes
    dll_cuda.scalar_iso_3d_6_float_backward.argtypes = scalar_iso_float_backward_argtypes
    dll_cuda.scalar_iso_3d_8_float_backward.argtypes = scalar_iso_float_backward_argtypes
    dll_cuda.scalar_iso_3d_2_double_backward.argtypes = scalar_iso_double_backward_argtypes
    dll_cuda.scalar_iso_3d_4_double_backward.argtypes = scalar_iso_double_backward_argtypes
    dll_cuda.scalar_iso_3d_6_double_backward.argtypes = scalar_iso_double_backward_argtypes
    dll_cuda.scalar_iso_3d_8_double_backward.argtypes = scalar_iso_double_backward_argtypes

