
#ifndef DW_COMMON_H
#define DW_COMMON_H

#if DW_ACCURACY == 2
// 2nd order accuracy
#define DIFFZ1(a) (((DW_DTYPE)(1.0 / 2.0) * (a(1, 0, 0) - a(-1, 0, 0))) * rdz)
#define DIFFZ2(a) ((-2 * a(0, 0, 0) + (a(1, 0, 0) + a(-1, 0, 0))) * rdz2)
#define DIFFY1(a) (((DW_DTYPE)(1.0 / 2.0) * (a(0, 1, 0) - a(0, -1, 0))) * rdy)
#define DIFFY2(a) ((-2 * a(0, 0, 0) + (a(0, 1, 0) + a(0, -1, 0))) * rdy2)
#define DIFFX1(a) (((DW_DTYPE)(1.0 / 2.0) * (a(0, 0, 1) - a(0, 0, -1))) * rdx)
#define DIFFX2(a) ((-2 * a(0, 0, 0) + (a(0, 0, 1) + a(0, 0, -1))) * rdx2)
#define FD_PAD 1

#elif DW_ACCURACY == 4
// 4th order accuracy
#define DIFFZ1(a) \
  (((DW_DTYPE)(8.0 / 12.0) * (a(1, 0, 0) - a(-1, 0, 0)) + \
    (DW_DTYPE)(-1.0 / 12.0) * (a(2, 0, 0) - a(-2, 0, 0))) * rdz)
#define DIFFZ2(a) \
  (((DW_DTYPE)(-5.0 / 2.0) * a(0, 0, 0) + \
    (DW_DTYPE)(4.0 / 3.0) * (a(1, 0, 0) + a(-1, 0, 0)) + \
    (DW_DTYPE)(-1.0 / 12.0) * (a(2, 0, 0) + a(-2, 0, 0))) * rdz2)
#define DIFFY1(a) \
  (((DW_DTYPE)(8.0 / 12.0) * (a(0, 1, 0) - a(0, -1, 0)) + \
    (DW_DTYPE)(-1.0 / 12.0) * (a(0, 2, 0) - a(0, -2, 0))) * rdy)
#define DIFFY2(a) \
  (((DW_DTYPE)(-5.0 / 2.0) * a(0, 0, 0) + \
    (DW_DTYPE)(4.0 / 3.0) * (a(0, 1, 0) + a(0, -1, 0)) + \
    (DW_DTYPE)(-1.0 / 12.0) * (a(0, 2, 0) + a(0, -2, 0))) * rdy2)
#define DIFFX1(a) \
  (((DW_DTYPE)(8.0 / 12.0) * (a(0, 0, 1) - a(0, 0, -1)) + \
    (DW_DTYPE)(-1.0 / 12.0) * (a(0, 0, 2) - a(0, 0, -2))) * rdx)
#define DIFFX2(a) \
  (((DW_DTYPE)(-5.0 / 2.0) * a(0, 0, 0) + \
    (DW_DTYPE)(4.0 / 3.0) * (a(0, 0, 1) + a(0, 0, -1)) + \
    (DW_DTYPE)(-1.0 / 12.0) * (a(0, 0, 2) + a(0, 0, -2))) * rdx2)
#define FD_PAD 2

#elif DW_ACCURACY == 6
// 6th order accuracy
#define DIFFZ1(a) \
  (((DW_DTYPE)(3.0 / 4.0) * (a(1, 0, 0) - a(-1, 0, 0)) + \
    (DW_DTYPE)(-3.0 / 20.0) * (a(2, 0, 0) - a(-2, 0, 0)) + \
    (DW_DTYPE)(1.0 / 60.0) * (a(3, 0, 0) - a(-3, 0, 0))) * rdz)
#define DIFFZ2(a) \
  (((DW_DTYPE)(-49.0 / 18.0) * a(0, 0, 0) + \
    (DW_DTYPE)(3.0 / 2.0) * (a(1, 0, 0) + a(-1, 0, 0)) + \
    (DW_DTYPE)(-3.0 / 20.0) * (a(2, 0, 0) + a(-2, 0, 0)) + \
    (DW_DTYPE)(1.0 / 90.0) * (a(3, 0, 0) + a(-3, 0, 0))) * rdz2)
#define DIFFY1(a) \
  (((DW_DTYPE)(3.0 / 4.0) * (a(0, 1, 0) - a(0, -1, 0)) + \
    (DW_DTYPE)(-3.0 / 20.0) * (a(0, 2, 0) - a(0, -2, 0)) + \
    (DW_DTYPE)(1.0 / 60.0) * (a(0, 3, 0) - a(0, -3, 0))) * rdy)
#define DIFFY2(a) \
  (((DW_DTYPE)(-49.0 / 18.0) * a(0, 0, 0) + \
    (DW_DTYPE)(3.0 / 2.0) * (a(0, 1, 0) + a(0, -1, 0)) + \
    (DW_DTYPE)(-3.0 / 20.0) * (a(0, 2, 0) + a(0, -2, 0)) + \
    (DW_DTYPE)(1.0 / 90.0) * (a(0, 3, 0) + a(0, -3, 0))) * rdy2)
#define DIFFX1(a) \
  (((DW_DTYPE)(3.0 / 4.0) * (a(0, 0, 1) - a(0, 0, -1)) + \
    (DW_DTYPE)(-3.0 / 20.0) * (a(0, 0, 2) - a(0, 0, -2)) + \
    (DW_DTYPE)(1.0 / 60.0) * (a(0, 0, 3) - a(0, 0, -3))) * rdx)
#define DIFFX2(a) \
  (((DW_DTYPE)(-49.0 / 18.0) * a(0, 0, 0) + \
    (DW_DTYPE)(3.0 / 2.0) * (a(0, 0, 1) + a(0, 0, -1)) + \
    (DW_DTYPE)(-3.0 / 20.0) * (a(0, 0, 2) + a(0, 0, -2)) + \
    (DW_DTYPE)(1.0 / 90.0) * (a(0, 0, 3) + a(0, 0, -3))) * rdx2)
#define FD_PAD 3

#elif DW_ACCURACY == 8
// 8th order accuracy
#define DIFFZ1(a) \
  (((DW_DTYPE)(4.0 / 5.0) * (a(1, 0, 0) - a(-1, 0, 0)) + \
    (DW_DTYPE)(-1.0 / 5.0) * (a(2, 0, 0) - a(-2, 0, 0)) + \
    (DW_DTYPE)(4.0 / 105.0) * (a(3, 0, 0) - a(-3, 0, 0)) + \
    (DW_DTYPE)(-1.0 / 280.0) * (a(4, 0, 0) - a(-4, 0, 0))) * rdz)
#define DIFFZ2(a) \
  (((DW_DTYPE)(-205.0 / 72.0) * a(0, 0, 0) + \
    (DW_DTYPE)(8.0 / 5.0) * (a(1, 0, 0) + a(-1, 0, 0)) + \
    (DW_DTYPE)(-1.0 / 5.0) * (a(2, 0, 0) + a(-2, 0, 0)) + \
    (DW_DTYPE)(8.0 / 315.0) * (a(3, 0, 0) + a(-3, 0, 0)) + \
    (DW_DTYPE)(-1.0 / 560.0) * (a(4, 0, 0) + a(-4, 0, 0))) * rdz2)
#define DIFFY1(a) \
  (((DW_DTYPE)(4.0 / 5.0) * (a(0, 1, 0) - a(0, -1, 0)) + \
    (DW_DTYPE)(-1.0 / 5.0) * (a(0, 2, 0) - a(0, -2, 0)) + \
    (DW_DTYPE)(4.0 / 105.0) * (a(0, 3, 0) - a(0, -3, 0)) + \
    (DW_DTYPE)(-1.0 / 280.0) * (a(0, 4, 0) - a(0, -4, 0))) * rdy)
#define DIFFY2(a) \
  (((DW_DTYPE)(-205.0 / 72.0) * a(0, 0, 0) + \
    (DW_DTYPE)(8.0 / 5.0) * (a(0, 1, 0) + a(0, -1, 0)) + \
    (DW_DTYPE)(-1.0 / 5.0) * (a(0, 2, 0) + a(0, -2, 0)) + \
    (DW_DTYPE)(8.0 / 315.0) * (a(0, 3, 0) + a(0, -3, 0)) + \
    (DW_DTYPE)(-1.0 / 560.0) * (a(0, 4, 0) + a(0, -4, 0))) * rdy2)
#define DIFFX1(a) \
  (((DW_DTYPE)(4.0 / 5.0) * (a(0, 0, 1) - a(0, 0, -1)) + \
    (DW_DTYPE)(-1.0 / 5.0) * (a(0, 0, 2) - a(0, 0, -2)) + \
    (DW_DTYPE)(4.0 / 105.0) * (a(0, 0, 3) - a(0, 0, -3)) + \
    (DW_DTYPE)(-1.0 / 280.0) * (a(0, 0, 4) - a(0, 0, -4))) * rdx)
#define DIFFX2(a) \
  (((DW_DTYPE)(-205.0 / 72.0) * a(0, 0, 0) + \
    (DW_DTYPE)(8.0 / 5.0) * (a(0, 0, 1) + a(0, 0, -1)) + \
    (DW_DTYPE)(-1.0 / 5.0) * (a(0, 0, 2) + a(0, 0, -2)) + \
    (DW_DTYPE)(8.0 / 315.0) * (a(0, 0, 3) + a(0, 0, -3)) + \
    (DW_DTYPE)(-1.0 / 560.0) * (a(0, 0, 4) + a(0, 0, -4))) * rdx2)
#define FD_PAD 4

#else
#error DW_ACCURACY must be specified
#endif // DW_ACCURACY

#ifdef DW_DEBUG
#define CHECK_KERNEL_ERROR { \
    gpuErrchk(cudaPeekAtLastError()); \
    gpuErrchk(cudaDeviceSynchronize()); \
}
#else
#define CHECK_KERNEL_ERROR gpuErrchk(cudaPeekAtLastError());
#endif // DW_DEBUG

#endif // DW_COMMON_H
