/*
 * Scalar wave equation propagator
 *
 * See scalar.c
 */

#include <stdio.h>

#include <cstdint>

#include "common.h"

#define CAT_I(name, accuracy, dtype) scalar_iso_3d_##accuracy##_##dtype##_##name
#define CAT(name, accuracy, dtype) CAT_I(name, accuracy, dtype)
#define FUNC(name) CAT(name, DW_ACCURACY, DW_DTYPE)


// 三维索引宏
// #define IDX(dz, dy, dx) ((z + dz) * nynx + (y + dy) * nx + (x + dx))

// 三维波场访问
#define WFC(dz, dy, dx) wfc[i + dz * nynx + dy * nx + dx]
#define WFC_FWD(dz, dy, dx) wfc_forward[i + (dz) * nynx + (dy) * nx + (dx)]
#define WFP_FWD(dz, dy, dx) wfp_forward[i + (dz) * nynx + (dy) * nx + (dx)]
#define AY_PSIY(dz, dy, dx) ay[y + dy] * psiy[i + dz * nynx + dy * nx + dx]
#define AX_PSIX(dz, dy, dx) ax[x + dx] * psix[i + dz * nynx + dy * nx + dx]
#define AZ_PSIZ(dz, dy, dx) az[z + dz] * psiz[i + dz * nynx + dy * nx + dx]
#define V2DT2_WFC(dz, dy, dx) v2dt2[j + dz * nynx + dy * nx + dx] * wfc[i + dz * nynx + dy * nx + dx]

// PML项和有限差分项（z/y/x方向）

#define UT_TERMZ1(dz, dy, dx) \
  (dbzdz[z + dz] * ((1 + bz[z + dz]) * v2dt2[j + dz * nynx + dy * nx + dx] * wfc[i + dz * nynx + dy * nx + dx] + \
                    bz[z + dz] * zetaz[i + dz * nynx + dy * nx + dx]) + \
   bz[z + dz] * psiz[i + dz * nynx + dy * nx + dx])

#define UT_TERMY1(dz, dy, dx) \
  (dbydy[y + dy] * ((1 + by[y + dy]) * v2dt2[j + dz * nynx + dy * nx + dx] * wfc[i + dz * nynx + dy * nx + dx] + \
                    by[y + dy] * zetay[i + dz * nynx + dy * nx + dx]) + \
   by[y + dy] * psiy[i + dz * nynx + dy * nx + dx])

#define UT_TERMX1(dz, dy, dx) \
  (dbxdx[x + dx] * ((1 + bx[x + dx]) * v2dt2[j + dz * nynx + dy * nx + dx] * wfc[i + dz * nynx + dy * nx + dx] + \
                    bx[x + dx] * zetax[i + dz * nynx + dy * nx + dx]) + \
   bx[x + dx] * psix[i + dz * nynx + dy * nx + dx])

#define UT_TERMZ2(dz, dy, dx) \
  ((1 + bz[z + dz]) * ((1 + bz[z + dz]) * v2dt2[j + dz * nynx + dy * nx + dx] * wfc[i + dz * nynx + dy * nx + dx] + \
                       bz[z + dz] * zetaz[i + dz * nynx + dy * nx + dx]))

#define UT_TERMY2(dz, dy, dx) \
  ((1 + by[y + dy]) * ((1 + by[y + dy]) * v2dt2[j + dz * nynx + dy * nx + dx] * wfc[i + dz * nynx + dy * nx + dx] + \
                       by[y + dy] * zetay[i + dz * nynx + dy * nx + dx]))

#define UT_TERMX2(dz, dy, dx) \
  ((1 + bx[x + dx]) * ((1 + bx[x + dx]) * v2dt2[j + dz * nynx + dy * nx + dx] * wfc[i + dz * nynx + dy * nx + dx] + \
                       bx[x + dx] * zetax[i + dz * nynx + dy * nx + dx]))

#define PSIZ_TERM(dz, dy, dx) \
  ((1 + bz[z + dz]) * v2dt2[j + dz * nynx + dy * nx + dx] * wfc[i + dz * nynx + dy * nx + dx] + \
   bz[z + dz] * zetaz[i + dz * nynx + dy * nx + dx])

#define PSIY_TERM(dz, dy, dx) \
  ((1 + by[y + dy]) * v2dt2[j + dz * nynx + dy * nx + dx] * wfc[i + dz * nynx + dy * nx + dx] + \
   by[y + dy] * zetay[i + dz * nynx + dy * nx + dx])

#define PSIX_TERM(dz, dy, dx) \
  ((1 + bx[x + dx]) * v2dt2[j + dz * nynx + dy * nx + dx] * wfc[i + dz * nynx + dy * nx + dx] + \
   bx[x + dx] * zetax[i + dz * nynx + dy * nx + dx])

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

namespace {

__constant__ DW_DTYPE dt2;
__constant__ DW_DTYPE rdz;
__constant__ DW_DTYPE rdy;
__constant__ DW_DTYPE rdx;
__constant__ DW_DTYPE rdz2;
__constant__ DW_DTYPE rdy2;
__constant__ DW_DTYPE rdx2;
__constant__ int64_t n_shots;
__constant__ int64_t nz;
__constant__ int64_t ny;
__constant__ int64_t nx;
__constant__ int64_t nynx;
__constant__ int64_t nznynx;
__constant__ int64_t n_sources_per_shot;
__constant__ int64_t n_receivers_per_shot;
__constant__ int64_t step_ratio;
__constant__ int64_t pml_z0;
__constant__ int64_t pml_z1;
__constant__ int64_t pml_y0;
__constant__ int64_t pml_y1;
__constant__ int64_t pml_x0;
__constant__ int64_t pml_x1;



__global__ void add_sources(DW_DTYPE *__restrict const wf,
                            DW_DTYPE const *__restrict const f,
                            int64_t const *__restrict const sources_i) {
  int64_t source_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (source_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_sources_per_shot + source_idx;
    wf[shot_idx * nznynx + sources_i[k]] += f[k];
  }
}


__global__ void record_receivers(DW_DTYPE *__restrict const r,
                                 DW_DTYPE const *__restrict const wf,
                                 int64_t const *__restrict receivers_i) {
  int64_t receiver_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t shot_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (receiver_idx < n_receivers_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_receivers_per_shot + receiver_idx;
    r[k] = wf[shot_idx * nznynx + receivers_i[k]];
  }
}



__global__ void combine_grad_v(DW_DTYPE *__restrict__ const grad_v,
                               DW_DTYPE const *__restrict__ const grad_v_shot) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;
  int64_t i = z * nynx + y * nx + x;
  if (y < ny - FD_PAD && x < nx - FD_PAD && z < nz - FD_PAD) {
    int64_t shot_idx;
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      grad_v[i] += grad_v_shot[shot_idx * nznynx + i];
    }
  }
}


// 三维 forward_kernel
__global__ void forward_kernel(
    DW_DTYPE const *__restrict const v, 
    DW_DTYPE const *__restrict const wfc, DW_DTYPE *__restrict const wfp, 
    DW_DTYPE const *__restrict const psiz, DW_DTYPE const *__restrict const psiy, DW_DTYPE const *__restrict const psix, 
    DW_DTYPE *__restrict const psizn, DW_DTYPE *__restrict const psiyn, DW_DTYPE *__restrict const psixn, 
    DW_DTYPE *__restrict const zetaz, DW_DTYPE *__restrict const zetay, DW_DTYPE *__restrict const zetax, 
    DW_DTYPE *__restrict const boundary_front,
    DW_DTYPE *__restrict const boundary_back,
    DW_DTYPE *__restrict const boundary_top,
    DW_DTYPE *__restrict const boundary_bottom,
    DW_DTYPE *__restrict const boundary_left,
    DW_DTYPE *__restrict const boundary_right,
    DW_DTYPE *__restrict const wfc_last,
    DW_DTYPE *__restrict const wfp_last,
    DW_DTYPE *__restrict const wfc_mid,
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const *__restrict const dbzdz, DW_DTYPE const *__restrict const dbydy, DW_DTYPE const *__restrict const dbxdx,
    int64_t shot_idx,
    int64_t const N_boundary,
    int64_t t,
    int64_t nt,
    bool const v_requires_grad) {

  int64_t x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;


  if (z < nz - FD_PAD && y < ny - FD_PAD && x < nx - FD_PAD) {
    //int64_t shot_idx=0;
    int64_t j = z * nynx + y * nx + x;
    int64_t i = shot_idx * nznynx + j;
    bool pml_z = z < pml_z0 || z >= pml_z1;
    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;
    DW_DTYPE w_sum;
    if (!pml_z) {
      w_sum = DIFFZ2(WFC);
    } else {
      DW_DTYPE dwfcdz = DIFFZ1(WFC);
      DW_DTYPE tmpz =
          ((1 + bz[z]) * DIFFZ2(WFC) + dbzdz[z] * dwfcdz + DIFFZ1(AZ_PSIZ));
      w_sum = (1 + bz[z]) * tmpz + az[z] * zetaz[i];
      psizn[i] = bz[z] * dwfcdz + az[z] * psiz[i];
      zetaz[i] = bz[z] * tmpz + az[z] * zetaz[i];
    }
    if (!pml_y) {
      w_sum += DIFFY2(WFC);
    } else {
      DW_DTYPE dwfcdy = DIFFY1(WFC);
      DW_DTYPE tmpy =
          ((1 + by[y]) * DIFFY2(WFC) + dbydy[y] * dwfcdy + DIFFY1(AY_PSIY));
      w_sum += (1 + by[y]) * tmpy + ay[y] * zetay[i];
      psiyn[i] = by[y] * dwfcdy + ay[y] * psiy[i];
      zetay[i] = by[y] * tmpy + ay[y] * zetay[i];
    }
    if (!pml_x) {
      w_sum += DIFFX2(WFC);
    } else {
      DW_DTYPE dwfcdx = DIFFX1(WFC);
      DW_DTYPE tmpx =
          ((1 + bx[x]) * DIFFX2(WFC) + dbxdx[x] * dwfcdx + DIFFX1(AX_PSIX));
      w_sum += (1 + bx[x]) * tmpx + ax[x] * zetax[i];
      psixn[i] = bx[x] * dwfcdx + ax[x] * psix[i];
      zetax[i] = bx[x] * tmpx + ax[x] * zetax[i];
    }

    wfp[i] = v[j] * v[j] * dt2 * w_sum + 2 * wfc[i] - wfp[i];

    if (v_requires_grad) {
        // 1. 只在最后两个时间步保存完整波场
        if (t == nt - 1) {
            wfc_last[i] = wfc[i];
            wfp_last[i] = wfp[i];
        } else if (t == nt - 2) {
            wfc_last[i + nznynx * n_shots] = wfc[i];  // 存储倒数第二帧
            wfp_last[i + nznynx * n_shots] = wfp[i];
        }
        
        // 2. 保存特定时间步的检查点波场
        if (t == nt - 3 - 3 * 1) { 
            wfc_mid[i] = wfc[i];
        }

        // 3. 边界存储：只在检查点时间步存储
        if ((t % step_ratio) == 0) {
            int checkpoint_idx = t / step_ratio;
            
            // Z方向前边界 (PML外边界前N_boundary层)
            if (z >= pml_z0 - N_boundary && z < pml_z0) {
                int zb = z - (pml_z0 - N_boundary);
                int boundary_idx = checkpoint_idx * N_boundary * ny * nx * n_shots + 
                                  shot_idx * N_boundary * ny * nx + 
                                  zb * ny * nx + y * nx + x;
                boundary_front[boundary_idx] = wfc[i];
            }
            
            // Z方向后边界 (PML外边界后N_boundary层)
            if (z >= pml_z1 && z < pml_z1 + N_boundary) {
                int zb = z - pml_z1;
                int boundary_idx = checkpoint_idx * N_boundary * ny * nx * n_shots + 
                                  shot_idx * N_boundary * ny * nx + 
                                  zb * ny * nx + y * nx + x;
                boundary_back[boundary_idx] = wfc[i];
            }
            
            // Y方向前边界
            if (y >= pml_y0 - N_boundary && y < pml_y0) {
                int yb = y - (pml_y0 - N_boundary);
                int boundary_idx = checkpoint_idx * N_boundary * nz * nx * n_shots + 
                                  shot_idx * N_boundary * nz * nx + 
                                  yb * nz * nx + z * nx + x;
                boundary_top[boundary_idx] = wfc[i];
            }
            
            // Y方向后边界
            if (y >= pml_y1 && y < pml_y1 + N_boundary) {
                int yb = y - pml_y1;
                int boundary_idx = checkpoint_idx * N_boundary * nz * nx * n_shots + 
                                  shot_idx * N_boundary * nz * nx + 
                                  yb * nz * nx + z * nx + x;
                boundary_bottom[boundary_idx] = wfc[i];
            }
            
            // X方向前边界
            if (x >= pml_x0 - N_boundary && x < pml_x0) {
                int xb = x - (pml_x0 - N_boundary);
                int boundary_idx = checkpoint_idx * N_boundary * nz * ny * n_shots + 
                                  shot_idx * N_boundary * nz * ny + 
                                  xb * nz * ny + z * ny + y;
                boundary_left[boundary_idx] = wfc[i];
            }
            
            // X方向后边界
            if (x >= pml_x1 && x < pml_x1 + N_boundary) {
                int xb = x - pml_x1;
                int boundary_idx = checkpoint_idx * N_boundary * nz * ny * n_shots + 
                                  shot_idx * N_boundary * nz * ny + 
                                  xb * nz * ny + z * ny + y;
                boundary_right[boundary_idx] = wfc[i];
            }
        }
    }

  }
}

// 三维 backward_kernel
__global__ void backward_kernel(
    DW_DTYPE const *__restrict const v, 
    DW_DTYPE const *__restrict const v2dt2,
    DW_DTYPE *__restrict__ wfc, DW_DTYPE *__restrict const wfp,
    DW_DTYPE* __restrict wfc_forward,
    DW_DTYPE* __restrict wfp_forward,
    DW_DTYPE* __restrict wfm_forward,
    DW_DTYPE const *__restrict const psiz, DW_DTYPE const *__restrict const psiy, DW_DTYPE const *__restrict const psix, 
    DW_DTYPE *__restrict const psizn, DW_DTYPE *__restrict const psiyn, DW_DTYPE *__restrict const psixn, 
    DW_DTYPE *__restrict const zetaz, DW_DTYPE *__restrict const zetay, DW_DTYPE *__restrict const zetax, 
    DW_DTYPE *__restrict const zetazn, DW_DTYPE *__restrict const zetayn, DW_DTYPE *__restrict const zetaxn, 
    DW_DTYPE const *__restrict const boundary_front,
    DW_DTYPE const *__restrict const boundary_back,
    DW_DTYPE const *__restrict const boundary_top,
    DW_DTYPE const *__restrict const boundary_bottom,
    DW_DTYPE const *__restrict const boundary_left,
    DW_DTYPE const *__restrict const boundary_right,
    DW_DTYPE const *__restrict const wfc_last,
    DW_DTYPE const *__restrict const wfp_last,
    DW_DTYPE *__restrict const grad_v, 
    DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ax, 
    DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const bx, 
    DW_DTYPE const *__restrict const dbzdz, DW_DTYPE const *__restrict const dbydy, DW_DTYPE const *__restrict const dbxdx,
    int64_t shot_idx,
    int64_t const N_boundary,
    int64_t t,
    int64_t nt,
    int64_t step_ratio_h,
    bool const v_requires_grad) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
  int64_t y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
  int64_t z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;

  if (z < nz - FD_PAD && y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t j = z * nynx + y * nx + x;
    int64_t i = shot_idx * nznynx + j;


    bool pml_z = z < pml_z0 || z >= pml_z1;
    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;

    wfp[i] =
        (pml_z ? -DIFFZ1(UT_TERMZ1) + DIFFZ2(UT_TERMZ2) : DIFFZ2(V2DT2_WFC)) +            
        (pml_y ? -DIFFY1(UT_TERMY1) + DIFFY2(UT_TERMY2) : DIFFY2(V2DT2_WFC)) +
        (pml_x ? -DIFFX1(UT_TERMX1) + DIFFX2(UT_TERMX2) : DIFFX2(V2DT2_WFC)) +
        2 * wfc[i] - wfp[i];

    if (pml_z) {
      psizn[i] = -az[z] * DIFFZ1(PSIZ_TERM) + az[z] * psiz[i];
      zetazn[i] = az[z] * v2dt2[j] * wfc[i] + az[z] * zetaz[i];
    }
    if (pml_y) {
      psiyn[i] = -ay[y] * DIFFY1(PSIY_TERM) + ay[y] * psiy[i];
      zetayn[i] = ay[y] * v2dt2[j] * wfc[i] + ay[y] * zetay[i];
    }
    if (pml_x) {
      psixn[i] = -ax[x] * DIFFX1(PSIX_TERM) + ax[x] * psix[i];
      zetaxn[i] = ax[x] * v2dt2[j] * wfc[i] + ax[x] * zetax[i];
    }

    
    //if (v_requires_grad) {
    //  grad_v[i] += 2 * v[j] * dt2 * wfc_forward[i] * wfc[i] * step_ratio;
    //}

    if (v_requires_grad) {
        int forward_t = nt - 1 - t;  // 对应的正演时间步
        
        // 1. 初始化：如果是最后一个反向时间步，从缓存中恢复最后两帧
        if (t == nt - 1) {
            // 最后一帧
            wfc_forward[i] = wfc_last[i];
            wfp_forward[i] = wfp_last[i];
        } else if (t == nt - 2) {
            // 倒数第二帧
            wfc_forward[i] = wfc_last[i + nznynx * n_shots];
            wfp_forward[i] = wfp_last[i + nznynx * n_shots];
        } else {
            // 2. 重建非最后两帧的正演波场
            
            // 从边界恢复波场（使用正确的时间索引和范围）
            bool is_boundary_restored = false;
            int checkpoint_idx = forward_t / step_ratio_h;
            
            // Z方向前边界恢复（与正向存储范围一致）
            if (z >= pml_z0 - N_boundary && z < pml_z0) {
                int zb = z - (pml_z0 - N_boundary);
                int boundary_idx = checkpoint_idx * N_boundary * ny * nx * n_shots + 
                                  shot_idx * N_boundary * ny * nx + 
                                  zb * ny * nx + y * nx + x;
                wfc_forward[i] = boundary_front[boundary_idx];
                is_boundary_restored = true;
            }
            // Z方向后边界恢复
            else if (z >= pml_z1 && z < pml_z1 + N_boundary) {
                int zb = z - pml_z1;
                int boundary_idx = checkpoint_idx * N_boundary * ny * nx * n_shots + 
                                  shot_idx * N_boundary * ny * nx + 
                                  zb * ny * nx + y * nx + x;
                wfc_forward[i] = boundary_back[boundary_idx];
                is_boundary_restored = true;
            }
            // Y方向前边界恢复
            else if (y >= pml_y0 - N_boundary && y < pml_y0) {
                int yb = y - (pml_y0 - N_boundary);
                int boundary_idx = checkpoint_idx * N_boundary * nz * nx * n_shots + 
                                  shot_idx * N_boundary * nz * nx + 
                                  yb * nz * nx + z * nx + x;
                wfc_forward[i] = boundary_top[boundary_idx];
                is_boundary_restored = true;
            }
            // Y方向后边界恢复
            else if (y >= pml_y1 && y < pml_y1 + N_boundary) {
                int yb = y - pml_y1;
                int boundary_idx = checkpoint_idx * N_boundary * nz * nx * n_shots + 
                                  shot_idx * N_boundary * nz * nx + 
                                  yb * nz * nx + z * nx + x;
                wfc_forward[i] = boundary_bottom[boundary_idx];
                is_boundary_restored = true;
            }
            // X方向前边界恢复
            else if (x >= pml_x0 - N_boundary && x < pml_x0) {
                int xb = x - (pml_x0 - N_boundary);
                int boundary_idx = checkpoint_idx * N_boundary * nz * ny * n_shots + 
                                  shot_idx * N_boundary * nz * ny + 
                                  xb * nz * ny + z * ny + y;
                wfc_forward[i] = boundary_left[boundary_idx];
                is_boundary_restored = true;
            }
            // X方向后边界恢复
            else if (x >= pml_x1 && x < pml_x1 + N_boundary) {
                int xb = x - pml_x1;
                int boundary_idx = checkpoint_idx * N_boundary * nz * ny * n_shots + 
                                  shot_idx * N_boundary * nz * ny + 
                                  xb * nz * ny + z * ny + y;
                wfc_forward[i] = boundary_right[boundary_idx];
                is_boundary_restored = true;
            }
            
            // 3. 对于非边界点，通过时间步进重建
            if (!is_boundary_restored) {
                // 简化的内部区域重建（非PML区域）
                if (!pml_z && !pml_y && !pml_x) {
                    DW_DTYPE w_sum = DIFFZ2(WFC_FWD) + DIFFY2(WFC_FWD) + DIFFX2(WFC_FWD);
                    DW_DTYPE new_wfp = v2dt2[j] * w_sum + 2 * wfc_forward[i] - wfp_forward[i];
                    
                    // 更新波场指针
                    DW_DTYPE temp = wfc_forward[i];
                    wfc_forward[i] = new_wfp;
                    wfp_forward[i] = temp;
                } else {
                    // PML区域需要更复杂的重建，这里简化处理
                    // 实际应用中可能需要从最近的检查点开始完整重建
                    wfc_forward[i] = 0.0;  // 临时处理
                }
            }
        }
        
        // 4. 计算梯度（成像条件）
        // 使用零延迟交叉相关
        if ((forward_t % step_ratio_h) == 0) {
            grad_v[i] += 2 * v[j] * dt2 * wfc_forward[i] * wfc[i] * step_ratio_h;
        }
    }

    //if (t == nt-1500) { 
    //    grad_v[i] =  wfc_forward[i];
    //}
  }
}
// 注意：__global__ 函数体结束后不能直接定义 inline 函数，需移出 namespace 或放到文件后部

// ===================== 文件末尾补充 =====================
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// 向上取整除法
inline unsigned int ceil_div(unsigned int numerator, unsigned int denominator) {
  return (numerator + denominator - 1) / denominator;
}

void set_config(
    DW_DTYPE const dt2_h, DW_DTYPE const rdz_h, DW_DTYPE const rdy_h, DW_DTYPE const rdx_h,
    DW_DTYPE const rdz2_h, DW_DTYPE const rdy2_h, DW_DTYPE const rdx2_h,
    int64_t const n_shots_h, int64_t const nz_h, int64_t const ny_h, int64_t const nx_h,
    int64_t const n_sources_per_shot_h, int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h,
    int64_t const pml_z0_h, int64_t const pml_z1_h,
    int64_t const pml_y0_h, int64_t const pml_y1_h,
    int64_t const pml_x0_h, int64_t const pml_x1_h) {
  int64_t const nynx_h = ny_h * nx_h;
  int64_t const nznynx_h = nz_h * nynx_h;
  gpuErrchk(cudaMemcpyToSymbol(dt2, &dt2_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdz, &rdz_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdz2, &rdz2_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdy2, &rdy2_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdx2, &rdx2_h, sizeof(DW_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(nz, &nz_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(nynx, &nynx_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(nznynx, &nznynx_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_sources_per_shot, &n_sources_per_shot_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(step_ratio, &step_ratio_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_z0, &pml_z0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_z1, &pml_z1_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t)));
}

}  // namespace

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
        void FUNC(forward)(
            DW_DTYPE const *__restrict const v,
            DW_DTYPE const *__restrict const f, DW_DTYPE *__restrict const wfc,
            DW_DTYPE *__restrict const wfp, 
            DW_DTYPE *__restrict const psiz, DW_DTYPE *__restrict const psiy, DW_DTYPE *__restrict const psix, 
            DW_DTYPE *__restrict const psizn, DW_DTYPE *__restrict const psiyn, DW_DTYPE *__restrict const psixn, 
            DW_DTYPE *__restrict const zetaz, DW_DTYPE *__restrict const zetay, DW_DTYPE *__restrict const zetax, 
            DW_DTYPE *__restrict const boundary_front,
            DW_DTYPE *__restrict const boundary_back,
            DW_DTYPE *__restrict const boundary_top,
            DW_DTYPE *__restrict const boundary_bottom,
            DW_DTYPE *__restrict const boundary_left,
            DW_DTYPE *__restrict const boundary_right,
            DW_DTYPE *__restrict const wfc_last,
            DW_DTYPE *__restrict const wfp_last,
            DW_DTYPE *__restrict const wfc_mid,
            DW_DTYPE *__restrict const r, 
            DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ax,
            DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const bx,
            DW_DTYPE const *__restrict const dbzdz, DW_DTYPE const *__restrict const dbydy, DW_DTYPE const *__restrict const dbxdx,
            int64_t const *__restrict const sources_i,
            int64_t const *__restrict const receivers_i,
            DW_DTYPE const rdz_h, DW_DTYPE const rdy_h, DW_DTYPE const rdx_h,
            DW_DTYPE const rdz2_h, DW_DTYPE const rdy2_h, DW_DTYPE const rdx2_h,
            DW_DTYPE const dt2_h, int64_t const nt, int64_t const n_shots_h,
            int64_t const nz_h, int64_t const ny_h, int64_t const nx_h,
            int64_t const n_sources_per_shot_h,
            int64_t const n_receivers_per_shot_h, int64_t const step_ratio_h,
            bool const v_requires_grad,
            int64_t const pml_z0_h, int64_t const pml_z1_h,
            int64_t const pml_y0_h, int64_t const pml_y1_h,
            int64_t const pml_x0_h, int64_t const pml_x1_h,
            int64_t const N_boundary,
            int64_t const device) {
  dim3 dimBlock(8, 8, 8);
  unsigned int gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int gridz = ceil_div(nz_h - 2 * FD_PAD, dimBlock.z);
  dim3 dimGrid(gridx, gridy, gridz);

  dim3 dimBlock_sources(32, 1, 1);
  unsigned int gridx_sources =
      ceil_div(n_sources_per_shot_h, dimBlock_sources.x);
  unsigned int gridy_sources = ceil_div(n_shots_h, dimBlock_sources.y);
  unsigned int gridz_sources = 1;
  dim3 dimGrid_sources(gridx_sources, gridy_sources, gridz_sources);

  dim3 dimBlock_receivers(32, 1, 1);
  unsigned int gridx_receivers =
      ceil_div(n_receivers_per_shot_h, dimBlock_receivers.x);
  unsigned int gridy_receivers = ceil_div(n_shots_h, dimBlock_receivers.y);
  unsigned int gridz_receivers = 1;
  dim3 dimGrid_receivers(gridx_receivers, gridy_receivers, gridz_receivers);

  int64_t t;
  int64_t shot_idx;
  gpuErrchk(cudaSetDevice(device));
  set_config(dt2_h, rdz_h, rdy_h, rdx_h, rdz2_h, rdy2_h, rdx2_h,
             n_shots_h, nz_h, ny_h, nx_h,
             n_sources_per_shot_h, n_receivers_per_shot_h, step_ratio_h,
             pml_z0_h, pml_z1_h, pml_y0_h, pml_y1_h, pml_x0_h, pml_x1_h);
  for (t = 0; t < nt; ++t) {
    if (t & 1) {
      for (shot_idx = 0; shot_idx < n_shots_h; ++shot_idx) {
      forward_kernel<<<dimGrid, dimBlock>>>(
          v, wfp, wfc, psizn, psiyn, psixn, psiz, psiy, psix, zetaz, zetay, zetax,
          boundary_front + (t / step_ratio_h) * N_boundary * ny_h * nx_h * n_shots_h,
          boundary_back +  (t / step_ratio_h) * N_boundary * ny_h * nx_h * n_shots_h,
          boundary_top +   (t / step_ratio_h) * N_boundary * nz_h * nx_h * n_shots_h,
          boundary_bottom +(t / step_ratio_h) * N_boundary * nz_h * nx_h * n_shots_h,
          boundary_left +  (t / step_ratio_h) * N_boundary * nz_h * ny_h * n_shots_h,
          boundary_right + (t / step_ratio_h) * N_boundary * nz_h * ny_h * n_shots_h, 
          wfc_last, wfp_last, wfc_mid,
          az, ay, ax, bz, by, bx,
          dbzdz, dbydy, dbxdx,
          shot_idx,
          N_boundary,
          t, nt,
          v_requires_grad && ((t % step_ratio_h) == 0)
          );
      }
      CHECK_KERNEL_ERROR
      if (n_sources_per_shot_h > 0) {
        add_sources<<<dimGrid_sources, dimBlock_sources>>>(
            wfc, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
        CHECK_KERNEL_ERROR
      }
      if (n_receivers_per_shot_h > 0) {
        record_receivers<<<dimGrid_receivers, dimBlock_receivers>>>(
            r + t * n_shots_h * n_receivers_per_shot_h, wfp, receivers_i);
        CHECK_KERNEL_ERROR
      }
    } else {
      for (shot_idx = 0; shot_idx < n_shots_h; ++shot_idx) {
      forward_kernel<<<dimGrid, dimBlock>>>(
          v, wfc, wfp, psiz, psiy, psix, psizn, psiyn, psixn, zetaz, zetay, zetax,
          boundary_front + (t / step_ratio_h) * N_boundary * ny_h * nx_h * n_shots_h,
          boundary_back +  (t / step_ratio_h) * N_boundary * ny_h * nx_h * n_shots_h,
          boundary_top +   (t / step_ratio_h) * N_boundary * nz_h * nx_h * n_shots_h,
          boundary_bottom +(t / step_ratio_h) * N_boundary * nz_h * nx_h * n_shots_h,
          boundary_left +  (t / step_ratio_h) * N_boundary * nz_h * ny_h * n_shots_h,
          boundary_right + (t / step_ratio_h) * N_boundary * nz_h * ny_h * n_shots_h, 
          wfc_last, wfp_last, wfc_mid,
          az, ay, ax, bz, by, bx,
          dbzdz, dbydy, dbxdx,
          shot_idx,
          N_boundary,
          t, nt,
          v_requires_grad && ((t % step_ratio_h) == 0)
          );
      }
      CHECK_KERNEL_ERROR
      if (n_sources_per_shot_h > 0) {
        add_sources<<<dimGrid_sources, dimBlock_sources>>>(
            wfp, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
        CHECK_KERNEL_ERROR
      }
      if (n_receivers_per_shot_h > 0) {
        record_receivers<<<dimGrid_receivers, dimBlock_receivers>>>(
            r + t * n_shots_h * n_receivers_per_shot_h, wfc, receivers_i);
        CHECK_KERNEL_ERROR
      }
    }
  }
}

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
        void FUNC(backward)(
            DW_DTYPE const *__restrict const v,
            DW_DTYPE const *__restrict const v2dt2,
            DW_DTYPE const *__restrict const grad_r,
            DW_DTYPE *__restrict const wfc, DW_DTYPE *__restrict const wfp,
            DW_DTYPE *__restrict const psiz, DW_DTYPE *__restrict const psiy, DW_DTYPE *__restrict const psix,
            DW_DTYPE *__restrict const psizn, DW_DTYPE *__restrict const psiyn, DW_DTYPE *__restrict const psixn,
            DW_DTYPE *__restrict const zetaz, DW_DTYPE *__restrict const zetay, DW_DTYPE *__restrict const zetax,
            DW_DTYPE *__restrict const zetazn, DW_DTYPE *__restrict const zetayn, DW_DTYPE *__restrict const zetaxn,
            DW_DTYPE const *__restrict const boundary_front,
            DW_DTYPE const *__restrict const boundary_back,
            DW_DTYPE const *__restrict const boundary_top,
            DW_DTYPE const *__restrict const boundary_bottom,
            DW_DTYPE const *__restrict const boundary_left,
            DW_DTYPE const *__restrict const boundary_right,
            DW_DTYPE const *__restrict const wfc_last,    // 新增：正向最后两帧波场缓存
            DW_DTYPE const *__restrict const wfp_last,
            DW_DTYPE *__restrict const grad_f,
            DW_DTYPE *__restrict const grad_v,
            DW_DTYPE *__restrict const grad_v_shot,
            DW_DTYPE const *__restrict const az, DW_DTYPE const *__restrict const ay, DW_DTYPE const *__restrict const ax,
            DW_DTYPE const *__restrict const bz, DW_DTYPE const *__restrict const by, DW_DTYPE const *__restrict const bx,
            DW_DTYPE const *__restrict const dbzdz, DW_DTYPE const *__restrict const dbydy, DW_DTYPE const *__restrict const dbxdx,
            int64_t const *__restrict const sources_i,
            int64_t const *__restrict const receivers_i,
            DW_DTYPE const rdz_h, DW_DTYPE const rdy_h, DW_DTYPE const rdx_h,
            DW_DTYPE const rdz2_h, DW_DTYPE const rdy2_h, DW_DTYPE const rdx2_h,
            int64_t const nt, int64_t const n_shots_h,
            int64_t const nz_h, int64_t const ny_h, int64_t const nx_h,
            int64_t const n_sources_per_shot_h,
            int64_t const n_receivers_per_shot_h, int64_t const step_ratio_h,
            bool const v_requires_grad,
            int64_t const pml_z0_h, int64_t const pml_z1_h,
            int64_t const pml_y0_h, int64_t const pml_y1_h,
            int64_t const pml_x0_h, int64_t const pml_x1_h,
            int64_t const N_boundary,
            int64_t const device) {
  dim3 dimBlock(8, 8, 8);
  unsigned int gridx = ceil_div(nx_h - 2 * FD_PAD, dimBlock.x);
  unsigned int gridy = ceil_div(ny_h - 2 * FD_PAD, dimBlock.y);
  unsigned int gridz = ceil_div(nz_h - 2 * FD_PAD, dimBlock.z);
  dim3 dimGrid(gridx, gridy, gridz);

  dim3 dimBlock_sources(32, 1, 1);
  unsigned int gridx_sources =
      ceil_div(n_sources_per_shot_h, dimBlock_sources.x);
  unsigned int gridy_sources = ceil_div(n_shots_h, dimBlock_sources.y);
  unsigned int gridz_sources = 1;
  dim3 dimGrid_sources(gridx_sources, gridy_sources, gridz_sources);

  dim3 dimBlock_receivers(32, 1, 1);
  unsigned int gridx_receivers =
      ceil_div(n_receivers_per_shot_h, dimBlock_receivers.x);
  unsigned int gridy_receivers = ceil_div(n_shots_h, dimBlock_receivers.y);
  unsigned int gridz_receivers = 1;
  dim3 dimGrid_receivers(gridx_receivers, gridy_receivers, gridz_receivers);

  dim3 dimBlock_combine(8, 8, 8);
  unsigned int gridx_combine = ceil_div(nx_h - 2 * FD_PAD, dimBlock_combine.x);
  unsigned int gridy_combine = ceil_div(ny_h - 2 * FD_PAD, dimBlock_combine.y);
  unsigned int gridz_combine = ceil_div(nz_h - 2 * FD_PAD, dimBlock_combine.z);
  dim3 dimGrid_combine(gridx_combine, gridy_combine, gridz_combine);

  int64_t t;
  int64_t shot_idx;

  size_t total_size = (size_t)n_shots_h * nz_h * ny_h * nx_h;
  DW_DTYPE* wfc_forward;
  DW_DTYPE* wfp_forward;
  DW_DTYPE* wfm_forward;
  gpuErrchk(cudaMalloc(&wfc_forward, total_size * sizeof(DW_DTYPE)));
  gpuErrchk(cudaMalloc(&wfp_forward, total_size * sizeof(DW_DTYPE)));
  gpuErrchk(cudaMalloc(&wfm_forward, total_size * sizeof(DW_DTYPE)));

  gpuErrchk(cudaSetDevice(device));
  set_config(0, rdz_h, rdy_h, rdx_h, rdz2_h, rdy2_h, rdx2_h,
             n_shots_h, nz_h, ny_h, nx_h,
             n_receivers_per_shot_h, n_sources_per_shot_h, step_ratio_h,
             pml_z0_h, pml_z1_h, pml_y0_h, pml_y1_h, pml_x0_h, pml_x1_h);

  for (t = nt - 1; t >= 0; --t) {
    if ((nt - 1 - t) & 1) {
      if (n_sources_per_shot_h > 0) {
        record_receivers<<<dimGrid_sources, dimBlock_sources>>>(
            grad_f + t * n_shots_h * n_sources_per_shot_h, wfp, sources_i);
        CHECK_KERNEL_ERROR
      }
      for (shot_idx = 0; shot_idx < n_shots_h; ++shot_idx) {
      backward_kernel<<<dimGrid, dimBlock>>>(
          v, v2dt2, wfp, wfc, wfp_forward, wfc_forward, wfm_forward, psizn, psiyn, psixn, psiz, psiy, psix, zetazn, zetayn, zetaxn, zetaz, zetay, zetax, 
          boundary_front + (t / step_ratio_h) * N_boundary * ny_h * nx_h * n_shots_h,
          boundary_back  + (t / step_ratio_h) * N_boundary * ny_h * nx_h * n_shots_h,
          boundary_top   + (t / step_ratio_h) * N_boundary * nz_h * nx_h * n_shots_h,
          boundary_bottom+ (t / step_ratio_h) * N_boundary * nz_h * nx_h * n_shots_h,
          boundary_left  + (t / step_ratio_h) * N_boundary * nz_h * ny_h * n_shots_h,
          boundary_right + (t / step_ratio_h) * N_boundary * nz_h * ny_h * n_shots_h,
          wfc_last, wfp_last,
          grad_v_shot, az, ay, ax, bz, by, bx, dbzdz, dbydy, dbxdx,
          shot_idx,
          N_boundary,
          t, nt, step_ratio_h,
          v_requires_grad && ((t % step_ratio_h) == 0));
      }
      CHECK_KERNEL_ERROR
      if (n_receivers_per_shot_h > 0) {
        add_sources<<<dimGrid_receivers, dimBlock_receivers>>>(
            wfc, grad_r + t * n_shots_h * n_receivers_per_shot_h, receivers_i);
        CHECK_KERNEL_ERROR
      }
    } else {
      if (n_sources_per_shot_h > 0) {
        record_receivers<<<dimGrid_sources, dimBlock_sources>>>(
            grad_f + t * n_shots_h * n_sources_per_shot_h, wfc, sources_i);
        CHECK_KERNEL_ERROR
      }
      for (shot_idx = 0; shot_idx < n_shots_h; ++shot_idx) {
      backward_kernel<<<dimGrid, dimBlock>>>(
          v, v2dt2, wfc, wfp, wfc_forward, wfp_forward, wfm_forward, psiz, psiy, psix, psizn, psiyn, psixn, zetaz, zetay, zetax, zetazn, zetayn, zetaxn,
          boundary_front + (t / step_ratio_h) * N_boundary * ny_h * nx_h * n_shots_h,
          boundary_back  + (t / step_ratio_h) * N_boundary * ny_h * nx_h * n_shots_h,
          boundary_top   + (t / step_ratio_h) * N_boundary * nz_h * nx_h * n_shots_h,
          boundary_bottom+ (t / step_ratio_h) * N_boundary * nz_h * nx_h * n_shots_h,
          boundary_left  + (t / step_ratio_h) * N_boundary * nz_h * ny_h * n_shots_h,
          boundary_right + (t / step_ratio_h) * N_boundary * nz_h * ny_h * n_shots_h,
          wfc_last, wfp_last,
          grad_v_shot, az, ay, ax, bz, by, bx, dbzdz, dbydy, dbxdx,
          shot_idx,
          N_boundary,
          t, nt, step_ratio_h,
          v_requires_grad && ((t % step_ratio_h) == 0));
      }
      CHECK_KERNEL_ERROR
      if (n_receivers_per_shot_h > 0) {
        add_sources<<<dimGrid_receivers, dimBlock_receivers>>>(
            wfp, grad_r + t * n_shots_h * n_receivers_per_shot_h, receivers_i);
        CHECK_KERNEL_ERROR
      }
    }
  }
  if (v_requires_grad && n_shots_h > 1) {
    combine_grad_v<<<dimGrid_combine, dimBlock_combine>>>(grad_v, grad_v_shot);
    CHECK_KERNEL_ERROR
  }
  gpuErrchk(cudaFree(wfc_forward));
  gpuErrchk(cudaFree(wfp_forward));
  gpuErrchk(cudaFree(wfm_forward));
}
