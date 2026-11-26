// 添加到 scalar_boundary_improved.cu 中的验证函数

// 验证边界存储和恢复的正确性
__global__ void validate_boundary_reconstruction(
    DW_DTYPE const *__restrict const wfc_original,
    DW_DTYPE const *__restrict const wfc_reconstructed,
    DW_DTYPE *__restrict const error_map,
    int64_t shot_idx,
    int64_t const N_boundary) {
    
    int64_t x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
    int64_t y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
    int64_t z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;

    if (z < nz - FD_PAD && y < ny - FD_PAD && x < nx - FD_PAD) {
        int64_t j = z * nynx + y * nx + x;
        int64_t i = shot_idx * nznynx + j;
        
        // 计算重建误差
        DW_DTYPE error = abs(wfc_original[i] - wfc_reconstructed[i]);
        error_map[i] = error;
        
        // 在边界区域进行特别检查
        bool is_boundary = false;
        
        // 检查是否在边界区域
        if ((z >= pml_z0 - N_boundary && z < pml_z0) ||           // Z前边界
            (z >= pml_z1 && z < pml_z1 + N_boundary) ||           // Z后边界
            (y >= pml_y0 - N_boundary && y < pml_y0) ||           // Y前边界
            (y >= pml_y1 && y < pml_y1 + N_boundary) ||           // Y后边界
            (x >= pml_x0 - N_boundary && x < pml_x0) ||           // X前边界
            (x >= pml_x1 && x < pml_x1 + N_boundary)) {           // X后边界
            is_boundary = true;
        }
        
        // 边界区域的误差应该很小（理论上为0）
        if (is_boundary && error > 1e-6) {
            printf("Border reconstruction error at (%ld,%ld,%ld): %f\n", z, y, x, error);
        }
    }
}

// 打印重建统计信息
__global__ void compute_reconstruction_stats(
    DW_DTYPE const *__restrict const error_map,
    DW_DTYPE *__restrict const max_error,
    DW_DTYPE *__restrict const avg_error,
    int64_t shot_idx) {
    
    int64_t x = blockIdx.x * blockDim.x + threadIdx.x + FD_PAD;
    int64_t y = blockIdx.y * blockDim.y + threadIdx.y + FD_PAD;
    int64_t z = blockIdx.z * blockDim.z + threadIdx.z + FD_PAD;

    if (z < nz - FD_PAD && y < ny - FD_PAD && x < nx - FD_PAD) {
        int64_t j = z * nynx + y * nx + x;
        int64_t i = shot_idx * nznynx + j;
        
        DW_DTYPE error = error_map[i];
        
        // 使用原子操作更新最大误差
        atomicMax(max_error, error);
        
        // 累加误差（需要后续除以总点数）
        atomicAdd(avg_error, error);
    }
}

// 辅助函数：验证内存索引正确性
__device__ bool validate_boundary_index(
    int64_t z, int64_t y, int64_t x,
    int64_t checkpoint_idx, int64_t shot_idx,
    int64_t N_boundary, int64_t expected_idx) {
    
    bool is_valid = false;
    
    // Z方向前边界
    if (z >= pml_z0 - N_boundary && z < pml_z0) {
        int zb = z - (pml_z0 - N_boundary);
        int64_t computed_idx = checkpoint_idx * N_boundary * ny * nx * n_shots + 
                              shot_idx * N_boundary * ny * nx + 
                              zb * ny * nx + y * nx + x;
        is_valid = (computed_idx == expected_idx);
    }
    
    // 类似地验证其他方向...
    
    return is_valid;
}

// CPU端调用的验证函数
extern "C" void validate_reconstruction_accuracy(
    DW_DTYPE const *wfc_original,
    DW_DTYPE const *wfc_reconstructed,
    int64_t n_shots, int64_t nz, int64_t ny, int64_t nx,
    int64_t N_boundary) {
    
    // 分配误差映射
    size_t total_size = n_shots * nz * ny * nx;
    DW_DTYPE *error_map;
    DW_DTYPE *max_error, *avg_error;
    
    cudaMalloc(&error_map, total_size * sizeof(DW_DTYPE));
    cudaMalloc(&max_error, sizeof(DW_DTYPE));
    cudaMalloc(&avg_error, sizeof(DW_DTYPE));
    
    // 初始化
    cudaMemset(max_error, 0, sizeof(DW_DTYPE));
    cudaMemset(avg_error, 0, sizeof(DW_DTYPE));
    
    dim3 dimBlock(8, 8, 8);
    unsigned int gridx = (nx - 2 * FD_PAD + dimBlock.x - 1) / dimBlock.x;
    unsigned int gridy = (ny - 2 * FD_PAD + dimBlock.y - 1) / dimBlock.y;
    unsigned int gridz = (nz - 2 * FD_PAD + dimBlock.z - 1) / dimBlock.z;
    dim3 dimGrid(gridx, gridy, gridz);
    
    for (int shot_idx = 0; shot_idx < n_shots; shot_idx++) {
        // 计算误差
        validate_boundary_reconstruction<<<dimGrid, dimBlock>>>(
            wfc_original, wfc_reconstructed, error_map, shot_idx, N_boundary);
        
        // 计算统计信息
        compute_reconstruction_stats<<<dimGrid, dimBlock>>>(
            error_map, max_error, avg_error, shot_idx);
    }
    
    // 获取结果
    DW_DTYPE h_max_error, h_avg_error;
    cudaMemcpy(&h_max_error, max_error, sizeof(DW_DTYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_avg_error, avg_error, sizeof(DW_DTYPE), cudaMemcpyDeviceToHost);
    
    // 计算平均误差
    h_avg_error /= (total_size - 2 * FD_PAD * (nx * ny + ny * nz + nz * nx));
    
    printf("Reconstruction Validation Results:\n");
    printf("  Maximum Error: %e\n", h_max_error);
    printf("  Average Error: %e\n", h_avg_error);
    
    if (h_max_error < 1e-6) {
        printf("  Status: PASS - Reconstruction is accurate\n");
    } else {
        printf("  Status: FAIL - Reconstruction has significant errors\n");
    }
    
    // 清理
    cudaFree(error_map);
    cudaFree(max_error);
    cudaFree(avg_error);
}

// 调试用：打印特定区域的波场值
__global__ void debug_print_wavefield_region(
    DW_DTYPE const *__restrict const wfc,
    int64_t shot_idx, int64_t z_start, int64_t z_end,
    int64_t y_start, int64_t y_end, int64_t x_start, int64_t x_end,
    const char* label) {
    
    int64_t x = blockIdx.x * blockDim.x + threadIdx.x + x_start;
    int64_t y = blockIdx.y * blockDim.y + threadIdx.y + y_start;
    int64_t z = blockIdx.z * blockDim.z + threadIdx.z + z_start;
    
    if (z < z_end && y < y_end && x < x_end) {
        int64_t j = z * nynx + y * nx + x;
        int64_t i = shot_idx * nznynx + j;
        
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            printf("%s: wfc[%ld,%ld,%ld] = %f\n", label, z, y, x, wfc[i]);
        }
    }
}
