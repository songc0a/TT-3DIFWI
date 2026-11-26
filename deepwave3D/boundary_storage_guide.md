# 边界存储恢复震源波场的正确实现方法

## 背景问题

在全波形反演（FWI）中，需要计算目标函数对速度模型的梯度。这通常通过反向传播实现，需要正演波场和反向传播波场的交叉相关。由于存储所有时间步的完整波场会消耗大量内存，因此采用检查点策略结合边界存储来重建正演波场。

## 核心策略

### 1. 检查点策略 (Checkpointing)
- 每隔 `step_ratio` 时间步保存一次完整的波场
- 减少内存使用，但需要重新计算中间时间步

### 2. 边界存储策略 (Boundary Storage)
- 在PML边界附近存储 `N_boundary` 层的波场值
- 用于反向传播时重建边界条件

### 3. 震源波场重建 (Source Wavefield Reconstruction)
- 结合检查点和边界存储重建完整的正演波场
- 用于梯度计算中的成像条件

## 实现细节

### 正演阶段 (Forward Propagation)

```cuda
// 在 forward_kernel 中的关键存储逻辑
if (v_requires_grad) {
    // 1. 检查点存储：每隔 step_ratio 时间步存储完整波场
    if ((t % step_ratio) == 0) {
        int checkpoint_idx = (t / step_ratio) * nznynx * n_shots + i;
        wfc_checkpoints[checkpoint_idx] = wfc[i];
    }
    
    // 2. 边界存储：只在检查点时间步存储边界区域
    if ((t % step_ratio) == 0) {
        // 存储六个方向的边界
        // Z前后、Y前后、X前后边界
        store_boundary_values(t, i, wfc[i]);
    }
}
```

### 反向传播阶段 (Backward Propagation)

```cuda
// 在 backward_kernel 中的重建逻辑
if (v_requires_grad) {
    int forward_t = nt - 1 - t;  // 对应的正演时间步
    
    // 1. 检查点时间步：直接从存储中恢复
    if ((forward_t % step_ratio_h) == 0) {
        int checkpoint_idx = (forward_t / step_ratio_h) * nznynx * n_shots + i;
        wfc_forward[i] = wfc_checkpoints[checkpoint_idx];
    } else {
        // 2. 非检查点时间步：重建波场
        reconstruct_wavefield_from_boundary(forward_t, i);
    }
    
    // 3. 计算梯度（成像条件）
    grad_v[i] += 2 * v[j] * dt2 * wfc_forward[i] * wfc[i] * step_ratio_h;
}
```

## 关键改进点

### 1. 边界存储时机修正
**原问题**：每个时间步都存储边界，浪费内存
**改进**：只在检查点时间步存储边界

```cuda
// 错误的实现
if (v_requires_grad) {
    // 每个时间步都存储 - 浪费内存
    store_boundary(t, wfc[i]);
}

// 正确的实现  
if (v_requires_grad && (t % step_ratio) == 0) {
    // 只在检查点时间步存储
    store_boundary(t, wfc[i]);
}
```

### 2. 边界区域定义修正
**原问题**：边界范围定义不准确
**改进**：基于PML区域正确定义边界范围

```cuda
// 正确的边界定义
// Z方向前边界：PML外边界前 N_boundary 层
if (z >= pml_z0 - N_boundary && z < pml_z0) {
    // 存储前边界
}

// Z方向后边界：PML外边界后 N_boundary 层  
if (z >= pml_z1 && z < pml_z1 + N_boundary) {
    // 存储后边界
}
```

### 3. 震源项处理
在重建过程中需要正确添加震源项：

```cuda
// 重建时需要考虑震源贡献
if (is_source_point(i, sources_i, forward_t)) {
    wfc_forward[i] += source_f[get_source_index(forward_t, shot_idx)];
}
```

### 4. 内存索引优化
改进边界存储的内存索引计算：

```cuda
// 优化的索引计算
int boundary_idx = boundary_time_idx * N_boundary * spatial_size * n_shots + 
                   shot_idx * N_boundary * spatial_size + 
                   local_boundary_idx;
```

## 内存需求分析

### 原始方法（存储所有波场）
- 内存需求：`nt * nz * ny * nx * n_shots * sizeof(DW_DTYPE)`
- 对于大规模3D问题，内存需求过大

### 检查点+边界存储方法
- 检查点内存：`(nt/step_ratio) * nz * ny * nx * n_shots * sizeof(DW_DTYPE)`
- 边界存储内存：`(nt/step_ratio) * N_boundary * surface_size * n_shots * sizeof(DW_DTYPE)`
- 总内存减少约 `step_ratio` 倍

## 数值精度考虑

### 1. 重建精度
- 边界重建可能引入数值误差
- 需要足够的边界层数（推荐 N_boundary >= 10）

### 2. 时间步长稳定性
- 重建过程中需要保持数值稳定性
- CFL条件仍需满足

### 3. PML边界处理
- PML区域的特殊处理需要额外注意
- 确保PML辅助变量的正确重建

## 实际应用建议

### 1. 参数选择
- `step_ratio`: 4-8（平衡内存和计算）
- `N_boundary`: 10-20（保证重建精度）

### 2. 内存优化
- 使用GPU统一内存管理
- 预分配所有需要的内存缓冲区

### 3. 调试验证
- 比较完整存储和边界重建的结果
- 验证梯度计算的准确性

## 总结

正确实现边界存储恢复震源波场需要：

1. **合理的存储策略**：检查点+边界存储相结合
2. **准确的边界定义**：基于PML区域定义边界范围  
3. **正确的重建逻辑**：时间反向的波场重建
4. **完整的震源处理**：重建时正确添加震源项
5. **数值稳定性**：保证重建过程的数值精度

这种方法能够在大幅减少内存使用的同时，保持梯度计算的准确性，是大规模3D全波形反演的关键技术。
