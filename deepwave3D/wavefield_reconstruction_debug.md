# 震源波场重构问题诊断与修正

## 原始问题分析

您的代码中存在以下关键问题导致无法正常重构震源波场：

### 1. 边界范围不一致错误

**正向存储（正确）：**
```cuda
// 标准边界定义
if (z >= pml_z0 - N_boundary && z < pml_z0)
```

**原始反向恢复（错误）：**
```cuda
// 有不必要的-1偏移，导致范围不匹配
if (z >= pml_z0 - N_boundary -1 && z < pml_z0 - 1)
```

**后果：** 正向存储和反向恢复的网格点不对应，导致边界数据无法正确恢复。

### 2. 缺少时间索引

**原始反向恢复（错误）：**
```cuda
// 没有时间维度索引
boundary_front[shot_idx * N_boundary * ny * nx + ...]
```

**修正后（正确）：**
```cuda
// 包含完整的时间索引
int checkpoint_idx = forward_t / step_ratio_h;
int boundary_idx = checkpoint_idx * N_boundary * ny * nx * n_shots + 
                   shot_idx * N_boundary * ny * nx + ...;
boundary_front[boundary_idx]
```

### 3. 波场重建逻辑错误

**原始问题：**
- 重建顺序混乱：先计算时间步进，后恢复边界
- 缺少对最后两帧的正确处理
- 内部区域重建方法不完整

## 修正后的重建流程

### 1. 正确的时间映射
```cuda
int forward_t = nt - 1 - t;  // 反向时间步对应的正演时间步
```

### 2. 分层重建策略

#### 第一层：最后两帧处理
```cuda
if (t == nt - 1) {
    // 最后一帧：直接从缓存恢复
    wfc_forward[i] = wfc_last[i];
    wfp_forward[i] = wfp_last[i];
} else if (t == nt - 2) {
    // 倒数第二帧：从缓存的第二部分恢复
    wfc_forward[i] = wfc_last[i + nznynx * n_shots];
    wfp_forward[i] = wfp_last[i + nznynx * n_shots];
}
```

#### 第二层：边界恢复
```cuda
// 使用正确的时间索引和边界范围
int checkpoint_idx = forward_t / step_ratio_h;
int boundary_idx = checkpoint_idx * N_boundary * surface_size * n_shots + ...;
```

#### 第三层：内部重建
```cuda
// 对于非边界点，使用有限差分重建
if (!is_boundary_restored && !pml_z && !pml_y && !pml_x) {
    DW_DTYPE w_sum = DIFFZ2(WFC_FWD) + DIFFY2(WFC_FWD) + DIFFX2(WFC_FWD);
    DW_DTYPE new_wfp = v2dt2[j] * w_sum + 2 * wfc_forward[i] - wfp_forward[i];
}
```

### 3. 正确的梯度计算
```cuda
// 只在检查点时间步计算梯度
if ((forward_t % step_ratio_h) == 0) {
    grad_v[i] += 2 * v[j] * dt2 * wfc_forward[i] * wfc[i] * step_ratio_h;
}
```

## 关键修正点

### 1. 边界范围统一
- 正向存储和反向恢复使用完全相同的边界范围定义
- 移除所有不必要的±1偏移

### 2. 完整时间索引
- 边界恢复包含完整的4D索引：[time][shot][boundary_layer][spatial]
- 确保时间映射正确：`forward_t = nt - 1 - t`

### 3. 逻辑顺序优化
- 先处理特殊情况（最后两帧）
- 再处理边界恢复
- 最后处理内部重建

### 4. 内存访问优化
- 使用正确的内存布局和索引计算
- 避免越界访问

## 验证方法

### 1. 边界一致性检查
```cuda
// 在正向存储时添加验证
if (boundary_idx_computed == expected_boundary_idx) {
    // 存储成功
}
```

### 2. 重建精度验证
```cuda
// 比较重建波场与参考波场的差异
DW_DTYPE error = abs(wfc_reconstructed[i] - wfc_reference[i]);
if (error > tolerance) {
    // 报告重建错误
}
```

### 3. 梯度收敛测试
- 使用不同的`step_ratio`参数
- 验证梯度计算的收敛性
- 与完整存储方法对比

## 实际应用建议

### 1. 参数调优
- `step_ratio = 4`: 平衡内存和精度
- `N_boundary = 10-20`: 确保边界重建精度
- 根据实际模型大小调整

### 2. 内存预分配
```cpp
// 预先计算所需内存大小
int n_checkpoints = (nt + step_ratio - 1) / step_ratio;
size_t boundary_memory = n_checkpoints * n_shots * N_boundary * max_surface_size;
```

### 3. 调试工具
- 添加边界数据验证
- 输出中间重建结果
- 比较不同重建方法的精度

## 预期改进效果

修正后的实现应该能够：

1. **正确重建边界波场**：边界范围和时间索引完全匹配
2. **准确计算梯度**：成像条件正确应用
3. **稳定数值表现**：避免重建过程中的数值发散
4. **高效内存使用**：合理的边界存储策略

通过这些修正，您的边界存储恢复震源波场实现应该能够正常工作并产生准确的梯度结果。
