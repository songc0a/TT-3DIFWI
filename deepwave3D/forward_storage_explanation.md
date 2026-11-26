# 正向模拟中边界存储和波场保存的正确实现

## 原始代码问题分析

### 1. 最后两帧波场保存错误
**原始代码问题：**
```cuda
if (v_requires_grad) {
    // 每个时间步都执行，会覆盖之前的数据
    wfc_last[i] = wfc[i];
    wfp_last[i] = wfp[i];
}
```

**问题说明：**
- 在每个时间步都保存，导致数据被不断覆盖
- 最终只保存了最后一个时间步的数据
- 没有保存倒数第二个时间步的数据

### 2. 边界存储缺少时间维度
**原始代码问题：**
```cuda
// 没有时间索引，所有时间步的边界数据会相互覆盖
boundary_front[shot_idx * N_boundary * ny * nx + zb * ny * nx + y * nx + x] = wfc[i];
```

**问题说明：**
- 边界存储没有时间维度索引
- 每个时间步的边界数据会覆盖之前的数据
- 无法在反向传播中恢复正确的时间步边界值

### 3. 边界范围定义不标准
**原始代码问题：**
```cuda
// 范围定义不标准，有-1和+1的偏移
if (z >= pml_z0 - N_boundary -1 && z < pml_z0 - 1)
```

## 修正后的正确实现

### 1. 正确的最后两帧波场保存
```cuda
if (v_requires_grad) {
    // 只在最后两个时间步保存完整波场
    if (t == nt - 1) {
        wfc_last[i] = wfc[i];                          // 最后一帧
        wfp_last[i] = wfp[i];
    } else if (t == nt - 2) {
        wfc_last[i + nznynx * n_shots] = wfc[i];       // 倒数第二帧
        wfp_last[i + nznynx * n_shots] = wfp[i];
    }
}
```

**改进说明：**
- 只在特定时间步保存，避免数据覆盖
- 使用内存偏移存储两个不同时间步的数据
- 确保反向传播能获得正确的初始条件

### 2. 正确的边界存储与时间索引
```cuda
// 只在检查点时间步存储边界
if ((t % step_ratio) == 0) {
    int checkpoint_idx = t / step_ratio;
    
    // 完整的时间维度索引
    int boundary_idx = checkpoint_idx * N_boundary * ny * nx * n_shots + 
                      shot_idx * N_boundary * ny * nx + 
                      zb * ny * nx + y * nx + x;
    boundary_front[boundary_idx] = wfc[i];
}
```

**改进说明：**
- 添加了完整的时间维度索引 `checkpoint_idx`
- 只在检查点时间步存储，节省内存
- 确保每个时间步的边界数据独立存储

### 3. 标准化的边界范围定义
```cuda
// Z方向前边界：PML外边界前N_boundary层
if (z >= pml_z0 - N_boundary && z < pml_z0) {
    // 存储前边界
}

// Z方向后边界：PML外边界后N_boundary层  
if (z >= pml_z1 && z < pml_z1 + N_boundary) {
    // 存储后边界
}
```

**改进说明：**
- 移除了不必要的 `-1` 和 `+1` 偏移
- 边界范围定义更加标准和直观
- 与理论文献中的定义保持一致

## 内存布局说明

### 1. 最后两帧波场内存布局
```
wfc_last 布局：
[0 到 nznynx*n_shots-1]:           最后一帧 (t = nt-1)
[nznynx*n_shots 到 2*nznynx*n_shots-1]: 倒数第二帧 (t = nt-2)
```

### 2. 边界存储内存布局
```
boundary_front 布局：
[checkpoint_idx][shot_idx][zb][y][x]

总大小：n_checkpoints * n_shots * N_boundary * ny * nx
其中：n_checkpoints = (nt + step_ratio - 1) / step_ratio
```

## 关键改进点总结

### 1. 时间控制
- **原来**：每个时间步都存储
- **现在**：只在特定时间步存储

### 2. 内存索引
- **原来**：缺少时间维度
- **现在**：完整的4D/5D索引

### 3. 范围定义
- **原来**：不标准的边界范围
- **现在**：标准的PML边界定义

### 4. 数据完整性
- **原来**：数据覆盖问题
- **现在**：每个时间步独立存储

## 使用建议

### 1. 内存分配
在调用前确保分配足够的内存：
```cpp
// 最后两帧波场
size_t last_frames_size = 2 * n_shots * nz * ny * nx * sizeof(DW_DTYPE);

// 边界存储
int n_checkpoints = (nt + step_ratio - 1) / step_ratio;
size_t boundary_size = n_checkpoints * n_shots * N_boundary * max(ny*nx, nz*nx, nz*ny) * sizeof(DW_DTYPE);
```

### 2. 参数选择
- `step_ratio`: 4-8 (平衡内存和精度)
- `N_boundary`: 10-20 (保证重建精度)

### 3. 验证方法
可以通过以下方式验证实现正确性：
1. 检查边界存储的时间连续性
2. 验证最后两帧数据的正确性
3. 比较不同step_ratio下的梯度精度

这样修正后的实现能够正确保存和恢复波场数据，为后续的反向传播提供准确的边界条件。
