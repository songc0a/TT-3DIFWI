"""
边界存储恢复震源波场的Python接口示例
基于改进的CUDA实现
"""

import torch
import numpy as np
from typing import Tuple, Optional

class BoundaryStorageWavePropagator:
    """
    使用边界存储策略的波传播器
    """
    
    def __init__(self, 
                 nz: int, ny: int, nx: int,
                 dz: float, dy: float, dx: float,
                 dt: float,
                 pml_width: int = 20,
                 step_ratio: int = 4,
                 boundary_layers: int = 10):
        """
        初始化波传播器
        
        Args:
            nz, ny, nx: 三维网格尺寸
            dz, dy, dx: 网格间距
            dt: 时间步长
            pml_width: PML层厚度
            step_ratio: 检查点间隔
            boundary_layers: 边界存储层数
        """
        self.nz, self.ny, self.nx = nz, ny, nx
        self.dz, self.dy, self.dx = dz, dy, dx
        self.dt = dt
        self.pml_width = pml_width
        self.step_ratio = step_ratio
        self.boundary_layers = boundary_layers
        
        # 计算PML边界位置
        self.fd_pad = 2  # 有限差分填充
        self.pml_z0 = pml_width + 2 * self.fd_pad
        self.pml_z1 = nz - pml_width - 2 * self.fd_pad
        self.pml_y0 = pml_width + 2 * self.fd_pad  
        self.pml_y1 = ny - pml_width - 2 * self.fd_pad
        self.pml_x0 = pml_width + 2 * self.fd_pad
        self.pml_x1 = nx - pml_width - 2 * self.fd_pad
        
        # 验证CFL条件
        self._check_cfl_condition()
    
    def _check_cfl_condition(self):
        """检查CFL稳定性条件"""
        # 简化的CFL检查，实际应该基于最大速度
        max_dt = 0.5 / np.sqrt(1/self.dx**2 + 1/self.dy**2 + 1/self.dz**2)
        if self.dt > max_dt:
            print(f"警告：时间步长 {self.dt} 可能违反CFL条件，建议 < {max_dt}")
    
    def allocate_boundary_storage(self, nt: int, n_shots: int) -> dict:
        """
        分配边界存储内存
        
        Args:
            nt: 时间步数
            n_shots: 炮数
            
        Returns:
            包含各边界存储张量的字典
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float32
        
        n_checkpoints = nt // self.step_ratio + 1
        
        # 计算边界面大小
        boundary_storage = {}
        
        # Z方向边界 (front/back)
        z_boundary_size = (n_checkpoints, n_shots, self.boundary_layers, self.ny, self.nx)
        boundary_storage['front'] = torch.zeros(z_boundary_size, device=device, dtype=dtype)
        boundary_storage['back'] = torch.zeros(z_boundary_size, device=device, dtype=dtype)
        
        # Y方向边界 (top/bottom)  
        y_boundary_size = (n_checkpoints, n_shots, self.boundary_layers, self.nz, self.nx)
        boundary_storage['top'] = torch.zeros(y_boundary_size, device=device, dtype=dtype)
        boundary_storage['bottom'] = torch.zeros(y_boundary_size, device=device, dtype=dtype)
        
        # X方向边界 (left/right)
        x_boundary_size = (n_checkpoints, n_shots, self.boundary_layers, self.nz, self.ny) 
        boundary_storage['left'] = torch.zeros(x_boundary_size, device=device, dtype=dtype)
        boundary_storage['right'] = torch.zeros(x_boundary_size, device=device, dtype=dtype)
        
        # 检查点存储
        checkpoint_size = (n_checkpoints, n_shots, self.nz, self.ny, self.nx)
        boundary_storage['checkpoints'] = torch.zeros(checkpoint_size, device=device, dtype=dtype)
        
        return boundary_storage
    
    def forward_with_boundary_storage(self,
                                    v: torch.Tensor,
                                    source_amplitudes: torch.Tensor,
                                    source_locations: torch.Tensor,
                                    receiver_locations: Optional[torch.Tensor] = None,
                                    nt: int = None) -> Tuple[torch.Tensor, dict]:
        """
        带边界存储的正演传播
        
        Args:
            v: 速度模型 [nz, ny, nx]
            source_amplitudes: 震源振幅 [nt, n_shots, n_sources_per_shot]
            source_locations: 震源位置 [n_shots, n_sources_per_shot, 3]
            receiver_locations: 接收器位置 [n_shots, n_receivers_per_shot, 3]
            nt: 时间步数
            
        Returns:
            接收器记录和边界存储数据
        """
        
        if nt is None:
            nt = source_amplitudes.shape[0]
        
        n_shots = source_amplitudes.shape[1]
        device = v.device
        dtype = v.dtype
        
        # 分配边界存储
        boundary_storage = self.allocate_boundary_storage(nt, n_shots)
        
        # 初始化波场
        wfc = torch.zeros(n_shots, self.nz, self.ny, self.nx, device=device, dtype=dtype)
        wfp = torch.zeros_like(wfc)
        
        # 初始化PML辅助变量
        psiz = torch.zeros_like(wfc)
        psiy = torch.zeros_like(wfc) 
        psix = torch.zeros_like(wfc)
        zetaz = torch.zeros_like(wfc)
        zetay = torch.zeros_like(wfc)
        zetax = torch.zeros_like(wfc)
        
        # 接收器记录
        if receiver_locations is not None:
            n_receivers = receiver_locations.shape[1]
            receiver_data = torch.zeros(nt, n_shots, n_receivers, device=device, dtype=dtype)
        else:
            receiver_data = torch.empty(0, device=device, dtype=dtype)
        
        # 时间循环（这里简化，实际应调用CUDA核函数）
        for t in range(nt):
            # 调用改进的forward_kernel_improved
            # 这里需要实际的CUDA接口调用
            
            # 示例：存储检查点和边界
            if t % self.step_ratio == 0:
                checkpoint_idx = t // self.step_ratio
                boundary_storage['checkpoints'][checkpoint_idx] = wfc.clone()
                
                # 存储边界数据（简化示例）
                self._store_boundary_data(wfc, boundary_storage, checkpoint_idx)
            
            # 记录接收器数据
            if receiver_locations is not None:
                self._record_receivers(wfc, receiver_locations, receiver_data, t)
        
        return receiver_data, boundary_storage
    
    def _store_boundary_data(self, wfc: torch.Tensor, boundary_storage: dict, checkpoint_idx: int):
        """
        存储边界数据
        """
        # Z方向前边界
        z_start = self.pml_z0 - self.boundary_layers
        z_end = self.pml_z0
        boundary_storage['front'][checkpoint_idx] = wfc[:, z_start:z_end, :, :].clone()
        
        # Z方向后边界
        z_start = self.pml_z1
        z_end = self.pml_z1 + self.boundary_layers
        boundary_storage['back'][checkpoint_idx] = wfc[:, z_start:z_end, :, :].clone()
        
        # Y方向边界
        y_start = self.pml_y0 - self.boundary_layers
        y_end = self.pml_y0
        boundary_storage['top'][checkpoint_idx] = wfc[:, :, y_start:y_end, :].clone()
        
        y_start = self.pml_y1  
        y_end = self.pml_y1 + self.boundary_layers
        boundary_storage['bottom'][checkpoint_idx] = wfc[:, :, y_start:y_end, :].clone()
        
        # X方向边界
        x_start = self.pml_x0 - self.boundary_layers
        x_end = self.pml_x0
        boundary_storage['left'][checkpoint_idx] = wfc[:, :, :, x_start:x_end].clone()
        
        x_start = self.pml_x1
        x_end = self.pml_x1 + self.boundary_layers
        boundary_storage['right'][checkpoint_idx] = wfc[:, :, :, x_start:x_end].clone()
    
    def _record_receivers(self, wfc: torch.Tensor, receiver_locations: torch.Tensor, 
                         receiver_data: torch.Tensor, t: int):
        """
        记录接收器数据（简化实现）
        """
        # 实际应该使用插值，这里简化为最近邻
        for shot_idx in range(receiver_locations.shape[0]):
            for rec_idx in range(receiver_locations.shape[1]):
                z, y, x = receiver_locations[shot_idx, rec_idx].long()
                receiver_data[t, shot_idx, rec_idx] = wfc[shot_idx, z, y, x]
    
    def backward_with_wavefield_reconstruction(self,
                                             v: torch.Tensor,
                                             grad_receiver_data: torch.Tensor,
                                             boundary_storage: dict,
                                             source_amplitudes: torch.Tensor,
                                             source_locations: torch.Tensor,
                                             receiver_locations: torch.Tensor) -> torch.Tensor:
        """
        使用波场重建的反向传播
        
        Args:
            v: 速度模型
            grad_receiver_data: 接收器数据的梯度
            boundary_storage: 边界存储数据
            source_amplitudes: 正演震源
            source_locations: 震源位置
            receiver_locations: 接收器位置
            
        Returns:
            速度模型的梯度
        """
        
        nt = grad_receiver_data.shape[0]
        n_shots = v.shape[0] if v.ndim == 4 else 1
        device = v.device
        dtype = v.dtype
        
        # 初始化梯度
        grad_v = torch.zeros_like(v)
        
        # 初始化反向传播波场
        wfc_back = torch.zeros(n_shots, self.nz, self.ny, self.nx, device=device, dtype=dtype)
        wfp_back = torch.zeros_like(wfc_back)
        
        # 初始化重建的正演波场
        wfc_forward = torch.zeros_like(wfc_back)
        wfp_forward = torch.zeros_like(wfc_back)
        
        # 反向时间循环
        for t in range(nt-1, -1, -1):
            forward_t = nt - 1 - t
            
            # 重建正演波场
            self._reconstruct_forward_wavefield(
                wfc_forward, wfp_forward, boundary_storage, 
                forward_t, source_amplitudes, source_locations
            )
            
            # 反向传播
            # 这里应该调用improved_backward_kernel
            
            # 计算梯度（成像条件）
            if forward_t % self.step_ratio == 0:
                grad_v += 2 * v * self.dt**2 * wfc_forward * wfc_back * self.step_ratio
            
            # 注入接收器数据梯度
            if receiver_locations is not None:
                self._inject_receiver_gradients(
                    wfc_back, grad_receiver_data, receiver_locations, t
                )
        
        return grad_v
    
    def _reconstruct_forward_wavefield(self,
                                     wfc_forward: torch.Tensor,
                                     wfp_forward: torch.Tensor, 
                                     boundary_storage: dict,
                                     forward_t: int,
                                     source_amplitudes: torch.Tensor,
                                     source_locations: torch.Tensor):
        """
        重建正演波场
        """
        # 如果是检查点时间步，直接恢复
        if forward_t % self.step_ratio == 0:
            checkpoint_idx = forward_t // self.step_ratio
            wfc_forward.copy_(boundary_storage['checkpoints'][checkpoint_idx])
        else:
            # 从边界和内部重建
            self._reconstruct_from_boundary(wfc_forward, boundary_storage, forward_t)
        
        # 添加震源项
        self._add_source_terms(wfc_forward, source_amplitudes, source_locations, forward_t)
    
    def _reconstruct_from_boundary(self,
                                  wfc_forward: torch.Tensor,
                                  boundary_storage: dict,
                                  forward_t: int):
        """
        从边界数据重建波场
        """
        checkpoint_idx = forward_t // self.step_ratio
        
        # 恢复边界数据
        # Z方向边界
        z_start = self.pml_z0 - self.boundary_layers
        z_end = self.pml_z0
        wfc_forward[:, z_start:z_end, :, :] = boundary_storage['front'][checkpoint_idx]
        
        z_start = self.pml_z1
        z_end = self.pml_z1 + self.boundary_layers  
        wfc_forward[:, z_start:z_end, :, :] = boundary_storage['back'][checkpoint_idx]
        
        # Y方向边界
        y_start = self.pml_y0 - self.boundary_layers
        y_end = self.pml_y0
        wfc_forward[:, :, y_start:y_end, :] = boundary_storage['top'][checkpoint_idx]
        
        y_start = self.pml_y1
        y_end = self.pml_y1 + self.boundary_layers
        wfc_forward[:, :, y_start:y_end, :] = boundary_storage['bottom'][checkpoint_idx]
        
        # X方向边界  
        x_start = self.pml_x0 - self.boundary_layers
        x_end = self.pml_x0
        wfc_forward[:, :, :, x_start:x_end] = boundary_storage['left'][checkpoint_idx]
        
        x_start = self.pml_x1
        x_end = self.pml_x1 + self.boundary_layers
        wfc_forward[:, :, :, x_start:x_end] = boundary_storage['right'][checkpoint_idx]
    
    def _add_source_terms(self,
                         wfc_forward: torch.Tensor,
                         source_amplitudes: torch.Tensor,
                         source_locations: torch.Tensor,
                         t: int):
        """
        添加震源项
        """
        for shot_idx in range(source_locations.shape[0]):
            for src_idx in range(source_locations.shape[1]):
                z, y, x = source_locations[shot_idx, src_idx].long()
                wfc_forward[shot_idx, z, y, x] += source_amplitudes[t, shot_idx, src_idx]
    
    def _inject_receiver_gradients(self,
                                  wfc_back: torch.Tensor,
                                  grad_receiver_data: torch.Tensor,
                                  receiver_locations: torch.Tensor,
                                  t: int):
        """
        注入接收器梯度
        """
        for shot_idx in range(receiver_locations.shape[0]):
            for rec_idx in range(receiver_locations.shape[1]):
                z, y, x = receiver_locations[shot_idx, rec_idx].long()
                wfc_back[shot_idx, z, y, x] += grad_receiver_data[t, shot_idx, rec_idx]

    def estimate_memory_usage(self, nt: int, n_shots: int) -> dict:
        """
        估算内存使用量
        """
        element_size = 4  # float32
        
        # 波场内存
        wavefield_size = n_shots * self.nz * self.ny * self.nx * element_size
        
        # 检查点内存
        n_checkpoints = nt // self.step_ratio + 1
        checkpoint_memory = n_checkpoints * wavefield_size
        
        # 边界存储内存
        z_boundary_mem = 2 * n_checkpoints * n_shots * self.boundary_layers * self.ny * self.nx * element_size
        y_boundary_mem = 2 * n_checkpoints * n_shots * self.boundary_layers * self.nz * self.nx * element_size  
        x_boundary_mem = 2 * n_checkpoints * n_shots * self.boundary_layers * self.nz * self.ny * element_size
        
        boundary_memory = z_boundary_mem + y_boundary_mem + x_boundary_mem
        
        total_memory = checkpoint_memory + boundary_memory + 2 * wavefield_size  # 2个当前波场
        
        return {
            'total_memory_mb': total_memory / (1024**2),
            'checkpoint_memory_mb': checkpoint_memory / (1024**2),
            'boundary_memory_mb': boundary_memory / (1024**2),
            'wavefield_memory_mb': wavefield_size / (1024**2),
            'memory_reduction_factor': (nt * wavefield_size) / total_memory
        }


# 使用示例
def example_usage():
    """
    使用示例
    """
    # 创建传播器
    propagator = BoundaryStorageWavePropagator(
        nz=100, ny=100, nx=100,
        dz=10.0, dy=10.0, dx=10.0, 
        dt=0.001,
        step_ratio=4,
        boundary_layers=10
    )
    
    # 创建速度模型
    v = torch.ones(100, 100, 100, device='cuda') * 2000  # 2000 m/s
    
    # 创建震源
    nt = 1000
    n_shots = 4
    n_sources_per_shot = 1
    source_amplitudes = torch.randn(nt, n_shots, n_sources_per_shot, device='cuda')
    source_locations = torch.tensor([[[50, 50, 10]]] * n_shots, device='cuda')
    
    # 创建接收器
    n_receivers_per_shot = 10
    receiver_locations = torch.randint(20, 80, (n_shots, n_receivers_per_shot, 3), device='cuda')
    
    # 估算内存使用
    memory_info = propagator.estimate_memory_usage(nt, n_shots)
    print("内存使用估算:")
    for key, value in memory_info.items():
        print(f"  {key}: {value:.2f}")
    
    # 正演传播
    receiver_data, boundary_storage = propagator.forward_with_boundary_storage(
        v, source_amplitudes, source_locations, receiver_locations, nt
    )
    
    print(f"接收器数据形状: {receiver_data.shape}")
    print(f"边界存储数据类型: {list(boundary_storage.keys())}")
    
    # 反向传播（模拟）
    grad_receiver_data = torch.randn_like(receiver_data)
    grad_v = propagator.backward_with_wavefield_reconstruction(
        v, grad_receiver_data, boundary_storage,
        source_amplitudes, source_locations, receiver_locations
    )
    
    print(f"速度梯度形状: {grad_v.shape}")

if __name__ == "__main__":
    example_usage()
