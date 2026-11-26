'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2024-05-12 09:09:07
* LastEditors: LiuFeng
* LastEditTime: 2024-05-12 09:09:14
* Description: 
* Copyright (c) 2024 by liufeng, Email: liufeng2317@mail.ustc.edu.cn, All Rights Reserved.
'''
from .base import Misfit
import torch
import math
from .GlobalCorrelation import Misfit_global_correlation,Misfit_local1_correlation
from .Envelope import Misfit_envelope
from .L2 import Misfit_waveform_L2
from .Wasserstein_1 import Misfit_Wasserstein1

class Misfit_weighted_ECI(Misfit):
    """Weighted envelope misfit and global correlation misfit (Song Chao et al., 2023, IEEE TGRS)
    
    Paraemters:
    -----------
        dt (float)      : time sampling interval
        p (float)       : the norm order of the reuslt
        instaneous_phase (bool) : use instaeous phase or amplitude for the misfit
        obs (Tensors)   : the observed waveform 
        syn (Tensors)   : the synthetic waveform 
    """
    def __init__(self,max_iter=1000,dt=1,p=1.5,instaneous_phase=False) -> None:
        super().__init__()
        self.max_iter = max_iter
        self.iter = 0
        self.dt = dt
        self.p  = p
        self.instaneous_phase   = instaneous_phase
        self.GC_fn              = Misfit_global_correlation(dt=self.dt)
        self.L2_fn              = Misfit_local1_correlation()
        #self.L2_fn              = Misfit_waveform_L2()
        self.Envelope_fn        = Misfit_envelope(dt=self.dt,p=self.p,number=3,instaneous_phase=self.instaneous_phase)
        #self.L2_fn              = Misfit_Wasserstein1(trans_type='linear',dt=1,theta=1,p=1)
        
    def forward(self,obs,syn):
        N = self.max_iter
        
        initial_value = 3
        final_value = 1
        decay_rate = 1 / (1 + torch.exp(torch.tensor(-10/N * (self.iter + 1))))
        values = initial_value - (initial_value - final_value) * decay_rate
        self.Envelope_fn        = Misfit_envelope(dt=self.dt,p=values,number=1,instaneous_phase=self.instaneous_phase)
        # self.Envelope_fn        = Misfit_Wasserstein1(trans_type='linear',dt=1,theta=1,p=1)
        
        w_i = 1/(1+math.exp(-1*(self.iter - N//3)))
        GCN_loss = self.GC_fn.forward(obs=obs,syn=syn)
        # L2_loss = self.L2_fn.forward(obs=obs,syn=syn)
        ECI_loss = self.Envelope_fn.forward(obs=obs,syn=syn)
        loss = w_i*GCN_loss + (1-w_i)*ECI_loss
        self.iter += 1
        return loss
