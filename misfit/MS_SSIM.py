from .base import Misfit
import torch
import torch.nn.functional as F



class Misfit_M_SSIM3D(Misfit):
    def __init__(self, gaussian_sigmas=[0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
                 data_range = 1.0,
                 K=(0.01, 0.03),
                 alpha=1,
                 compensation=1.0):
        super(Misfit_M_SSIM3D, self).__init__()
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.alpha = alpha
        self.compensation=compensation
        self.pad = int(2 * gaussian_sigmas[-1])
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        
        
        # Create the Gaussian masks for convolution
        g_masks = torch.zeros((len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        
        self.g_masks = g_masks

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel"""
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel"""
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)
    def mean_filter(self,x, window_size=3):
    	kernel = torch.ones(1, 1, window_size, window_size) / (window_size ** 2)
    	kernel = kernel.to(x.device)
    	return F.conv2d(x, kernel, padding=window_size//2)

    def forward(self, obs, syn, chunk_size=8):
        ns, nc, nt, nr = obs.shape
    
        max_val = torch.max(torch.abs(syn), dim=3, keepdim=True).values
        max_val = torch.max(max_val[..., 0:], dim=2, keepdim=True).values
        syn = syn / (1e-20 + max_val)
    
        max_val = torch.max(torch.abs(obs), dim=3, keepdim=True).values
        max_val = torch.max(max_val[..., 0:], dim=2, keepdim=True).values
        obs = obs / (1e-20 + max_val)
    
        obs = self.alpha * obs
        syn = self.alpha * syn
    
        with torch.no_grad():
            base_g_mask = self.g_masks.to(syn.device)
    
        loss_list = []
    
        for c in range(0, nc, chunk_size):
            c_end = min(c + chunk_size, nc)
            obs_c = obs[:, c:c_end, :, :]
            syn_c = syn[:, c:c_end, :, :]
    
            g_mask = base_g_mask.repeat(c_end - c, 1, 1, 1)
    
            mux = F.conv2d(obs_c, g_mask, groups=c_end - c, padding=self.pad)
            muy = F.conv2d(syn_c, g_mask, groups=c_end - c, padding=self.pad)
    
            mux2 = mux * mux
            muy2 = muy * muy
            muxy = mux * muy
    
            sigmax2 = F.conv2d(obs_c * obs_c, g_mask, groups=c_end - c, padding=self.pad) - mux2
            sigmay2 = F.conv2d(syn_c * syn_c, g_mask, groups=c_end - c, padding=self.pad) - muy2
            sigmaxy = F.conv2d(obs_c * syn_c, g_mask, groups=c_end - c, padding=self.pad) - muxy
    
            loss_ms_ssim_c = torch.mean(
                -(((2 * muxy + self.C1) / (mux2 + muy2 + self.C1)) *
                  ((2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)))
            )
            loss_list.append(loss_ms_ssim_c)
    
            del obs_c, syn_c, g_mask, mux, muy, mux2, muy2, muxy, sigmax2, sigmay2, sigmaxy
    
        loss_mix = torch.stack(loss_list).mean()
        loss_mix = self.compensation * loss_mix
    
        return loss_mix

