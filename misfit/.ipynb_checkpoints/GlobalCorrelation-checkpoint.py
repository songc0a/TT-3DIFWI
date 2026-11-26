from .base import Misfit
import torch
import torch.nn.functional as F

class Misfit_global_correlation(Misfit):
    """ global correlation misfit functions
    
    Paraemters:
    ------------
        obs (Tensors)   : the observed waveform 
        syn (Tensors)   : the synthetic waveform 
    """
    def __init__(self,dt=1) -> None:
        super().__init__()
        self.dt = dt
    
    # One loop for calculate the misfits (traces)
    def forward(self, obs, syn):
        """Compute the global correlation misfit between observed and synthetic waveforms.
        
        Args:
            obs (Tensor): Observed waveform, shape (batch, channels, traces).
            syn (Tensor): Synthetic waveform, shape (batch, channels, traces).
        
        Returns:
            Tensor: Correlation-based misfit loss.
        """
        # Compute norms
        obs_norm = obs.norm(dim=1, keepdim=True)  # Shape: (N, 1, M)
        syn_norm = syn.norm(dim=1, keepdim=True)  # Shape: (N, 1, M)

        # Normalize the observed and synthetic waveforms
        obs_normalized = obs / obs_norm
        syn_normalized = syn / syn_norm

        # Initialize result tensor
        rsd = torch.empty(obs.shape[0], obs.shape[2], device=obs.device)

        # Compute correlation for each trace
        for itrace in range(obs.shape[2]):
            obs_trace = obs_normalized[:, :, itrace]  # Shape: (N, T)
            syn_trace = syn_normalized[:, :, itrace]  # Shape: (N, T)

            # Calculate covariance and variances
            cov = torch.mean(obs_trace * syn_trace, dim=1)  # Shape: (N,)
            var_obs = torch.var(obs_trace, dim=1)  # Shape: (N,)
            var_syn = torch.var(syn_trace, dim=1)  # Shape: (N,)

            # Avoid division by zero by masking
            corr = cov / (torch.sqrt(var_obs * var_syn) + 1e-8)  # Adding small value to avoid div by zero

            # Handle the case where both variances are zero
            corr[torch.isnan(corr)] = 0  # If both variances are zero, set correlation to zero

            rsd[:, itrace] = -corr
        
        loss = torch.sum(rsd * self.dt)
        return loss / obs.shape[2]
        
        
        

def hilbert_transform(data_in, p=1):
    ns, nt, nr = data_in.shape
    transforms = torch.fft.fftn(data_in,dim=1)
    #print(transforms.shape)
    transforms[:,1:nt//2,:]      *= 2.0
    transforms[:,nt//2 + 1: nt,:]  = 0+0j
    transforms[:,0,:] = 0;
    data_out = torch.abs(torch.fft.ifftn(transforms,dim=1))**p
    return data_out
    
def transform(f, g, trans_type, theta):
    """
        do the transform for the signal f and g
        Args:
            f, g: seismic data with the shape [num_time_steps,num_shots*num_receivers_per_shot]
            trans_type: type for transform
            theta:the scalar variable for transform            
        return:
            output with transfomation
    """ 
    c = 0.0 
    device = f.device
    if trans_type == 'linear':
        min_value = torch.min(f.detach().min(), g.detach().min())
        mu, nu = f, g
        c = -min_value if min_value<0 else 0
        c = c * theta
        d = torch.ones(f.shape).to(device)
    elif trans_type == 'abs':
        mu, nu = torch.abs(f), torch.abs(g)
        d = torch.sign(f).to(device)
    elif trans_type == 'square':
        mu = f * f
        nu = g * g
        d = 2 * f
    elif trans_type == 'exp':
        mu = torch.exp(theta*f)
        nu = torch.exp(theta*g)
        d = theta * mu
    elif trans_type == 'softplus':
        mu = torch.log(torch.exp(theta*f) + 1.01)
        nu = torch.log(torch.exp(theta*g) + 1.01)
        d = theta / torch.exp(-theta*f)
    elif trans_type == 'envelope':
    	mu = hilbert_transform(f, p=theta)
    	nu = hilbert_transform(g, p=theta)
    	d = torch.ones(f.shape).to(device)
    else:
        mu, nu = f, g
        d = torch.ones(f.shape).to(device)
    mu = mu + c + 1e-18 
    nu = nu + c + 1e-18
    return mu, nu, d

def trace_sum_normalize(x,time_dim=1):
    """
        normalization with the summation of each trace
        note that the channel should be 1
    """
    x = x / (x.sum(dim=time_dim,keepdim=True)+1e-18)
    return x
        

    
from .base import Misfit 
from .L2 import Misfit_waveform_L2      
class Misfit_local1_correlation(Misfit):
    def __init__(self, gaussian_sigmas=[0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
                 data_range = 1.0,
                 K=(0.01, 0.03),
                 alpha=1,
                 compensation=200.0,
                 cuda_dev=0,
                 ct=0,
                 p=1,
                 type="mean"):
        super(Misfit_local1_correlation, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(4 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.ct = ct
        self.compensation=compensation
        self.L2_fn = Misfit_waveform_L2()
        self.type = type
        self.p = p
        filter_size = int(8 * gaussian_sigmas[-1] + 1)
        
        # Create the Gaussian masks for convolution
        g_masks = torch.zeros((len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        
        #self.g_masks = g_masks.cuda(cuda_dev)
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

    def forward(self, obs, syn):

        # syn = syn/(1e-10 + torch.max(torch.abs(syn),axis=1,keepdim=True).values)
        # obs = obs/(1e-10 + torch.max(torch.abs(obs),axis=1,keepdim=True).values)
        # obs, syn, dd = transform(obs, syn, 'envelope', theta=1)

        obs = obs.unsqueeze(1)  # (B, 1, T, R)
        syn = syn.unsqueeze(1)  # (B, 1, T, R)
        
        # syn = F.avg_pool2d(syn, kernel_size=(4,1), stride=(4,1))
        # obs = F.avg_pool2d(obs, kernel_size=(4,1), stride=(4,1))

        
        B, C, T, R = syn.shape  # 获取批次大小、通道数、时间采样点和接收器数
        self.g_masks = self.g_masks.to(syn.device)
        
        # syn = self.mean_filter(syn, window_size=3)
        # obs = self.mean_filter(obs, window_size=3)
        

        # Apply 2D convolution with Gaussian kernels to get the necessary statistics
        mux = F.conv2d(obs, self.g_masks, groups=1, padding=self.pad)
        muy = F.conv2d(syn, self.g_masks, groups=1, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        # sigmax2 = torch.clamp(F.conv2d(obs * obs, self.g_masks, groups=1, padding=self.pad) - mux2, min = 0)
        # sigmay2 = torch.clamp(F.conv2d(syn * syn, self.g_masks, groups=1, padding=self.pad) - muy2, min = 0)
        # sigmaxy = torch.clamp(F.conv2d(obs * syn, self.g_masks, groups=1, padding=self.pad) - muxy, min = 0)
        sigmax2 = F.conv2d(obs * obs, self.g_masks, groups=1, padding=self.pad) - mux2
        sigmay2 = F.conv2d(syn * syn, self.g_masks, groups=1, padding=self.pad) - muy2
        sigmaxy = F.conv2d(obs * syn, self.g_masks, groups=1, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        contrast = self.ct * torch.sqrt(torch.clamp(sigmax2 * sigmay2, min=1e-20))
        L  = ((2 * muxy    + self.C1) / (mux2    + muy2    + self.C1))   # [B, 1, T, R]
        # CS = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2 + contrast)

        sigma_x = torch.sqrt(sigmax2 + 1e-20)
        sigma_y = torch.sqrt(sigmay2 + 1e-20)
        C = ((2 * sigma_x * sigma_y + self.C2) / (sigmax2 + sigmay2 + self.C2 + contrast) + 1) / 2
        S = ((sigmaxy + self.C2/2) / (sigma_x * sigma_y + self.C2/2) + 1) / 2
        CS = C*S

        # MS-SSIM loss calculation
        lM = L[:, -1, :, :]
        # lM = L.prod(dim=1)
        PIcs = CS.prod(dim=1)
        
        if self.type == "mean" :
            loss_ms_ssim = torch.mean(-(L*CS)**self.p)
            
        else :
            loss_ms_ssim = torch.sum(torch.mean( - (L*CS)**self.p, dim=(2, 3)))
            # loss_ms_ssim = self.L2_fn.forward(0 * (lM*PIcs), 1-(lM*PIcs)**self.p)
            # loss_ms_ssim = torch.sum(1 - lM *  PIcs)
            # loss_ms_ssim = 1 - lM * PIcs  # [B, T, R]

        # L1 Loss
        loss_l1 = F.l1_loss(obs, syn, reduction='none')  # [B, 1, T, R]
        # average l1 loss across the channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-1, length=1),
                               groups=1, padding=self.pad).mean(1)  # [B, T, R]

        # Mixed loss: MS-SSIM + L1
        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * loss_l1
        loss_mix = self.compensation * loss_mix

        return loss_mix.mean()
        
        


from .base import Misfit 
from .L2 import Misfit_waveform_L2      
class Misfit_local2_correlation(Misfit):
    def __init__(self, gaussian_sigmas=[0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
                 data_range = 1.0,
                 K=(0.01, 0.03),
                 alpha=1,
                 compensation=200.0,
                 cuda_dev=0,
                 ct=0,
                 p=1,
                 type="mean"):
        super(Misfit_local2_correlation, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(4 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.ct = ct
        self.compensation=compensation
        self.L2_fn = Misfit_waveform_L2()
        self.type = type
        self.p = p
        filter_size = int(8 * gaussian_sigmas[-1] + 1)
        
        # Create the Gaussian masks for convolution
        g_masks = torch.zeros((len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        
        #self.g_masks = g_masks.cuda(cuda_dev)
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

    def forward(self, obs, syn):

        syn = syn/(1e-10 + torch.max(torch.abs(syn),axis=1,keepdim=True).values)
        obs = obs/(1e-10 + torch.max(torch.abs(obs),axis=1,keepdim=True).values)
        # obs, syn, dd = transform(obs, syn, 'envelope', theta=1)


        obs = obs.unsqueeze(1)  # (B, 1, T, R)
        syn = syn.unsqueeze(1)  # (B, 1, T, R)
        
        # syn = F.avg_pool2d(syn, kernel_size=(2,1), stride=(2,1))
        # obs = F.avg_pool2d(obs, kernel_size=(2,1), stride=(2,1))

        
        B, C, T, R = syn.shape  # 获取批次大小、通道数、时间采样点和接收器数
        self.g_masks = self.g_masks.to(syn.device)
        
        #syn = self.mean_filter(syn, window_size=1)
        #obs = self.mean_filter(obs, window_size=1)
        

        # Apply 2D convolution with Gaussian kernels to get the necessary statistics
        mux = F.conv2d(obs, self.g_masks, groups=1, padding=self.pad)
        muy = F.conv2d(syn, self.g_masks, groups=1, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        # sigmax2 = torch.clamp(F.conv2d(obs * obs, self.g_masks, groups=1, padding=self.pad) - mux2, min = 0)
        # sigmay2 = torch.clamp(F.conv2d(syn * syn, self.g_masks, groups=1, padding=self.pad) - muy2, min = 0)
        # sigmaxy = torch.clamp(F.conv2d(obs * syn, self.g_masks, groups=1, padding=self.pad) - muxy, min = 0)
        sigmax2 = F.conv2d(obs * obs, self.g_masks, groups=1, padding=self.pad) - mux2
        sigmay2 = F.conv2d(syn * syn, self.g_masks, groups=1, padding=self.pad) - muy2
        sigmaxy = F.conv2d(obs * syn, self.g_masks, groups=1, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        # contrast = self.ct * torch.sqrt(torch.clamp(mux2 * muy2, min=1e-20))
        L  = (2 * muxy    + self.C1) / (mux2    + muy2    + self.C1 )  # [B, 1, T, R]
        contrast = self.ct * torch.sqrt(torch.clamp(sigmax2 * sigmay2, min=1e-20))
        CS = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2 + contrast)

        # MS-SSIM loss calculation
        lM = L[:, -1, :, :]
        # lM = L.prod(dim=1)
        PIcs = CS.prod(dim=1)
        
        if self.type == "mean" :
            loss_ms_ssim = torch.mean(-(L*CS)**self.p)
            # loss_ms_ssim = -torch.mean((torch.matmul(L.view(L.size(0), L.size(1), -1), CS.view(CS.size(0), CS.size(1), -1).transpose(1, 2)))**self.p)
            
        else :
            # loss_ms_ssim = torch.sum(torch.mean( - (L*CS)**self.p, dim=(2, 3)))
            # loss_ms_ssim = self.L2_fn.forward(0 * (lM*PIcs), 1-(lM*PIcs)**self.p)
            loss_ms_ssim = torch.mean(1 - lM*PIcs)
            # loss_ms_ssim = 1 - lM * PIcs  # [B, T, R]

        # L1 Loss
        loss_l1 = F.l1_loss(obs, syn, reduction='none')  # [B, 1, T, R]
        # average l1 loss across the channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-1, length=1),
                               groups=1, padding=self.pad).mean(1)  # [B, T, R]

        # Mixed loss: MS-SSIM + L1
        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * loss_l1
        loss_mix = self.compensation * loss_mix

        return loss_mix.mean()


class Misfit_local3_correlation(Misfit):
    def __init__(self, gaussian_sigmas=[0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
                 data_range = 1.0,
                 K=(0.01, 0.03),
                 alpha=1,
                 compensation=200.0,
                 cuda_dev=0,
                 ct=0,
                 p=1,
                 type="mean"):
        super(Misfit_local3_correlation, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(4 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.ct = ct
        self.compensation=compensation
        self.L2_fn = Misfit_waveform_L2()
        self.type = type
        self.p = p
        filter_size = int(8 * gaussian_sigmas[-1] + 1)
        
        # Create the Gaussian masks for convolution
        g_masks = torch.zeros((len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        
        #self.g_masks = g_masks.cuda(cuda_dev)
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

    def forward(self, obs, syn):

        syn = syn/(1e-10 + torch.max(torch.abs(syn),axis=1,keepdim=True).values)
        obs = obs/(1e-10 + torch.max(torch.abs(obs),axis=1,keepdim=True).values)
        # obs, syn, dd = transform(obs, syn, 'envelope', theta=1)


        obs = obs.unsqueeze(1)  # (B, 1, T, R)
        syn = syn.unsqueeze(1)  # (B, 1, T, R)
        
        # syn = F.avg_pool2d(syn, kernel_size=(2,1), stride=(2,1))
        # obs = F.avg_pool2d(obs, kernel_size=(2,1), stride=(2,1))

        
        B, C, T, R = syn.shape  # 获取批次大小、通道数、时间采样点和接收器数
        self.g_masks = self.g_masks.to(syn.device)
        
        #syn = self.mean_filter(syn, window_size=1)
        #obs = self.mean_filter(obs, window_size=1)
        

        # Apply 2D convolution with Gaussian kernels to get the necessary statistics
        mux = F.conv2d(obs, self.g_masks, groups=1, padding=self.pad)
        muy = F.conv2d(syn, self.g_masks, groups=1, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(obs * obs, self.g_masks, groups=1, padding=self.pad) - mux2
        sigmay2 = F.conv2d(syn * syn, self.g_masks, groups=1, padding=self.pad) - muy2
        sigmaxy = F.conv2d(obs * syn, self.g_masks, groups=1, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        # contrast = self.ct * torch.sqrt(torch.clamp(mux2 * muy2, min=1e-20))
        L  = (2 * muxy    + self.C1) / (mux2    + muy2    + self.C1)  # [B, 1, T, R]
        contrast = self.ct * torch.sqrt(torch.clamp(sigmax2 * sigmay2, min=1e-20))
        CS = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2 + contrast)

        # MS-SSIM loss calculation
        lM = L[:, -1, :, :]
        # lM = L.prod(dim=1)
        PIcs = CS.prod(dim=1)
        
        if self.type == "mean" :
            loss_ms_ssim = torch.mean(-(L*CS)**self.p)
            # loss_ms_ssim = -torch.mean((torch.matmul(L.view(L.size(0), L.size(1), -1), CS.view(CS.size(0), CS.size(1), -1).transpose(1, 2)))**self.p)
            
        else :
            # loss_ms_ssim = torch.sum(torch.mean( - (L*CS)**self.p, dim=(2, 3)))
            # loss_ms_ssim = self.L2_fn.forward(0 * (lM*PIcs), 1-(lM*PIcs)**self.p)
            loss_ms_ssim = torch.mean(1 - lM*PIcs)
            # loss_ms_ssim = 1 - lM * PIcs  # [B, T, R]

        # L1 Loss
        loss_l1 = F.l1_loss(obs, syn, reduction='none')  # [B, 1, T, R]
        # average l1 loss across the channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-1, length=1),
                               groups=1, padding=self.pad).mean(1)  # [B, T, R]

        # Mixed loss: MS-SSIM + L1
        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * loss_l1
        loss_mix = self.compensation * loss_mix

        return loss_mix.mean()
        




















        


