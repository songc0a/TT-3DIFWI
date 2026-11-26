import torch, os
import numpy as np
from torch import nn
from torch.nn import functional as F   
import random


        
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.0, type='sin'):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.type=type
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
        
    def forward(self, input):
        if self.type=='sin':
            return torch.sin(self.omega_0 *self.linear(input))
        

        
class Siren(nn.Module):

    def __init__(self, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features, 
                 outermost_linear=False, 
                 pretrained=None,
                 first_omega_0=30.0, 
                 hidden_omega_0=30.0, 
                 domain_shape=None, 
                 type='sin', 
                 **kwargs):
        super().__init__()
        self.domain_shape = domain_shape
        self.net = []
        self.type = type
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0, type = self.type))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0, type = self.type))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
            # self.net.append(nn.Sigmoid())
            self.net.append(nn.Tanh())
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
        self.coords = self.generate_mesh(domain_shape)
        self.load_pretrained(pretrained)

    def load_pretrained(self, pretrained):
        pretrained = '' if pretrained is None else pretrained
        if os.path.exists(pretrained):
            self.load_state_dict(torch.load(pretrained))
        else:
            print(f"Cannot find the pretrained model '{pretrained}'. Using random initialization.")
    
    def generate_mesh(self, mshape):
        '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
        sidelen: int
        dim: int'''
        tensors_for_meshgrid = []
        for size in mshape:
            tensors_for_meshgrid.append(torch.linspace(-1, 1, steps=size))
        mgrid = torch.stack(torch.meshgrid(*tensors_for_meshgrid, indexing='ij'), dim=-1)
        mgrid = mgrid.reshape(-1, len(mshape))
        return mgrid

    def step(self,):
        pass

    def forward(self, coords, coords_shape=None):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        # output = self.net(coords).view(self.domain_shape)
        # output = self.net(coords).view(coords.shape[0]/(self.domain_shape[1]*self.domain_shape[2]),self.domain_shape[1],self.domain_shape[2])
        if coords_shape == None:
            output = self.net(coords).view(self.domain_shape)
        else:
            output = self.net(coords).view(coords_shape)
        return output






class F_Siren(nn.Module):

    def __init__(self, 
                 in_features, 
                 hidden_features, 
                 hidden_layers, 
                 rank,
                 out_features, 
                 outermost_linear=False, 
                 pretrained=None,
                 first_omega_0=30.0, 
                 hidden_omega_0=30.0, 
                 domain_shape=None, 
                 type='sin', 
                 decomp_mode = "TT",
                 **kwargs):
        super().__init__()
        self.domain_shape = domain_shape
        self.net = []
        self.type = type
        self.rank = rank
        self.decomp_mode = decomp_mode
        print(f"Domain Shape: {domain_shape}, Mode: {decomp_mode}")

        if self.decomp_mode == "TT":
            self.fz = Siren(in_features=in_features, out_features=self.rank, 
                      hidden_features=hidden_features, hidden_layers=hidden_layers, first_omega_0=first_omega_0, 
                      hidden_omega_0=hidden_omega_0,type=type,
                      outermost_linear=True,domain_shape=(domain_shape[0],1,rank))
            self.fy = Siren(in_features=in_features, out_features=self.rank*self.rank, 
                      hidden_features=hidden_features, hidden_layers=hidden_layers, first_omega_0=first_omega_0, 
                      hidden_omega_0=hidden_omega_0,type=type,
                      outermost_linear=True,domain_shape=(domain_shape[1],rank,rank))
            self.fx = Siren(in_features=in_features, out_features=self.rank, 
                      hidden_features=hidden_features, hidden_layers=hidden_layers, first_omega_0=first_omega_0, 
                      hidden_omega_0=hidden_omega_0,type=type,
                      outermost_linear=True,domain_shape=(domain_shape[2],rank,1))
        elif self.decomp_mode == "CP":
            self.fz = Siren(in_features=in_features, out_features=self.rank, 
                      hidden_features=hidden_features, hidden_layers=hidden_layers, first_omega_0=first_omega_0, 
                      hidden_omega_0=hidden_omega_0,type=type,
                      outermost_linear=True,domain_shape=(domain_shape[0],rank))
            self.fy = Siren(in_features=in_features, out_features=self.rank, 
                      hidden_features=hidden_features, hidden_layers=hidden_layers, first_omega_0=first_omega_0, 
                      hidden_omega_0=hidden_omega_0,type=type,
                      outermost_linear=True,domain_shape=(domain_shape[1],rank))
            self.fx = Siren(in_features=in_features, out_features=self.rank, 
                      hidden_features=hidden_features, hidden_layers=hidden_layers, first_omega_0=first_omega_0, 
                      hidden_omega_0=hidden_omega_0,type=type,
                      outermost_linear=True,domain_shape=(domain_shape[2],rank))
        else:
            raise ValueError(f"Unknown decomp_mode: {decomp_mode}")



        self.zcoords = self.generate_mesh(domain_shape[0])
        self.ycoords = self.generate_mesh(domain_shape[1])
        self.xcoords = self.generate_mesh(domain_shape[2])
        self.ac = nn.Tanh()


    def generate_mesh(self, n):
        return torch.linspace(-1, 1, steps=n).unsqueeze(-1)

    def forward(self, zcoords, ycoords, xcoords):
        zcoords = zcoords.clone().detach().requires_grad_(True)
        ycoords = ycoords.clone().detach().requires_grad_(True)
        xcoords = xcoords.clone().detach().requires_grad_(True)

        if self.decomp_mode == "TT":
            # G1 = self.fz(zcoords, (self.domain_shape[0], self.rank, 1)).permute(2, 0, 1)
            G1 = self.fz(zcoords, (self.domain_shape[0],1,self.rank)).permute(1, 0, 2)
            G2 = self.fy(ycoords, (self.domain_shape[1],self.rank,self.rank)).permute(1, 0, 2)
            G3 = self.fx(xcoords, (self.domain_shape[2],self.rank,1)).permute(1, 0, 2)
            output = torch.einsum('aib, bjc, ckd -> ijk', G1, G2, G3)
               

        elif self.decomp_mode == "CP":
            A = self.fz(zcoords) # (D1, R)
            B = self.fy(ycoords) # (D2, R)
            C = self.fx(xcoords) # (D3, R)
            output = torch.einsum('ir, jr, kr -> ijk', A, B, C)



        return self.ac(output)