# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 14:24:20 2025

@author: 87877
"""

import scipy.io
import numpy as np
import torch
from torch import nn
import scipy.io
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import time
import random
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import cv2


import imageio
from argparse import ArgumentParser
from collections import OrderedDict

import einops
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Subset

from torchvision.transforms import Compose, ToTensor, Lambda, Resize, Pad
from torchvision.datasets.mnist import MNIST, FashionMNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"Using device: {device}\t"
    + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU")
)

feature_dim=400

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width      # the number of tokens
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    
width = 192

class signalTransformer(nn.Module):
    def __init__(self,num = 5):
        super().__init__()
        self.num = num
        self.A_conv=nn.Sequential(
            nn.Conv1d(feature_dim,20,3,1,1),
            nn.Conv1d(20,1,3,1,1),
        )
        self.transformer = Transformer(
            # width=72*self.num,          # Three test: rwidth = 72*5/30*5/384
            width = width,      # the number of tokens
            layers=2,
            heads=8,
            # heads=10,        # only for width = 60
            # attn_mask=build_attention_mask(32)
        )
        self.ln_final = LayerNorm(width)
        self.final_conv = nn.Conv1d(int(11520/width)*2,1,3,1,1)
        self.fc = nn.Linear(width,400)
    def forward(self,x):
        x = x.reshape(x.shape[0],feature_dim+1,11520)    # 1,32,360  B,w,h
        #x = x.permute(0,1,3,5,2,4)
        A = x[:,:feature_dim,:]
        y = x[:,-1,:]
        A = self.A_conv(A).reshape(x.shape[0],2, 16, 5, 12, 6)
        A = A.reshape(x.shape[0],int(11520/width),width)    # 1,32,360  B,w,h
        # x = x.permute(1,0,2)        # 32,1,360  # w,B,h
        A = self.transformer(A)
        y = y.reshape(x.shape[0],2, 16, 5, 12, 6)
        y = y.reshape(x.shape[0],int(11520/width),width)    # 1,32,360  B,w,h
        # x = x.permute(1,0,2)        # 32,1,360  # w,B,h
        y = self.transformer(y)
        # x = x.permute(1,0,2)
        x = torch.concat([A,y],dim=1)
        x = self.ln_final(x)
        x = self.final_conv(x)
        return self.fc(x.squeeze(1))
    
class MyDDPM(nn.Module):
    def __init__(
        self,
        network,
        n_steps=200,
        min_beta=10**-4,
        max_beta=0.02,
        device=None,
        image_chw=(1, 20, 200),
    ):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
            device
        )  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor(
            [torch.prod(self.alphas[: i + 1]) for i in range(len(self.alphas))]
        ).to(device)

    def forward(self, x0, t, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        noisy = (
            a_bar.sqrt().reshape(n, 1, 1, 1) * x0
            + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        )
        return noisy

    def backward(self, x, y, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network(x, y, t)
    


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk[:, ::2])
    embedding[:, 1::2] = torch.cos(t * wk[:, ::2])

    return embedding

class MyBlock(nn.Module):
    def __init__(
        self,
        shape,
        in_c,
        out_c,
        kernel_size=3,
        stride=1,
        padding=1,
        activation=None,
        normalize=True,
    ):
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out
    
class MySigUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(MySigUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.signal_transformer = signalTransformer(num=5)
        self.te0 = self._make_te(time_emb_dim, 1)
        self.mix_conv=nn.Conv2d(2, 1, 3, 1, 1)

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((2, 20, 20), 2, 10),
            MyBlock((10, 20, 20), 10, 10),
            MyBlock((10, 20, 20), 10, 10),
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            MyBlock((10, 10, 10), 10, 20),
            MyBlock((20, 10, 10), 20, 20),
            MyBlock((20, 10, 10), 20, 20),
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            MyBlock((20, 5, 5), 20, 40),
            MyBlock((40, 5, 5), 40, 40),
            MyBlock((40, 5, 5), 40, 40),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1), nn.SiLU(), nn.Conv2d(40, 40, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyBlock((40, 2, 2), 40, 20),
            MyBlock((20, 2, 2), 20, 20),
            MyBlock((20, 2, 2), 20, 40),
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1),
        )

        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            MyBlock((80, 5, 5), 80, 40),
            MyBlock((40, 5, 5), 40, 20),
            MyBlock((20, 5, 5), 20, 20),
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            MyBlock((40, 10, 10), 40, 20),
            MyBlock((20, 10, 10), 20, 10),
            MyBlock((10, 10, 10), 10, 10),
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            MyBlock((20, 20, 20), 20, 10),
            MyBlock((10, 20, 20), 10, 10),
            MyBlock((10, 20, 20), 10, 10, normalize=False),
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)       

    def forward(self, x, y, t):
        # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
        
        
        t = self.time_embed(t)
        n = len(x)
        
        #A = y[:,:feature_dim,:]
        #y = y[:,-1,:]

        # out_y = self.signal_transformer(y + self.te0(t).reshape(n, -1, 1)).reshape(n,1,20,20)       # add time embedding to y

        # out_y = out_y + self.te0(t).reshape(n, -1, 1, 1)           # add time embedding to out_y

        # out_y = self.signal_transformer_embed_t(y, t).reshape(n,1,20,20)    # concat time embeding to y

        #out_A = self.signal_transformer_A(A).reshape(n,1,20,20)
        out_y = self.signal_transformer(y).reshape(n,1,20,20)       # no time embedding to y
        #out_y = torch.concat([out_A,out_y],dim=1)
        #out_y = self.mix_conv(out_y)
        
        out1 = self.b1(torch.cat((out_y, x + self.te1(t).reshape(n, -1, 1, 1)), dim=1))  # (N, 10, 28, 28)
        
        out2 = self.b2(
            self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1)
        )  # (N, 20, 14, 14)
        out3 = self.b3(
            self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1)
        )  # (N, 40, 7, 7)

        out_mid = self.b_mid(
            self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1)
        )  # (N, 40, 3, 3)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 7, 7)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 14, 14)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)


        out = self.conv_out(out)[:,0,:,:]
        out = out[:, None, :, :]

        return out, out_y

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out)
        )
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
setup_seed(3407)
    
def snr_out(signal_source,signal_source_noise):
    signal_noise = signal_source - signal_source_noise
    snr = 10 * torch.log10(torch.sum(signal_source.abs()**2) / torch.sum(signal_noise.abs()**2))
    return snr

transform_size = transforms.Resize((7,7))

class objects(Dataset):
    def __init__(self,train,num_samples,train_dict,test_dict):
        if train:
            df = pd.read_csv(train_dict,sep=',')
        else:
            df  = pd.read_csv(test_dict ,sep = ',')
        data = np.array(df, dtype = 'float32')
        # 目前是 784 28*28 更改为
        data = data[:,1:]/255
        data = torch.FloatTensor(data).reshape(data.shape[0],1,28,28)
        self.mydata = torch.zeros((num_samples,20,20))
        for i in tqdm(range(num_samples)):
            index = (random.randint(3,8),random.randint(3,8))
#             random.shuffle(index)
#             self.mydata[i,index[:3]] = 1 
            nowdata = transform_size(data[i]) 
#             print(nowdata.shape,self.mydata[i,index[0]-3:index[0]+4,index[1]-3:index[1]+4].shape)
            self.mydata[i,index[0]-3:index[0]+4,index[1]-3:index[1]+4] = nowdata
    def __getitem__(self,index):
        return self.mydata[index].reshape(-1)
    def __len__(self):
        return self.mydata.shape[0]
    
class mmWaveSimulator(nn.Module):
    def __init__(self,distance=0.3, resolution=0.01, meta_size=60,addNoise=False):
        super(mmWaveSimulator,self).__init__()
        
        self.f = torch.tensor([range(int(77e9),int(81e9+1),int(1e9))],dtype=torch.float32).squeeze(0).to(device) #torch.tensor([79e9],dtype=torch.float32).to(device) # 
        self.c = 3e8
        self.wave_lambda = 0.0038  #self.c / 79e9
        
        self.tx_positions = torch.FloatTensor([[0,-8*self.wave_lambda,0],
                                              [0,-6*self.wave_lambda,0],
                                              [0,-4*self.wave_lambda,0],
                                              [0,-2*self.wave_lambda,0],
                                              [0,0,0],
                                              [0,2*self.wave_lambda,0],
                                              [0,4*self.wave_lambda,0],
                                              [0,6*self.wave_lambda,0],
                                              [0,8*self.wave_lambda,0],
                                              [0,-2.5*self.wave_lambda,-3*self.wave_lambda],
                                              [0,-3.0*self.wave_lambda,-2*self.wave_lambda],
                                              [0,-3.5*self.wave_lambda,-0.5*self.wave_lambda]]).to(device)

        self.rx_positions = torch.FloatTensor([
           [0,-16*self.wave_lambda,-0.066],
           [0,-15.5*self.wave_lambda,-0.066],
           [0,-15*self.wave_lambda,-0.066],
           [0,-14.5*self.wave_lambda,-0.066],
           [0,-10.5*self.wave_lambda,-0.066],
           [0,-10*self.wave_lambda,-0.066],
           [0,-9.5*self.wave_lambda,-0.066],
           [0,-9*self.wave_lambda,-0.066],
           [0,7*self.wave_lambda,-0.066],
           [0,7.5*self.wave_lambda,-0.066],
           [0,8*self.wave_lambda,-0.066],
           [0,8.5*self.wave_lambda,-0.066],
           [0,9*self.wave_lambda,-0.066],
           [0,9.5*self.wave_lambda,-0.066],
           [0,10*self.wave_lambda,-0.066],
           [0,10.5*self.wave_lambda,-0.066],
       ]).to(device)
        

        
        metasurface_size = meta_size
        if metasurface_size == 1:
            meta_positions = np.array([[0.1,0,0]],dtype=np.float32)
        else: 
            resolution = 0.0015 #self.wave_lambda / 2 
            size = metasurface_size
            edge_left  = [-i for i in range(1,size,2)]
            edge_left.reverse()
            edge_right = [i for i in range(1,size,2)]
            edge = edge_left + edge_right
            edge = np.array(edge) * resolution / 2.0
            wx,wy = np.meshgrid(edge,edge)
            wz = np.zeros((size,size),dtype=np.float32)
            meta_positions = np.stack((wz,wx,wy),2).reshape(size**2,-1)
            meta_positions[:,0] = 0.1
            
        self.meta_positions = torch.tensor(meta_positions,dtype=torch.float32).to(device)
        self.meta_theta = nn.Parameter(torch.randint(-180,20,[meta_positions.shape[0],1]).to(device)*torch.pi/180)
        #self.meta_theta = nn.Parameter(torch.ones([meta_positions.shape[0],1]).to(device))
        
        
        resolution = resolution 
        size = 20
        edge_left  = [-i for i in range(1,size,2)]
        edge_left.reverse()
        edge_right = [i for i in range(1,size,2)]
        edge = edge_left + edge_right
        if size % 2 == 1:
            edge = [i for i in range(-(size-1),size,2)]
        edge = np.array(edge) * resolution / 2.0
        wx,wy = np.meshgrid(edge,edge)
        wz = np.zeros((size,size),dtype=np.float32)
        scene_positions = np.stack((wz,wx,wy),2).reshape(size**2,-1)
        scene_positions[:,0] = distance
        self.scene_positions = torch.tensor(scene_positions,dtype=torch.float32).to(device)
        
        
        ## code book of speakers
        
        self.w = nn.Parameter(torch.randn((torch.numel(self.f), 6, self.tx_positions.shape[0]), dtype=torch.float32).to(device))
        #self.w = nn.Parameter(torch.ones((torch.numel(self.f), 6, self.tx_positions.shape[0]), dtype=torch.float32).to(device))
        

        self.snr = 20
        self.excitation = False
        self.deNoise = True
        self.addNoise = addNoise
        
    
    
    def compute_distance(self,locs_start, locs_end):
        locs_start = locs_start.unsqueeze(dim=1)# row major
        locs_end = locs_end.unsqueeze(dim=0)# column major
        d = torch.norm(locs_start - locs_end, dim=2)# 2D matrix from 2 vectors
        return d
    
    def compute_channel(self,locs_start, locs_end, f):
#         f = f.unsqueeze(dim=1).unsqueeze(dim=2)
        f = f.unsqueeze(1).unsqueeze(2)
        d = self.compute_distance(locs_start, locs_end)
        d = d.unsqueeze(0)
        H = 0.001 / d**2 * torch.exp(-1j*2*torch.pi*f*d/self.c)
        return  H
    
    
    def compute_measurement_matrix(self):
        meta_theta = self.meta_theta.clamp(-180*torch.pi/180,20*torch.pi/180)
        pri = torch.exp(1j*self.w)
        
        H1 = self.compute_channel(self.tx_positions, self.meta_positions, self.f)
        H2 = self.compute_channel(self.meta_positions, self.scene_positions, self.f)
        H3 = self.compute_channel(self.scene_positions, self.meta_positions, self.f)
        H4 = self.compute_channel(self.meta_positions, self.rx_positions, self.f)

        A = torch.matmul(pri.unsqueeze(2), H1.unsqueeze(1))

        A = A * torch.exp(1j * meta_theta[:,0])
        A = torch.matmul(A, H2.unsqueeze(1))
        A = torch.diag_embed(A.squeeze(2),dim1=2)
        A = torch.matmul(A,H3.unsqueeze(1))
        A = A * torch.exp(1j * meta_theta[:,0])
        A = torch.matmul(A,H4.unsqueeze(1)) #update A
        return A
    
    def add_noise(self,mic_data):
        noise_real = (torch.randn(mic_data.shape) * (0.028)).type(torch.float32).to(device)
        noise_imag = (torch.randn(mic_data.shape) * (0.028)).type(torch.float32).to(device)
        noise = noise_real + 1.j*noise_imag
        mic_data_noise = mic_data + noise
        return mic_data_noise,noise,mic_data
    
    def forward(self,T):
        A = self.compute_measurement_matrix()
        #A = A.permute(0,1,3,2)
        A = A.permute(3,0,1,2)
        A=A.repeat(1,1,12,1)
        A = A.reshape(-1,A.shape[3])
        
        T_complex = torch.zeros_like(T,dtype=torch.complex64)
        T_complex.real = T
        mic_data = torch.matmul(A,T_complex)
        
        if self.addNoise:
            mic_data,noise,mic_data_or = self.add_noise(mic_data)
            s_noise = torch.cat((noise.real,noise.imag),0)
            s_mic_data_or = torch.cat((mic_data_or.real,mic_data_or.imag),0)
            s_mic_data_or = s_mic_data_or.unsqueeze(-1)
        
        s_A =  torch.cat((A.real,A.imag),0)
        s_mic_data =  torch.cat((mic_data.real,mic_data.imag),0)
        s_mic_data = s_mic_data.unsqueeze(-1)
        if self.addNoise:
            return s_A,s_mic_data,s_noise,s_mic_data_or
        else:
            return s_A,s_mic_data
    def getA(self):
        A = self.compute_measurement_matrix()
        #A = A.permute(0,1,3,2)
        A = A.permute(3,0,1,2)
        #print(A.shape)
        A=A.repeat(1,1,12,1)
        A = A.reshape(-1,A.shape[3])
        s_A = torch.cat((A.real,A.imag),0)
        #print(A.shape)
        return A,s_A

class mymseLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,T,Tpred):
        return torch.sum((T-Tpred)**2)
    
class admmSolver(nn.Module):
    def __init__(self,rho=0.1,lam=0.01):
        super(admmSolver,self).__init__()
        self.rho = nn.Parameter(torch.FloatTensor([rho]))
        self.lam = nn.Parameter(torch.FloatTensor([lam]))
        self.criterion = nn.MSELoss()
        self.iter_number = 1000
        
    def ADMM_S(self,tau, g):
        a = g - tau
        a[a<0] = 0
        b = g + tau
        b[b>0] = 0
        return a+b
    def forward(self,T,s_A,s_mic_data,log=True,norm_para=1e-5):

        Tpred = torch.zeros_like(T).to(device)
        if log == True:
            mseLossHistory = []
        z_old = torch.zeros_like(Tpred).to(device)
        u_old = torch.zeros_like(Tpred).to(device)
        
        I = torch.eye(s_A.shape[1]).to(device)
        rho = torch.abs(self.rho)+1e-10
        lam = torch.abs(self.lam)
        for i in range(self.iter_number):  
            Tpred = torch.matmul(torch.linalg.inv((torch.matmul(s_A.T,s_A)+rho*I)),(torch.matmul(s_A.T,s_mic_data)+ rho * (z_old-u_old)))  
            z_new = self.ADMM_S(lam/rho, Tpred+u_old) 
            z_new = z_new.clamp(0.0,1.0)
            u_new  = u_old + Tpred-z_new
            z_old = z_new
            u_old = u_new
            if log == True:
                mseLossHistory.append(self.criterion(Tpred.clamp(0.0,1.0),T).item())
        if log == True:
            return Tpred,mseLossHistory
        else:
            return Tpred

def trloss(A):
    return A.norm(p='nuc') 
def corr_loss(A,mode="Mean"):
    if mode == 'Max':
        #max loss
        D = torch.nn.functional.normalize(A.real, p=2, dim=0).to(device)
        B = torch.matmul(D.T, D)
        I = torch.eye(B.shape[0]).to(device)
        output = torch.max(torch.abs(B.view(-1) - I.view(-1)))
    elif mode == "Mean":
        #mean loss
        D = torch.nn.functional.normalize(A.real, p=2, dim=0).to(device)
        B = torch.matmul(D.T, D)
        I = torch.eye(B.shape[0]).to(device)
        output = torch.sum((B.view(-1) - I.view(-1)) ** 2)
    elif mode == "t-Mean":
        D = torch.nn.functional.normalize(A.real, p=2, dim=0).to(device)
        B = torch.matmul(D.T, D)
        I = torch.eye(B.shape[0]).to(device)
        temp = B.view(-1) - I.view(-1)
        temp[temp<=0.2] = 0
        output = torch.sum(temp**2)
    return output

def new_ddpm_sample(
        y,        
    ddpm,
    ddpm_steps = 100,
    device=None,
    c=1,
    h=20,
    w=20):
    with torch.no_grad():
        if device is None:
            device = ddpm.device
        n=y.shape[0]
        # Starting from random noise
        x = torch.randn(n, c, h, w).to(device)

        factor = ddpm.n_steps // ddpm_steps
        ddim_timestep_seq = np.asarray(list(range(0, ddpm.n_steps, factor)))
        # ddim_timestep_seq = ddim_timestep_seq + 1

        # for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
        for i in tqdm(reversed(range(0, ddpm_steps)), desc='sampling loop time step', total=ddpm_steps):
            time_tensor = (torch.ones(n) * ddim_timestep_seq[i]).to(device).long()
            
            eta_theta,_ = ddpm.backward(x, y, time_tensor)

            # alpha_t = ddpm.alphas[t]
            # alpha_t_bar = ddpm.alpha_bars[t]
            alpha_t = ddpm.alphas[ddim_timestep_seq[i]]
            alpha_t_bar = ddpm.alpha_bars[ddim_timestep_seq[i]]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (
                x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
            )

            # if t > 0:
            if ddim_timestep_seq[i] > 0:
                z = torch.randn(1, c, h, w).to(device)

                # Option 1: sigma_t squared = beta_t
                # beta_t = ddpm.betas[t]
                # sigma_t = beta_t.sqrt()
                beta_t = ddpm.betas[ddim_timestep_seq[i]]
                sigma_t = beta_t.sqrt()

                # Option 2: sigma_t squared = beta_tilda_t
                # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                # sigma_t = beta_tilda_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z

        normalized = x.clone()
        normalized -= torch.min(normalized)
        normalized *= 1 / torch.max(normalized)

        return normalized

def optimize_step(optimizer,Runtime):
    epochs   = 50
    max_loss = 99999
    loss_history_train = []
    loss_history_valid = []
    #################
    rho_history = []
    lam_history = []
    #################
    best_loss = max_loss
    criterion=nn.MSELoss()
    n_steps = best_model.n_steps
    #admmsolver.iter_number = 400
    for i in tqdm(range(epochs)):
        simulator.train()
        #admmsolver.train()
        loss_train = 0.0
        loss_valid = 0.0
        for T in tqdm(loader_train):
            T = T.to(device)
            n=T.shape[0]
            T = T.transpose(1,0)
            A,mic_data,_,_ = simulator(T)
            A=A.permute(1,0).unsqueeze(0).repeat(n,1,1)
            mic_data=mic_data.permute(1,2,0)
            y = torch.concat([A,mic_data],dim=1)

            T0=T.permute(1,0).reshape(-1,1,20,20)
            eta = torch.randn_like(T0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)
            noisy_imgs = best_model(T0, t, eta)
            eta_theta,_ = best_model.backward(noisy_imgs, y, t.reshape(n, -1))
            loss = criterion(eta_theta, eta)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        loss_history_train.append(loss_train / len(loader_train))
        if(scheduler is not None):
            scheduler.step()
        simulator.eval()

        j = 0
        with torch.no_grad():
            for T in loader_valid:
                T = T.to(device)
                n=T.shape[0]
                T = T.transpose(1,0)
                A,mic_data,_,_ = simulator(T)
                A=A.permute(1,0).unsqueeze(0).repeat(n,1,1)
                mic_data=mic_data.permute(1,2,0)
                y = torch.concat([A,mic_data],dim=1)

                T0=T.permute(1,0).reshape(-1,1,20,20)
                eta = torch.randn_like(T0).to(device)
                t = torch.randint(0, n_steps, (n,)).to(device)
                noisy_imgs = best_model(T0, t, eta)
                eta_theta,_ = best_model.backward(noisy_imgs, y, t.reshape(n, -1))
                loss = criterion(eta_theta, eta)
                
                loss_valid += loss.item()
            loss_history_valid.append(loss_valid / len(loader_valid))
            if(loss_valid/len(loader_valid) < best_loss):
                best_loss = loss_valid/len(loader_valid)
                torch.save(simulator.state_dict(), './mts_model/mmSimulator'+Runtime+'.pkl')
                #torch.save(admmsolver.state_dict(), './model/admmsolverForMM'+Runtime+'.pkl')
            print(COMMENTS + " iter: ", i,". Rank of measurement matrix is ", torch.linalg.matrix_rank(A))
            print(COMMENTS + " iter: ", i,". loss_train: ",loss_train / len(loader_train),". loss_valid: ", loss_valid/len(loader_valid))
    
if __name__ == "__main__":
    train_dict='dataset/fashion-mnist_train.csv'
    test_dict='dataset/fashion-mnist_test.csv'
    train_set = objects(True,1000,train_dict,test_dict)
    test_set = objects(False,200,train_dict,test_dict)
    batch_size = 16
    loader_train = DataLoader(train_set,batch_size=batch_size,shuffle=True)
    loader_valid = DataLoader(test_set,batch_size=batch_size,shuffle=False)
    loader_test  = DataLoader(test_set,batch_size=batch_size,shuffle=False)
    simulator = mmWaveSimulator(addNoise=True).to(device)
    A,s_A=simulator.getA()
    lr = 1e-2
    optimizer = torch.optim.Adam([
                       {'params': simulator.w,'lr': lr},
                       {'params': simulator.meta_theta, 'lr': lr},
                        ], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
    '''
    initialization
    '''
    iter_number = 0
    COMMENTS = " "
    epochs = 2000
    rank = []
    corr = []
    for i in tqdm(range(epochs)):
        simulator.train()
        A,s_A = simulator.getA()
        Loss = corr_loss(A)
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        corr.append(Loss.item())
        if i % 1000 == 0:
            U,S,D = torch.linalg.svd(s_A, full_matrices=False)
            S = S.cpu().detach()
            for i in range(len(S)):
                k = torch.sum(S[:i])
                if k / torch.sum(S) > 0.99:
                    print(i)
                    rank.append(i)
                    break
            print("Loss:{},rank:{},real_rank:{}".format(Loss.item(),torch.linalg.matrix_rank(A),torch.linalg.matrix_rank(s_A)))
    width = 192
    time_embed = False
    trained_with_fmnist = True      # pure gt / gt+fashionmnist
    
    
    prefix_model_name = "conditional_ddpm_width"
    suffix_model_name = "_fmnist_simulatefmnist_random.pt"
    store_path = prefix_model_name + str(width) + suffix_model_name
    print("Loading model: ", store_path)
    
    n_steps, min_beta, max_beta = 1000, 10**-4, 0.02  
    unet_structure = MySigUNet(n_steps)
    
    best_model = MyDDPM(
        unet_structure,
        n_steps=n_steps,
        min_beta=min_beta,
        max_beta=max_beta,
        device=device,
    )
    #best_model.load_state_dict(torch.load(store_path, map_location=device))
    #best_model.eval()
    print("Model loaded")
    
    Runtime = "_MMwave_"+str(int(time.time()))
    
    '''
    iteration to optimize metasurface
    '''
    lr = 1e-2
    optimizer = torch.optim.Adam([
                       {'params': simulator.w,'lr': lr},
                       {'params': simulator.meta_theta, 'lr': lr},
                       
                        ], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
    optimize_step(optimizer,Runtime)
    
    '''
    iteration to optimize diffusion
    '''
    lr = 0.001
    optimizer = torch.optim.Adam([
                       {'params': best_model.parameters(),'lr': 0.001},
                       
                        ], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
    optimize_step(optimizer,Runtime)
    
    '''
    output optimized phasemap
    ''' 
    print(simulator.meta_theta)
        
    '''
    evaluate optimization
    ''' 
    criterion=nn.MSELoss()
    mse_loss = []
    with torch.no_grad():
        for T in tqdm(loader_test):
            T = T.to(device)
            n=T.shape[0]
            T = T.transpose(1,0)
            A,mic_data,_,_ = simulator(T)
            A=A.permute(1,0).unsqueeze(0).repeat(n,1,1)
            mic_data=mic_data.permute(1,2,0)
            y = torch.concat([A,mic_data],dim=1)
            print(y.shape)
            Tpred = new_ddpm_sample(y, best_model, ddpm_steps=600)
            print(Tpred.shape)
            loss = criterion(Tpred.reshape(n,-1), T.permute(1,0))
            mse_loss.append(loss.cpu())
            for i in range(16):
                plt.imshow(Tpred[i,:].reshape(20,20).cpu().detach().clone())
                plt.colorbar()
                plt.show()
                plt.imshow(T[:,i].reshape(20,20).cpu().detach().clone())
                plt.colorbar()
                plt.show()
            break
    print('mean RMSE',np.mean(np.asarray(mse_loss)))
   