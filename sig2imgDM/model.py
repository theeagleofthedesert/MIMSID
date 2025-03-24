# Import of libraries
import numpy as np
from argparse import ArgumentParser
import scipy.io
import os
from collections import OrderedDict

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Subset

from torchvision.transforms import Compose, ToTensor, Lambda, Resize, Pad
from torchvision.datasets.mnist import MNIST, FashionMNIST

feature_dim=400
width = 192

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

# width = 72*5
# width = 60
# width = 384
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
    
class signalTransformerEmbedT(nn.Module):
    def __init__(self,num = 5, time_emb_dim = 100):
        super().__init__()
        self.num = num
        self.transformer = Transformer(
            # width=72*self.num,          # Three test: rwidth = 72*5/30*5/384
            width = width,
            layers=2,
            heads=8,
            # heads=10,        # only for width = 60
            # attn_mask=build_attention_mask(32)
        )
        self.ln_final = LayerNorm(width)
        self.final_conv = nn.Conv1d(int(11520/width)+1,1,3,1,1)

        self.make_te = nn.Linear(time_emb_dim, width)
        self.fc = nn.Linear(width,400)

    def forward(self,x,t):
       
        x = x.reshape(x.shape[0],2, 16, 5, 12, 6)    # 1,32,360  B,w,h
        x = x.reshape(x.shape[0],int(11520/width),width)    # 1,32,360  B,w,h
        te = self.make_te(t)       # te: 1,width
        x = torch.concat([x,te], dim=1) # x: 1, int(11520/width)+1, 
        # x = x.permute(1,0,2)        # 32,1,360  # w,B,h
        
        x = self.transformer(x)
        # x = x.permute(1,0,2)
        x = self.ln_final(x)
        x = self.final_conv(x)
        return self.fc(x.squeeze(1))
    

class signalTransformerEmbedTPro(nn.Module):
    def __init__(self,num = 5, time_emb_dim = 100, time_emb_factor = 10):
        super().__init__()
        if time_emb_dim % time_emb_factor != 0:
            assert("Factor should be able to divdie time_emb_dim!")
        self.num = num
        self.time_emb_dim = time_emb_dim
        self.time_emb_factor = time_emb_factor
        self.time_emb_din = time_emb_dim//time_emb_factor
        self.transformer = Transformer(
            # width=72*self.num,          # Three test: rwidth = 72*5/30*5/384
            width = width,
            layers=2,
            heads=8,
            # heads=10,        # only for width = 60
            # attn_mask=build_attention_mask(32)
        )
        self.ln_final = LayerNorm(width)
        self.final_conv = nn.Conv1d(int(11520/width)+time_emb_factor,1,3,1,1)

        self.make_te = nn.ModuleList([nn.Linear(self.time_emb_din, width) for _ in range(time_emb_factor)])

        self.fc = nn.Linear(width,400)

    def forward(self,x,t):
       
        x = x.reshape(x.shape[0],2, 16, 5, 12, 6)    # 1,32,360  B,w,h
        x = x.reshape(x.shape[0],int(11520/width),width)    # 1,32,360  B,w,h
        # te = self.make_te(t)       # te: 1,width
        te = torch.zeros(x.shape[0], self.time_emb_factor, width).to(x.device)   # t: B,1,100
        for i, l in enumerate(self.make_te):
            te[:,i,:] = l(t[:,0,self.time_emb_din*i:self.time_emb_din*(i+1)])   # te: B,10,192
        x = torch.concat([x,te], dim=1) # x: 1, int(11520/width)+1, 
        # x = x.permute(1,0,2)        # 32,1,360  # w,B,h
        
        x = self.transformer(x)
        # x = x.permute(1,0,2)
        x = self.ln_final(x)
        x = self.final_conv(x)
        return self.fc(x.squeeze(1))
    
class signalTransformer_A(nn.Module):
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
        self.final_conv = nn.Conv1d(int(11520/width),1,3,1,1)
        self.fc = nn.Linear(width,400)
    def forward(self,x):
        #input A(N,400,11520)
        x = x.reshape(x.shape[0],feature_dim, 11520)
        #x = x.reshape(x.shape[0],2, 16, 5, 12, 6)    # 1,32,360  B,w,h
        #x = x.permute(0,1,3,5,2,4)
        x = self.A_conv(x).reshape(x.shape[0],2, 16, 5, 12, 6)
        x = x.reshape(x.shape[0],int(11520/width),width)    # 1,32,360  B,w,h
        # x = x.permute(1,0,2)        # 32,1,360  # w,B,h
        x = self.transformer(x)
        # x = x.permute(1,0,2)
        x = self.ln_final(x)
        x = self.final_conv(x)
        
        return self.fc(x.squeeze(1))
    
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
        self.signal_transformer_A = signalTransformer_A(num=5)
        self.signal_transformer_embed_t = signalTransformerEmbedT(num=5, time_emb_dim=time_emb_dim)
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
    


class MySigUNetEmbedY(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(MySigUNetEmbedY, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.signal_transformer = signalTransformer(num=5)
        self.signal_transformer_embed_t = signalTransformerEmbedT(num=5, time_emb_dim=time_emb_dim)
        self.te0 = self._make_te(time_emb_dim, 1)

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

        # out_y = self.signal_transformer(y + self.te0(t).reshape(n, -1, 1)).reshape(n,1,20,20)       # add time embedding to y

        # out_y = out_y + self.te0(t).reshape(n, -1, 1, 1)           # add time embedding to out_y

        out_y = self.signal_transformer_embed_t(y, t).reshape(n,1,20,20)    # concat time embeding to y

        # out_y = self.signal_transformer(y).reshape(n,1,20,20)       # no time embedding to y
        


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
    
class MySigUNetEmbedYPro(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(MySigUNetEmbedYPro, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.signal_transformer = signalTransformerEmbedTPro(num=5, time_emb_dim=time_emb_dim, time_emb_factor=10)
        self.te0 = self._make_te(time_emb_dim, 1)

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

        out_y = self.signal_transformer(y, t).reshape(n,1,20,20) 

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