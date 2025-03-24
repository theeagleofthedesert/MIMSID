import scipy.constants as C
import torch
import torch.nn as nn
import numpy as np
import random


def A_distance_phasediff(d, device=None):
    metasurface_len=20
    metasurface_space=1
    f=79e9
    start_point_pos=torch.tensor([-30,0,0],dtype=torch.float32, device=device)
    start_point_pos=torch.unsqueeze(start_point_pos,dim=0)

    metasurface_position_y = torch.linspace(
        (metasurface_len-1)*metasurface_space/2, -(metasurface_len-1)*metasurface_space/2,  metasurface_len, dtype=torch.float32, device=device).expand(metasurface_len, metasurface_len)
    metasurface_position_z = torch.linspace(
        (metasurface_len-1)*metasurface_space/2, -(metasurface_len-1)*metasurface_space/2,  metasurface_len, dtype=torch.float32, device=device).expand(metasurface_len, metasurface_len).transpose(0, 1)
    metasurface_position = torch.stack((torch.zeros(metasurface_len, metasurface_len, dtype=torch.float32, device=device),
                                        metasurface_position_y,
                                        metasurface_position_z),
                                       dim=-1)

    metasurface_position = torch.flatten(
        metasurface_position, start_dim=0, end_dim=1)

    metasurface2_position = metasurface_position.clone()
    metasurface2_position[:, 0] = d

    dist_func = nn.PairwiseDistance(p=2)

    dist_metasurface = dist_func(
        metasurface_position.unsqueeze(0).expand(1, -1, -1),
        start_point_pos.unsqueeze(1).expand(-1, metasurface_len**2,  -1),
    )

    dist_metasurface2 = dist_func(
        metasurface2_position.unsqueeze(0).expand(1, -1, -1),
        start_point_pos.unsqueeze(1).expand(-1, metasurface_len**2,  -1),
    )

    h1=torch.exp(-1j*2*torch.pi*f*dist_metasurface/C.c)
    h2=torch.exp(-1j*2*torch.pi*f*dist_metasurface2/C.c)

    phase1=torch.angle(h1)
    phase2=torch.angle(h2)

    phase_diff=phase2-phase1
    
    return phase_diff

def A_noise(A,k,w,device=None, mode=[]):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    A=A.cpu()
    A_len=A.shape[-1]
    A_real=A[:,:,:int(A_len/2)]
    A_img=A[:,:,int(A_len/2):]
    A_complex=A_real+1j*A_img
    A_shifted=[]
    
    A_shifted.append(A.to(device).float())
    for m in mode:
        if m == 'phase':
            #k random phase shift
            for i in range(k):
                np.random.seed(k)
                rand_phase_mat=np.random.randn(A.shape[0],A.shape[1],int(A_len/2))*w
                A_shift=A_complex*np.exp(-1j*rand_phase_mat)
                A_shift = torch.concat([A_shift.real,A_shift.imag],dim=-1)
                A_shifted.append(A_shift.to(device).float())

        if m == 'amp':
            #k random amp shift
            for i in range(k):
                np.random.seed(k)
                rand_amp_mat=np.random.rand(A.shape[0],A.shape[1],int(A_len/2))*w+0.5
                A_shift=A_complex*rand_amp_mat
                A_shift = torch.concat([A_shift.real,A_shift.imag],dim=-1)
                A_shifted.append(A_shift.to(device).float())

        if m == 'noise':
            #k guassian noise
            for i in range(k):
                np.random.seed(k)
                guassian_real=np.random.randn(A.shape[0],A.shape[1],int(A_len/2))*w
                guassian_img=np.random.randn(A.shape[0],A.shape[1],int(A_len/2))*w
                A_shift=A_complex+guassian_real+1j*guassian_img
                A_shift = torch.concat([A_shift.real,A_shift.imag],dim=-1)
                A_shifted.append(A_shift.to(device).float())

        if m == 'dist':
            #different distance A phase change
            for d in range(1,k+1):
                phase_diff=A_distance_phasediff(d, device)
                phase_diff_mat=phase_diff.unsqueeze(-1).expand(-1,-1,int(A_len/2))
                A_shift=A_complex*np.exp(-1j*phase_diff_mat)
                A_shift=A_shift*np.exp(-1j*phase_diff_mat)
                A_shift = torch.concat([A_shift.real,A_shift.imag],dim=-1)
                A_shifted.append(A_shift.to(device).float())
    return A_shifted

def random_mask(x,mask_size, device=None):
    
    reflection_rate=random.uniform(0.1,1)
    
    mask_block=torch.ones((mask_size,mask_size))*reflection_rate
    mask=torch.ones((20,20))
    x_pos=random.randint(0, 20-mask_size)
    y_pos=random.randint(0, 20-mask_size)
    
    mask[x_pos:x_pos+mask_size,y_pos:y_pos+mask_size]=mask_block
    mask=mask.to(device).float()
    x_masked=x*mask
    
    return x_masked

def A_random_mask(A,mask_size, device=None):
    x_pos=random.randint(0, 20-mask_size)
    y_pos=random.randint(0, 20-mask_size)
    A=A.cpu()
    A_len=A.shape[-1]
    
    Amask=A[x_pos:x_pos+mask_size,y_pos:y_pos+mask_size,:]
    Amask_real=Amask[:,:,:int(A_len/2)]
    Amask_img=Amask[:,:,int(A_len/2):]
    Amask_complex=Amask_real+1j*Amask_img
    reflection_phaseshift=random.uniform(-3.14,3.14)
    reflection_rate=random.uniform(0.1,1)
    Amask_shift=Amask_complex*np.exp(-1j*reflection_phaseshift)*reflection_rate
    Amask_shift = torch.concat([Amask_shift.real,Amask_shift.imag],dim=-1)
    A[x_pos:x_pos+mask_size,y_pos:y_pos+mask_size,:]=Amask_shift
    
    A_masked=A.to(device).float()
    
    return A_masked
    
