# Import of libraries
import random
import imageio
import numpy as np
import scipy.io
import time
import argparse

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import cv2
import skimage
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset,  Subset

from torchvision.transforms import Compose, ToTensor, Lambda, Resize, Pad
from torchvision.datasets.mnist import  FashionMNIST

from model import *
from dataset import *
from diffusion import *
from channel_adapt import *
from metric import *
from train import *

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Sig2Img DM')
    parser.add_argument('--if_Train', type=bool, default=False, help='whether to train the model')
    parser.add_argument('--if_Eval', type=bool, default=True, help='whether to test the model')
    args = parser.parse_args()

    # Getting device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using device: {device}\t"
        + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU")
    )
    
    if_Train = args.if_Train
    if_Eval = args.if_Eval
    
    ### Load Dataset
    A_path = r'datasets/data1111/calibration'
    Y_path = r'datasets/data1111/baseline'
    gt_path = r'datasets/gt_20x20'

    dataset_sample = MyDataset(A_path, Y_path, gt_path)
    ds_fn = FashionMNIST
    dataset_fmnist = ds_fn("./data", download=True, train=True, transform=Compose([ToTensor(), 
                        Lambda(lambda x: (x - 0.5) * 2),
                        Resize(24),
                        Pad(5, fill=-1, padding_mode='constant'),
                        Resize(20),
                        ]))
    dataset_number = GTDataset(gt_path)



    dataset_fmnist = Subset(dataset_fmnist, np.arange(5000))
    dataset = ConcatDataset([dataset_fmnist, dataset_number])

    print(len(dataset))

    loader = DataLoader(dataset, 45, shuffle=True)



    A=dataset_sample.get_A().to(device).float()
    A = A.permute(1,0)
    A = A[None, :, :]
    A_shifted=A_noise(A,5,1.2, device=device, mode=[])
    
    
    if if_Train:
        
    
    
        #### Training Progress
        no_train = False
        n_epochs = 500
        lr = 0.001
    
        n_steps, min_beta, max_beta = 1000, 10**-4, 0.02  # Originally used by the authors
    
        # temp
        ddpm = MyDDPM(
            MySigUNet(n_steps),     # normal case 
            # MySigUNetEmbedY(n_steps),     # t_embed case
            # MySigUNetEmbedYPro(n_steps),            # t_embed with 10x layer
            n_steps=n_steps,
            min_beta=min_beta,
            max_beta=max_beta,
            device=device,
        )
        sum([p.numel() for p in ddpm.parameters()])
        # Training
        store_path = "conditional_ddpm_width192_fmnist_Ay_train.pt"

        if not no_train:
            training_loop(
                ddpm,
                loader,
                dataset_sample.get_A().to(device).float(),
                n_epochs,
                optim=Adam(ddpm.parameters(), lr),
                device=device,
                store_path=store_path,
            )
    
    if if_Eval:
        ## Load Model
        width = 192
        time_embed = False
        trained_with_fmnist = True      # pure gt / gt+fashionmnist
    
    
        prefix_model_name = "conditional_ddpm_width"
        suffix_model_name = "_fmnist_Ay_train.pt" if if_Train else "_fmnist_Ay.pt"
        store_path = prefix_model_name + str(width) + suffix_model_name
        print("Loading model: ", store_path)
    
        n_steps, min_beta, max_beta = 1000, 10**-4, 0.02  # Originally used by the authors
        unet_structure = MySigUNetEmbedYPro(n_steps) if time_embed else MySigUNet(n_steps)
    
        best_model = MyDDPM(
            unet_structure,
            n_steps=n_steps,
            min_beta=min_beta,
            max_beta=max_beta,
            device=device,
        )
        best_model.load_state_dict(torch.load(store_path, map_location=device))
        best_model.eval()
        print("Model loaded")
    
    
        # Evaluation
       
        # Evaluation with real Y
        real_loader = DataLoader(dataset_sample, 1, shuffle=False)
        A = dataset_sample.get_A().to(device).float()
        A=A.permute(1,0)
        sample_num = 9
        sample_nums = sample_num
        psnr_sum = 0
        i=0
        mse = nn.MSELoss()
        mses=[]
        t_cost=[]
    
        for data in tqdm(real_loader):
            y,gt = data
            y,gt = y.to(device).float(),  gt.to(device).float()
            gt = gt.reshape(1,1,20,20)
            #y = torch.concat([A,y],dim=0).unsqueeze(0)
            t_0=time.time()
            sample_img = ddim_sample(best_model,y,
                                1,
                                image_size=20,
                                channels = 1,
                                ddim_timesteps= 10,   #
                                diffusion_steps = 1000,
                                ddim_eta=0.01,    # hyperparameter
                                clip_denoised=True, device=device)
            t_1=time.time()
            consume=t_1-t_0
            t_cost.append(consume)
            plt.figure()
            plt.subplot(121)
            plt.imshow(sample_img[0,0].detach().cpu().numpy())
            plt.subplot(122)
            plt.imshow(gt[0][0].detach().cpu().numpy())
            sample_num -= 1
            psnr_sum += psnr(sample_img,gt)
            loss=mse(sample_img, gt)
            print(loss)
            mses.append(torch.sqrt(loss))
            plt.imsave('output/'+str(i)+'_est.png',sample_img[0,0].detach().cpu().numpy())
            plt.imsave('output/'+str(i)+'_gt.png',gt[0,0].detach().cpu().numpy())

            i+=1
    
            #if sample_num ==0:
            #    break
            
        print('\n############################ result ############################')
        print("PSNR",psnr_sum/sample_nums)
        print("mean RMSE",sum(mses)/len(mses))
        print("mean infer time",sum(t_cost)/len(mses))

