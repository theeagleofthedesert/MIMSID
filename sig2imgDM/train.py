from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn as nn
from tqdm import tqdm
from metric import SSIM
import random
from channel_adapt import A_noise


def training_loop(
    ddpm, loader, A, n_epochs, optim, device, display=False, store_path="ddpm_model.pt"
):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps
    ssim = SSIM()
    max_grad_norm = 10
    '''
    A_shifted_set=[]
    
    for A in A_set:
        A = A.permute(1,0)
        A = A[None, :, :]
        A_shifted=A_noise(A,5,1.2)
        A_shifted_set.append(A_shifted)
    '''
    A = A.permute(1,0)
    A = A[None, :, :]
    A_shifted=A_noise(A,5,1.2)
    
    # TODO；
    # A_shifted[20]

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(
            tqdm(
                loader,
                leave=False,
                desc=f"Epoch {epoch + 1}/{n_epochs}",
                colour="#005500",
            )
        ):

            # Loading data
            x0 = batch[0].to(device)
            n = len(x0)

            # apply random mask
            x0_gt=x0.clone()
            '''
            mask_flag=random.random()
            if mask_flag>=0.6:
                mask_size=random.randint(8,12)
                x0=random_mask(x0,mask_size)
                '''
            #for A_shifted in A_shifted_set:
            # random_A_idx = torch.
            #A=random.choice(A_shifted)
            #A=A_shifted_set[step%30][0]
            
            A=random.choice(A_shifted)

            # apply random A masak
            '''
            mask_flag=random.random()
            if mask_flag>=0.5:
                mask_size=random.randint(5,15)
                A=A_random_mask(A,mask_size)
            '''
            A=A.repeat(n,1,1)

            y0 = x0.reshape(n, 1,-1) @ A    #TODO: A_new: 20种不同的A （5种相位漂移的A，5种幅度漂移的A，5种gaussian noise的A，5种幅度/相位的A）
            #print(y0.shape)
            #print(A.shape)
            y0 = torch.concat([A,y0],dim=1)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(x0_gt, t, eta)

            # Getting model estimation of noise based on the images and the time-step
            eta_theta,_ = ddpm.backward(noisy_imgs, y0, t.reshape(n, -1))

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta, eta)
            #ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
            #loss = 1 - ssim_loss(eta_theta, eta)
            #loss = -psnr(eta_theta, eta)

            optim.zero_grad()
            loss.backward()

            #clip_grad_norm_(ddpm.parameters(), max_grad_norm)
            optim.step()
            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)

