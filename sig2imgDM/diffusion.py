import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


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
    
def ddpm_sample(
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

        # Starting from random noise
        x = torch.randn(1, c, h, w).to(device)

        factor = ddpm.n_steps // ddpm_steps
        ddim_timestep_seq = np.asarray(list(range(0, ddpm.n_steps, factor)))
        # ddim_timestep_seq = ddim_timestep_seq + 1

        # for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
        for i in tqdm(reversed(range(0, ddpm_steps)), desc='sampling loop time step', total=ddpm_steps):
            time_tensor = (torch.ones(1, 1) * ddim_timestep_seq[i]).to(device).long()
            
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
    
def ddim_sample(
    diffusion,
    y,
    batch_size,
    image_size = 20,
    channels = 1,
    ddim_timesteps=10,
    diffusion_steps = 1000,
    ddim_eta=0.0,
    clip_denoised=True,
    device=None):

    c = diffusion_steps // ddim_timesteps
    ddim_timestep_seq = np.asarray(list(range(0, diffusion_steps, c)))
    ddim_timestep_seq = ddim_timestep_seq + 1
    ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
    sample_img = torch.randn((batch_size, channels, image_size, image_size), device=device)
#     model_kwargs = {"low_res": pred.to(device)}
    for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
        t = torch.full((batch_size,1), ddim_timestep_seq[i], device=device, dtype=torch.long)
        prev_t = torch.full((batch_size,1), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)

        # 1. get current and previous alpha_cumprod
        # alpha_cumprod_t =  torch.tensor(diffusion.alphas_cumprod[ddim_timestep_seq[i]]).to(device) 
        # alpha_cumprod_t_prev = torch.tensor(diffusion.alphas_cumprod_prev[ddim_timestep_prev_seq[i]]).to(device)
        alpha_cumprod_t = diffusion.alpha_bars[t]
        alpha_cumprod_t_prev = diffusion.alpha_bars[prev_t] if t > 0 else diffusion.alphas[0]
        
        # 2. predict noise using model
        pred_noise, _ = diffusion.network(sample_img, y, t)

        # 3. get the predicted x_0
        pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
        
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)

        # 4. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        sigmas_t = ddim_eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

        # 5. compute "direction pointing to x_t" of formula (12)
        pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise

        # 6. compute x_{t-1} of formula (12)
        x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)

        sample_img = x_prev

    sample_img -= torch.min(sample_img)
    sample_img *= 1 / torch.max(sample_img)

    return sample_img
