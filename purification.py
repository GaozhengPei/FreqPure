import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from utils import diff2clf, clf2diff, normalize
import random

def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(
        beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
    )
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas).float()


class PurificationForward(torch.nn.Module):
    def __init__(self, clf, diffusion, max_timestep, attack_steps, sampling_method, is_imagenet, device,amplitude_cut_range,phase_cut_range,delta,forward_noise_steps):
        super().__init__()
        self.clf = clf
        self.diffusion = diffusion
        self.device = device
        self.betas = get_beta_schedule(1e-4, 2e-2, 1000).to(device)
        self.max_timestep = max_timestep
        self.attack_steps = attack_steps
        self.sampling_method = sampling_method
        self.amplitude_cut_range = amplitude_cut_range
        self.phase_cut_range = phase_cut_range
        self.delta=delta
        assert sampling_method in ['ddim', 'ddpm']
        if self.sampling_method == 'ddim':
            self.eta = 0
        elif self.sampling_method == 'ddpm':
            self.eta = 1
        self.is_imagenet = is_imagenet
        self.forward_noise_steps = forward_noise_steps

    def compute_alpha(self, t):
        beta = torch.cat(
            [torch.zeros(1).to(self.betas.device), self.betas], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def get_noised_x(self, x, t):
        e = torch.randn_like(x)
        if type(t) == int:
            t = (torch.ones(x.shape[0]) * t).to(x.device).long()
        a = (1 - self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = x * a.sqrt() + e * (1.0 - a).sqrt()
        return x

    def denoising_process(self,ori_x, x, seq):
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xt = x
        for i, j in zip(reversed(seq), reversed(seq_next)):

            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(t.long())
            at_next = self.compute_alpha(next_t.long())
            et = self.diffusion(xt, t)
            if self.is_imagenet:
                et, _ = torch.split(et, 3, dim=1)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_t = self.amplitude_phase_exchange_torch(ori_x,x0_t)
            c1 = (
                self.eta * ((1 - at / at_next) *
                            (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        return xt

    def preprocess(self, x):
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        for i in range(len(self.max_timestep)):
            noised_x = self.get_noised_x(x_diff, self.max_timestep[i])
            x_diff = self.denoising_process(noised_x, self.attack_steps[i])

        x_clf = diff2clf(x_diff)
        return x_clf

    def classify(self, x):
        logits = self.clf(x)
        return logits

    def forward(self, x):
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        for i in range(len(self.max_timestep)):
            noised_x = self.get_noised_x(x_diff, self.max_timestep[i])
            x_diff = self.denoising_process(x_diff,noised_x, self.attack_steps[i])

        # classifier part
        if self.is_imagenet:
            x_clf = normalize(diff2clf(F.interpolate(x_diff, size=(
                224, 224), mode='bilinear', align_corners=False)))
        else:
            x_clf = diff2clf(x_diff)
        logits = self.clf(x_clf)
        return logits
    

    def compute_fft(self,image):  
        amplitude_channels = []  
        phase_channels = []  

        for channel in range(3):  
            f = torch.fft.fft2(image[channel, :, :])  
            fshift = torch.fft.fftshift(f)  
            amplitude = torch.abs(fshift)  
            amplitude_channels.append(amplitude)  
            phase = torch.angle(fshift)  
            phase_channels.append(phase + torch.pi)  

        return amplitude_channels, phase_channels

    def low_pass_exchange(self,amplitude_channels, amplitude_channels_0_t):
        filtered_amplitude_channels = []
        for i in range(3):
            rows, cols = amplitude_channels[i].shape
            u = np.arange(-cols // 2, cols // 2)
            v = np.arange(-rows // 2, rows // 2)
            U, V = np.meshgrid(u, v)
            frequency_map = np.sqrt(U ** 2 + V ** 2)
            low_frequency = (frequency_map <= self.amplitude_cut_range)
            low_frequency = torch.from_numpy(low_frequency).to(self.device)
            amplitude_channels_0_t[i] = torch.where(low_frequency, amplitude_channels[i], amplitude_channels_0_t[i])
            filtered_amplitude_channels.append(amplitude_channels_0_t[i])
        return filtered_amplitude_channels


    def phase_low_pass_exchange(self,phase_channels, phase_channels_0_t):
        filtered_amplitude_channels = []
        for i in range(3):
            rows, cols = phase_channels[i].shape
            u = np.arange(-cols // 2, cols // 2)
            v = np.arange(-rows // 2, rows // 2)
            U, V = np.meshgrid(u, v)
            frequency_map = np.sqrt(U ** 2 + V ** 2)
            low_frequency = (frequency_map <= self.phase_cut_range)
            low_frequency = torch.from_numpy(low_frequency).to(self.device)
            phase_channels_0_t[i] = torch.where(low_frequency, phase_channels[i], phase_channels_0_t[i])
            phase_channels_0_t[i][low_frequency] = torch.clip(phase_channels_0_t[i][low_frequency],phase_channels[i][low_frequency]-self.delta,phase_channels[i][low_frequency]+self.delta)
            filtered_amplitude_channels.append(phase_channels_0_t[i])
        return filtered_amplitude_channels
    

    
    def phase_exchange(self,phase_channels,phase_channels_0_t):
        exchanged_phase_channels = []
        for i in range(3):
            rows, cols = phase_channels[i].shape
            exchange_matrix = self.generate_frequency_exchange_matrix(rows, cols)
            phase_channels_0_t[i][exchange_matrix] = phase_channels[i][exchange_matrix]
            exchanged_phase_channels.append(phase_channels_0_t[i])
        return exchanged_phase_channels
    
    def phase_clip(self,phase_channels,phase_channels_0_t,delta=0.6):
        phase_channels_clip=[]
        for i in range(3):
            phase_channels_clip.append(np.clip(phase_channels_0_t[i],phase_channels[i]-delta,phase_channels[i]+delta))
        return phase_channels_clip

    
    def reconstruct_image(self,filtered_amplitude_channels, phase_channels):
        reconstructed_image = []
        for channel in range(3):
            amplitude = filtered_amplitude_channels[channel]
            phase = phase_channels[channel]-torch.pi
            fshift_filtered = amplitude * torch.exp(1j * phase)
            f_ishift = torch.fft.ifftshift(fshift_filtered)
            img_reconstructed = torch.fft.ifft2(f_ishift)
            img_reconstructed = torch.abs(img_reconstructed)
            img_reconstructed = torch.clip(img_reconstructed,0,255)
            reconstructed_image.append(img_reconstructed/255)
        return torch.stack(reconstructed_image,dim=2)

        
    
    def amplitude_phase_exchange_torch(self,x,x_0_t):
        x_t = self.get_noised_x(x, self.forward_noise_steps)
        t = (torch.ones(x.size(0)) * self.forward_noise_steps).to(x.device)
        at = self.compute_alpha(t.long())
        et = self.diffusion(x_t, t)
        x =  (x_t - et * (1 - at).sqrt()) / at.sqrt()
        # save_image(diff2clf(x), 'new_x_0_t.png')
        x = torch.clip((diff2clf(x)* 255),0,255)
        x_0_t = torch.clip((diff2clf(x_0_t)* 255),0,255)

        batch,channel,height,width = x.shape
        new_x_0_t = torch.zeros(size=(batch,height,width,channel))
        for batch_idx in range(batch):

            amplitude_channels, phase_channels = self.compute_fft(x[batch_idx])

            amplitude_channels_0_t, phase_channels_0_t = self.compute_fft(x_0_t[batch_idx])

            amplitude_channels_0_t_exchange = self.low_pass_exchange(amplitude_channels,amplitude_channels_0_t)
            
            phase_channels_0_t_exchange = self.phase_low_pass_exchange(phase_channels,phase_channels_0_t)
            reconstructed_image = self.reconstruct_image(amplitude_channels_0_t_exchange,phase_channels_0_t_exchange)

            new_x_0_t[batch_idx] = reconstructed_image
        new_x_0_t = new_x_0_t.float().permute(0,3,1,2).to(self.device)
        new_x_0_t = clf2diff(new_x_0_t)
        return new_x_0_t

    def get_img_logits(self, x):

        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        for i in range(len(self.max_timestep)):
            noised_x = self.get_noised_x(x_diff, self.max_timestep[i])
            x_diff = self.denoising_process(x_diff,noised_x, self.attack_steps[i])

        # classifier part
        if self.is_imagenet:
            x_clf = normalize(diff2clf(F.interpolate(x_diff, size=(
                224, 224), mode='bilinear', align_corners=False)))
        x_clf = diff2clf(x_diff)
        logits = self.clf(x_clf)
        return x_clf,logits