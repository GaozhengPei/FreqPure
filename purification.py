import numpy as np
import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchvision
from utils import diff2clf, clf2diff, normalize
from PIL import Image
from tqdm import *
import math
from torchvision import transforms
import cv2 
import numpy as np

def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(
        beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
    )
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a).float().to(device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,)+(1,)*(len(x_shape)-1))
    return out


def save_image(x):
    """
    x:[B,C,H,W]
    [-1,1]
    """
    x = (x + 1)/2
    x = (x[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(x).save('./image_noise.png')

def get_ddim_steps(total_time_steps,sample_steps,strength):
    step = total_time_steps//sample_steps
    ddim_steps = np.arange(0,total_time_steps-1,step)
    ddim_steps = ddim_steps[:int(sample_steps*strength)+1]
    ddim_steps = np.flip(ddim_steps)
    return ddim_steps


class PurificationForward(torch.nn.Module):
    def __init__(self, clf, diffusion,strength, is_imagenet,ddim_steps,amplitude_cut_range,phase_cut_range,delta,device):
        super().__init__()
        self.clf = clf
        self.diffusion = diffusion
        self.device = device
        self.is_imagenet = is_imagenet
        self.strength = strength
        self.amplitude_cut_range = amplitude_cut_range
        self.phase_cut_range = phase_cut_range
        self.delta=delta
        self.num_train_timesteps = 1000
        self.sample_steps = ddim_steps
        self.timesteps = get_ddim_steps(self.num_train_timesteps,self.sample_steps,self.strength)
        self.eta=0.0

        self.betas = get_beta_schedule(1e-4, 2e-2, 1000)
        self.alphas = 1. - self.betas
        self.sqrt_alphas = np.sqrt(self.alphas) 
        self.sqrt_one_minus_alphas = np.sqrt(1. - self.alphas)
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)# \bar{alpha}_t
        self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod - 1)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1]) # \bar{alpha}_t-1
        self.posterior_mean_coef1 = self.betas * np.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        self.posterior_variance = self.betas*(1.0-self.alphas_cumprod_prev) / (1.0-self.alphas_cumprod) # \tilde{beta}_t
        # self.logvar = np.log(np.maximum(self.posterior_variance, 1e-20))




    def diffuse_t_steps(self, x0, t):
        alpha_bar = self.alphas_cumprod[t]
        xt = torch.sqrt(torch.tensor(alpha_bar)) * x0 + torch.sqrt(torch.tensor(1-alpha_bar)) * torch.randn_like(x0)
        return xt

    def diffuse_one_step(self,x,t):
        noise = torch.randn_like(x)
        return (
        extract(self.sqrt_alphas, t, x.shape) * x +
        extract(self.sqrt_one_minus_alphas, t, x.shape) * noise
    )



    def diffuse_one_step_from_now(self,x_t,t,steps):
        n = x_t.shape[0]
        for i in range(steps):
            x_t = self.diffuse_one_step(x_t, (torch.ones(n)*(t+i+1)).to(self.device))
        return x_t,t+steps



    def denoising_step(self,x, t):
        """
        Sample from p(x_{t-1} | x_t)
        """
        # instead of using eq. (11) directly, follow original implementation which,
        # equivalently, predicts x_0 and uses it to compute mean of the posterior
        t = (torch.ones(x.shape[0])*t).to(self.device)
        # 1. predict eps via model
        model_output = self.diffusion(x, t)
        # 2. predict clipped x_0
        # (follows from x_t=sqrt_alpha_cumprod*x_0 + sqrt_one_minus_alpha*eps)
        pred_xstart = (extract(self.sqrt_recip_alphas_cumprod, t, x.shape)*x -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)*model_output)
        pred_xstart = torch.clamp(pred_xstart, -1, 1)
        # 3. compute mean of q(x_{t-1} | x_t, x_0) (eq. (6))
        mean = (extract(self.posterior_mean_coef1, t, x.shape)*pred_xstart +
                extract(self.posterior_mean_coef2, t, x.shape)*x)

        posterior_variance = extract(self.posterior_variance, t, x.shape)
        
        # sample - return mean for t==0
        noise = torch.randn_like(x)
        mask = 1-(t==0).float()
        mask = mask.reshape((x.shape[0],)+(1,)*(len(x.shape)-1))
        sample = mean + mask*torch.sqrt(posterior_variance)*noise
        sample = sample.float()

        return pred_xstart,sample

    def compute_fft(self,image):  
        """对彩色图像的每个通道进行傅里叶变换，计算振幅和相位"""  
        amplitude_channels = []  
        phase_channels = []  

        for channel in range(3):  
            # 使用 torch.fft.fft2 进行二维傅里叶变换  
            f = torch.fft.fft2(image[channel, :, :])  
            
            # 使用 torch.fft.fftshift 进行频谱移位  
            fshift = torch.fft.fftshift(f)  
            
            # 计算振幅  
            amplitude = torch.abs(fshift)  
            amplitude_channels.append(amplitude)  
            
            # 计算相位并加上 π  
            phase = torch.angle(fshift)  
            phase_channels.append(phase + torch.pi)  

        return amplitude_channels, phase_channels

    def low_pass_exchange(self,amplitude_channels, amplitude_channels_0_t):
        filtered_amplitude_channels = []
        """对振幅进行低通滤波，保留给定频率以下的成分"""
        for i in range(3):
            rows, cols = amplitude_channels[i].shape
            # 计算频率映射
            u = np.arange(-cols // 2, cols // 2)
            v = np.arange(-rows // 2, rows // 2)
            U, V = np.meshgrid(u, v)
            frequency_map = np.sqrt(U ** 2 + V ** 2)
            low_frequency = (frequency_map <= self.amplitude_cut_range)
            low_frequency = torch.from_numpy(low_frequency).to(self.device)
            # 应用低通滤波器
            amplitude_channels_0_t[i][low_frequency] = amplitude_channels[i][low_frequency]
            filtered_amplitude_channels.append(amplitude_channels_0_t[i])
        return filtered_amplitude_channels

    # def generate_frequency_exchange_matrix(self,rows, cols):
    #     matrix = np.zeros((rows, cols), dtype=bool)
    #     for idx in range(rows*cols):
    #         row = idx // rows  # 计算行索引
    #         col = idx % cols   # 计算列索引
    #         if int(self.ratio * idx) == idx:
    #             print(idx)
    #             matrix[row, col] = True
    #     return matrix
    def phase_low_pass_exchange(self,phase_channels, phase_channels_0_t):
        filtered_amplitude_channels = []
        """对振幅进行低通滤波，保留给定频率以下的成分"""
        for i in range(3):
            rows, cols = phase_channels[i].shape
            # 计算频率映射
            u = np.arange(-cols // 2, cols // 2)
            v = np.arange(-rows // 2, rows // 2)
            U, V = np.meshgrid(u, v)
            frequency_map = np.sqrt(U ** 2 + V ** 2)
            # print(np.max(frequency_map))
            # 创建低通滤波器
            low_frequency = (frequency_map <= self.phase_cut_range)
            low_frequency = torch.from_numpy(low_frequency).to(self.device)
            # 应用低通滤波器
            phase_channels_0_t[i][low_frequency] = phase_channels[i][low_frequency]
            ##再低频clip
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
        # return np.clip(phase_channels,phase_channels_0_t-delta,phase_channels_0_t+delta)
    
    def reconstruct_image(self,filtered_amplitude_channels, phase_channels):
        """使用低通滤波后的振幅和原始相位重建图像"""
        reconstructed_image = []
        for channel in range(3):
            amplitude = filtered_amplitude_channels[channel]
            phase = phase_channels[channel]-torch.pi

            # 使用振幅和相位重建频谱
            fshift_filtered = amplitude * torch.exp(1j * phase)

            # 进行傅里叶逆变换
            f_ishift = torch.fft.ifftshift(fshift_filtered)
            img_reconstructed = torch.fft.ifft2(f_ishift)
            img_reconstructed = torch.abs(img_reconstructed)
            # print(img_reconstructed)
            img_reconstructed = torch.clip(img_reconstructed,0,255)
            reconstructed_image.append(img_reconstructed/255)

        # 合并三个通道
        return torch.stack(reconstructed_image,dim=2)

        

    def amplitude_phase_exchange_torch(self,x,x_0_t):
        ### 先将[-1,1]转换到[0,1]范围再转到0-255 #
        x = (diff2clf(x)* 255).type(torch.uint8)
        x_0_t = (diff2clf(x_0_t)* 255).type(torch.uint8)
        batch,channel,height,width = x.shape
        new_x_0_t = torch.zeros(size=(batch,height,width,channel))
        for batch_idx in range(batch):
            ### 先计算正常图片的
            amplitude_channels, phase_channels = self.compute_fft(x[batch_idx])
            ### 再计算当前时间步预测的
            amplitude_channels_0_t, phase_channels_0_t = self.compute_fft(x_0_t[batch_idx])

            amplitude_channels_0_t_exchange = self.low_pass_exchange(amplitude_channels,amplitude_channels_0_t)
            phase_channels_0_t_exchange = self.phase_low_pass_exchange(phase_channels,phase_channels_0_t)

            reconstructed_image = self.reconstruct_image(amplitude_channels_0_t_exchange,phase_channels_0_t_exchange)

            new_x_0_t[batch_idx] = reconstructed_image
        new_x_0_t = new_x_0_t.float().permute(0,3,1,2).to(self.device)
        new_x_0_t = clf2diff(new_x_0_t)
        return new_x_0_t

    def denoise(self, x):


        ##### 实现参考https://blog.csdn.net/LittleNyima/article/details/139661712
        n = x.shape[0]
        x_t = self.diffuse_t_steps(x,self.timesteps[0])
        for t, tau in list(zip(self.timesteps[:-1], self.timesteps[1:])):

            ## 方差的系数
            if not math.isclose(self.eta, 0.0):
                one_minus_alpha_prod_tau = 1.0 - self.alphas_cumprod[tau]
                one_minus_alpha_prod_t = 1.0 - self.alphas_cumprod[t]
                one_minus_alpha_t = 1.0 - self.alphas[t]
                sigma_t = self.eta * (one_minus_alpha_prod_tau * one_minus_alpha_t / one_minus_alpha_prod_t) ** 0.5
            else:
                sigma_t = torch.zeros_like(torch.tensor(self.alphas[0]))
                

            ## DDIM 采样
            pred_noise = self.diffusion(x_t,(torch.ones(n)*t).to(self.device))
            if self.is_imagenet:
                pred_noise, _ = torch.split(pred_noise, 3, dim=1)
            # first term of x_tau
            alphas_cumprod_tau = (extract(self.alphas_cumprod, (torch.ones(n)*tau).to(self.device), x.shape))
            sqrt_alphas_cumprod_tau = alphas_cumprod_tau ** 0.5
            alphas_cumprod_t = (extract(self.alphas_cumprod, (torch.ones(n)*t).to(self.device), x.shape))
            sqrt_alphas_cumprod_t = alphas_cumprod_t ** 0.5
            sqrt_one_minus_alphas_cumprod_t = (1.0 - alphas_cumprod_t) ** 0.5
            x_0_t = (x_t - sqrt_one_minus_alphas_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t
            # print(torch.min(x),torch.max(x))

            ###将此处预测的x_0_t高频滤除掉并且以一定的频率对相位谱进行采样
            x_0_t = self.amplitude_phase_exchange_torch(x,x_0_t)

            first_term = sqrt_alphas_cumprod_tau * x_0_t

            # second term of x_tau
            coeff = (1.0 - alphas_cumprod_tau - sigma_t ** 2) ** 0.5
            second_term = coeff * pred_noise

            epsilon = torch.randn_like(x_t)
            x_t = first_term + second_term + sigma_t * epsilon

        x_0 = x_t
        return x_0

    def classify(self, x):
        logits = self.clf(x)
        return logits

    def forward(self, x):
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        
        x_diff = self.denoise(x_diff)

        # classifier part
        if self.is_imagenet:
            x_clf = normalize(diff2clf(F.interpolate(x_diff, size=(
                224, 224), mode='bilinear', align_corners=False)))
        
        x_clf = diff2clf(x_diff)
        logits = self.clf(x_clf)
        return logits
        # return x_clf

    def get_img_logits(self, x):
        # diffusion part
        if self.is_imagenet:
            x = F.interpolate(x, size=(256, 256),
                              mode='bilinear', align_corners=False)
        x_diff = clf2diff(x)
        
        x_diff = self.denoise(x_diff)

        # classifier part
        if self.is_imagenet:
            x_clf = normalize(diff2clf(F.interpolate(x_diff, size=(
                224, 224), mode='bilinear', align_corners=False)))
        
        x_clf = diff2clf(x_diff)
        logits = self.clf(x_clf)
        # return logits
        return x_clf,logits
