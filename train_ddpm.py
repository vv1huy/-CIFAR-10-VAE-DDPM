import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model_ddpm import DenoiseModel
from data_loader import get_dataloader
import os
import numpy as np

# DDPM超参数
T = 1000  # 扩散步数
beta_start = 1e-4
beta_end = 0.02

def setup_diffusion(device):
    """设置扩散过程参数并移动到正确的设备"""
    betas = torch.linspace(beta_start, beta_end, T).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.]).to(device), alphas_cumprod[:-1]], dim=0)
    return betas, alphas, alphas_cumprod, alphas_cumprod_prev

def q_sample(x_start, t, noise=None, alphas_cumprod=None):
    """正向扩散：加噪声"""
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod = alphas_cumprod[t].sqrt()
    sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod[t]).sqrt()
    return sqrt_alphas_cumprod[:, None, None, None] * x_start + \
           sqrt_one_minus_alphas_cumprod[:, None, None, None] * noise

def p_sample(model, x, t, betas=None, alphas=None, alphas_cumprod=None):
    """反向采样：去噪"""
    betas_t = betas[t]
    sqrt_one_minus_alphas_cumprod_t = (1 - alphas_cumprod[t]).sqrt()
    sqrt_recip_alphas_t = (1. / alphas[t]).sqrt()
    model_mean = sqrt_recip_alphas_t[:, None, None, None] * (x - betas_t[:, None, None, None] * model(x, t) / sqrt_one_minus_alphas_cumprod_t[:, None, None, None])
    if t[0] == 0:
        return model_mean
    noise = torch.randn_like(x)
    posterior_var = betas_t[:, None, None, None]
    return model_mean + posterior_var.sqrt() * noise

def sample_ddpm(model, shape, device, betas=None, alphas=None, alphas_cumprod=None):
    """从噪声采样生成图片"""
    model.eval()
    with torch.no_grad():
        x = torch.randn(shape, device=device)
        for i in reversed(range(T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = p_sample(model, x, t, betas, alphas, alphas_cumprod)
        x = torch.clamp(x, -1., 1.)
    return x

def train_ddpm():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 初始化模型和优化器
    model = DenoiseModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    
    # 设置扩散过程参数
    betas, alphas, alphas_cumprod, alphas_cumprod_prev = setup_diffusion(device)
    
    # 数据加载和日志设置
    train_loader, _ = get_dataloader(batch_size=128)
    log_dir = 'logs/ddpm'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    num_epochs = 20
    global_step = 0

    try:
        for epoch in range(num_epochs):
            model.train()
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.to(device)
                bsz = x.size(0)
                t = torch.randint(0, T, (bsz,), device=device).long()
                noise = torch.randn_like(x)
                x_noisy = q_sample(x, t, noise, alphas_cumprod)
                pred_noise = model(x_noisy, t)
                loss = F.mse_loss(pred_noise, noise)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                writer.add_scalar('Loss/train', loss.item(), global_step)
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}")
                global_step += 1
                
            # 每个epoch采样图片
            print(f"Sampling images for epoch {epoch}...")
            sample_imgs = sample_ddpm(model, (8, 3, 32, 32), device, betas, alphas, alphas_cumprod)
            writer.add_images('Samples', (sample_imgs+1)/2, epoch)  # 反归一化到[0,1]
            torch.save(model.state_dict(), f'ddpm_epoch{epoch}.pth')
            print(f"Epoch {epoch} completed")
            
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        writer.close()
        print("Training completed")

if __name__ == '__main__':
    train_ddpm() 