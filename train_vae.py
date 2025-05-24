import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model_vae import VAE
from data_loader import get_dataloader
import os

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader, _ = get_dataloader()
    
    # 确保日志目录存在
    log_dir = 'logs/vae'
    os.makedirs(log_dir, exist_ok=True)
    print(f"创建日志目录: {os.path.abspath(log_dir)}")
    
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard writer 已初始化")
    
    try:
        for epoch in range(10):
            model.train()
            total_loss = 0
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.to(device)
                optimizer.zero_grad()
                recon_x, mu, logvar = model(x)
                loss = vae_loss(recon_x, x, mu, logvar)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # 记录每个批次的损失
                if batch_idx % 100 == 0:
                    current_step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar('Loss/batch', loss.item(), current_step)
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, Step: {current_step}')
                    # 强制写入日志
                    writer.flush()
                
                # 每个 epoch 添加一些重建图像
                if batch_idx == 0:
                    writer.add_images('Original', x[:8], epoch)
                    writer.add_images('Reconstructed', recon_x[:8], epoch)
                    writer.flush()
                    print(f"Epoch {epoch}: 已保存图像")
            
            # 记录每个 epoch 的平均损失
            avg_loss = total_loss / len(train_loader)
            writer.add_scalar('Loss/epoch', avg_loss, epoch)
            print(f'Epoch {epoch} Average Loss: {avg_loss:.4f}')
            writer.flush()
        
        torch.save(model.state_dict(), 'vae.pth')
    finally:
        print("正在关闭 TensorBoard writer...")
        writer.close()
        print("TensorBoard writer 已关闭")