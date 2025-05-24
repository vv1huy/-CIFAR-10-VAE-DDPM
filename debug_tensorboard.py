from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

writer = SummaryWriter(log_dir='runs/debug_test')

for i in range(10):
    writer.add_scalar("test/loss", np.random.random(), i)
    img = torch.rand(1, 1, 28, 28)
    writer.add_image("test/image", img[0], i)

writer.close()
