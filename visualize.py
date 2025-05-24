import matplotlib.pyplot as plt
import torch

def show_images(images, title=""):
    grid_img = torch.cat([img for img in images[:8]], dim=2)
    plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
    plt.title(title)
    plt.axis('off')
    plt.show()