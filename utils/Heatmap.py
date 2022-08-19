import torch
import torchvision.transforms as T
from PIL import Image

def heatmap(WSI_heatmap,show=False):
    transform = T.ToPILImage()
    img = transform(WSI_heatmap)
    if show: # display the PIL image
        img.show()
    return img



