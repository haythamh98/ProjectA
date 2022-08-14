import torch
import torchvision.transforms as T
from PIL import Image

def heatmap(WSI_heatmap):
    transform = T.ToPILImage()
    img = transform(WSI_heatmap)
    # display the PIL image
    img.show()



