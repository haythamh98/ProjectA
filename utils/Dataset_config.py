import os
import torch
from torchvision.transforms import transforms

from WSI_Tools.PatchExtractor_Tools.PatchExtractor_config import LOCAL_DATABASE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
DATASET_DIR = os.path.join(LOCAL_DATABASE, 'Pytorch_Dataset')
DUMMY_DATASET_DIR = os.path.join(LOCAL_DATABASE, 'Pytorch_Dataset_dummy')


# image transforms after load
ds_transforms = transforms.Compose([  # TODO
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    # TODO: maybe use this?
    # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) #
])

# Properties
LOADER_N_WORKERS = 7

# training
BATCH_SIZE = 64
