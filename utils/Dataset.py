import torchvision
from utils.Dataset_config import *

# Dataset
camelyon17_ds = torchvision.datasets.ImageFolder(
    root=DATASET_DIR,
    transform=ds_transforms,
    target_transform=None,
    # TODO: maybe for now, merge macro and micro? ask david about this
    is_valid_file=None,
    # TODO:
    #  1) use this for separating validation and train in the same epoch, or find better way (sampler/loader)
    #  2) add heuristic to prevent usage of images which contain "much" white background
)

# Dataloader
camelyon17_dl = torch.utils.data.DataLoader(
    dataset=camelyon17_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # TODO: might be better if we implement sampler which separates between train and validation
    num_workers=LOADER_N_WORKERS,
)
