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
# TODO: use class_to_idx to map between tag number and tag name (subFolder name)





def draw_random_samples(n: int = 5):
    # not sure how to plot when using remote :(
    for i,Xy in enumerate(camelyon17_dl):
        X, y = Xy[0], Xy[1]  # X,y shape[0] == BATCH_SIZE
        print(f'Batch #{i} Size={y.shape[0]}')
        print(f'Tags {y}')
        if i >= n:
            break

