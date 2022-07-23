import torchvision
from utils.Dataset_config import *

camelyon17_ds = torchvision.datasets.ImageFolder(root=DATASET_DIR,
                                                 transform=ds_transforms,
                                                 target_transform=None,
                                                 # TODO: maybe for now, merge macro and micro? ask david about this
                                                 is_valid_file=None,
                                                 # TODO: use this for separating validation and train in the same epoch, or find better way (sampler/loader)
                                                 )
