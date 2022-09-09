import torch
from utils import Dataset
from torchvision import models

global model

def init_model():
    global model
    model = models.resnet18(pretrained=False)
    model.eval()
    print(list(model.children())[:])
    # model = torch.nn.Sequential(*(list(model.children())[:-1]))  # strips off last linear layer


def run_final_train_model():

    camelyon17_train_ds, camelyon17_train_dl, camelyon17_validation_ds, camelyon17_validation_dl = Dataset.init_ds_final_solution(
        validation_WSI_IDs=[],
        use_dummy_ds=True,
        only_train_set=True,
        negative_patches_ratio_train=0.7,
        negative_patches_ratio_validation=0.7,
    )





    # change validation and train
    new_validation_set = []
    camelyon17_train_ds.validation_WSI_IDs = new_validation_set
    camelyon17_validation_ds.validation_WSI_IDs = new_validation_set