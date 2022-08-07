import torchvision
from utils.Dataset_config import *
from PIL import Image

camelyon17_ds = None
camelyon17_dl = None


def is_valid_pil_image(image_path : str):
    img = Image.open(image_path)
    if img.size[0] > 0 and img.size[1] > 0:
        return True
    try:
        img =  Image.open(image_path)
        if img.size[0] > 0 and img.size[1] > 0 :
            return True
    except:
        print(f"image {image_path} is corrupted")

    return False



def init_ds_dl():
    global camelyon17_ds,camelyon17_dl
    # Dataset
    camelyon17_ds = torchvision.datasets.ImageFolder(
        root=DATASET_DIR,
        transform=ds_transforms,
        target_transform=None,
        # TODO: maybe for now, merge macro and micro? ask david about this
        is_valid_file=is_valid_pil_image,
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

