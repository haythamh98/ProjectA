import torchvision
from utils.Dataset_config import *
from PIL import Image
from utils.DropWSIsDataSet import Camelyon17IterableDataset

camelyon17_ds = None
camelyon17_train_ds = None
camelyon17_validation_ds = None
camelyon17_dl = None
camelyon17_train_dl = None
camelyon17_validation_dl = None

def to_wo_metastasis(x):
    return 1- int(x==0)

def init_ds_final_solution(validation_WSI_IDs, use_dummy_ds=False, only_train_set=False,
                           negative_patches_ratio_train=0.7, negative_patches_ratio_validation=0.7):
    global camelyon17_train_ds, camelyon17_train_dl, camelyon17_validation_ds, camelyon17_validation_dl

    # Dataset
    camelyon17_train_ds = Camelyon17IterableDataset(
        image_classes_root_path=DATASET_DIR if not use_dummy_ds else DUMMY_DATASET_DIR,
        transform=ds_transforms,
        target_transform=to_wo_metastasis,
        negative_patches_ratio=negative_patches_ratio_train,
        validation_WSI_IDs=validation_WSI_IDs,
        is_validation=False,
    )
    camelyon17_train_dl = torch.utils.data.DataLoader(
        dataset=camelyon17_train_ds,
        batch_size=BATCH_SIZE,
        num_workers=LOADER_N_WORKERS,  # TODO: num_workers not supported yet or it is....
    )
    if only_train_set:
        return camelyon17_train_ds, camelyon17_train_dl, None, None

    camelyon17_validation_ds = Camelyon17IterableDataset(
        image_classes_root_path=DATASET_DIR if not use_dummy_ds else DUMMY_DATASET_DIR,
        transform=None,  # in validation we dont use augmentation
        target_transform=to_wo_metastasis,
        negative_patches_ratio=negative_patches_ratio_validation,
        validation_WSI_IDs=validation_WSI_IDs,
        is_validation=True,
    )
    camelyon17_validation_dl = torch.utils.data.DataLoader(
        dataset=camelyon17_validation_ds,
        batch_size=BATCH_SIZE,
        num_workers=LOADER_N_WORKERS,  # TODO: num_workers not supported yet or it is....
    )
    print("Dataset & dataloaders init done")
    return camelyon17_train_ds, camelyon17_train_dl, camelyon17_validation_ds, camelyon17_validation_dl


def init3333_ds(validation_WSI_IDs, use_dummy_ds=False, only_train_set=False):
    global camelyon17_train_ds, camelyon17_train_dl, camelyon17_validation_ds, camelyon17_validation_dl
    # Dataset
    camelyon17_train_ds = Camelyon17IterableDataset(
        image_classes_root_path=DATASET_DIR if not use_dummy_ds else DUMMY_DATASET_DIR,
        transform=ds_transforms,
        # target_transform=None,
        negative_patches_ratio=0.8,
        validation_WSI_IDs=validation_WSI_IDs,
        is_validation=False,
    )
    camelyon17_train_dl = torch.utils.data.DataLoader(
        dataset=camelyon17_train_ds,
        batch_size=BATCH_SIZE,
        num_workers=LOADER_N_WORKERS,  # TODO: num_workers not supported yet or it is....
    )
    if only_train_set:
        return
    camelyon17_validation_ds = Camelyon17IterableDataset(
        image_classes_root_path=DATASET_DIR if not use_dummy_ds else DUMMY_DATASET_DIR,
        transform=ds_transforms,
        # target_transform=None,
        negative_patches_ratio=0.6,
        validation_WSI_IDs=validation_WSI_IDs,
        is_validation=True,
    )
    camelyon17_validation_dl = torch.utils.data.DataLoader(
        dataset=camelyon17_validation_ds,
        batch_size=BATCH_SIZE,
        num_workers=LOADER_N_WORKERS,  # TODO: num_workers not supported yet or it is....
    )
    print("Dataset & dataloaders init done")


def init2222_ds(validation_WSI_IDs, use_dummy_ds=False):
    global camelyon17_train_ds, camelyon17_train_dl
    # Dataset
    print("init camelyon17_train_ds")
    camelyon17_train_ds = DropWSIsDataSet(
        root=DATASET_DIR if not use_dummy_ds else DUMMY_DATASET_DIR,
        transform=ds_transforms,
        target_transform=None,
        validation_WSI_IDs=validation_WSI_IDs,
        is_validation=False,
    )
    print("init camelyon17_train_dl")
    camelyon17_train_dl = torch.utils.data.DataLoader(
        dataset=camelyon17_train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=LOADER_N_WORKERS,
    )


def init2222_test_ds(ds_path, use_dummy_ds=False):
    global camelyon17_validation_ds, camelyon17_validation_dl
    print("init camelyon17_validation_ds")
    camelyon17_validation_ds = DropWSIsDataSet(
        root=ds_path if not use_dummy_ds else DUMMY_DATASET_DIR,
        transform=ds_transforms,
        target_transform=None,
        validation_WSI_IDs=validation_WSI_IDs,
        is_validation=True,
    )
    # Dataloader

    # TODO: use class_to_idx to map between tag number and tag name (subFolder name)
    print("init camelyon17_validation_dl")
    camelyon17_validation_dl = torch.utils.data.DataLoader(
        dataset=camelyon17_validation_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=LOADER_N_WORKERS,
    )
    # TODO: use class_to_idx to map between tag number and tag name (subFolder name)


def is_valid_pil_image(image_path: str):
    try:
        img = Image.open(image_path)
        if img.size[0] > 0 and img.size[1] > 0:
            return True
    except:
        pass
        # print(f"image {image_path} is corrupted")

    return False


def init_ds_dl(sampler=None):
    global camelyon17_ds, camelyon17_dl
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
    print("sampling from camelyon17_train_dl")
    for i, Xy in enumerate(camelyon17_train_dl):
        X, y = Xy[0], Xy[1]  # X,y shape[0] == BATCH_SIZE
        print(f'Batch #{i} Size={y.shape[0]}')
        print(f'Tags {y}')
        if i >= n:
            break
    print("sampling from camelyon17_validation_dl")
    for i, Xy in enumerate(camelyon17_validation_dl):
        X, y = Xy[0], Xy[1]  # X,y shape[0] == BATCH_SIZE
        print(f'Batch #{i} Size={y.shape[0]}')
        print(f'Tags {y}')
        if i >= n:
            break
