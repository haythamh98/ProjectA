from typing import Callable, Any, Optional

from PIL import Image
from torchvision.datasets.folder import default_loader, DatasetFolder

from WSI_Tools.PatchExtractor_Tools.PatchExtractor_config import PYTORCH_IMAGE_DATASET_PATH
import torch
import os


class DropWSIsDataSet(DatasetFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str = PYTORCH_IMAGE_DATASET_PATH,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            validation_WSI_IDs = None, ##### array of tuples (patient_ID, node_ID) for validation
            is_validation = False
    ):
        self.validation_WSI_IDs = validation_WSI_IDs
        self.is_validation = is_validation

        self.is_valid_function = lambda img_path: self.is_valid(img_path,self.is_validation,validation_WSI_IDs)

        super().__init__(
            root,
            loader,
            None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=self.is_valid_function,
        )
        self.imgs = self.samples

    @staticmethod
    def is_valid(img_path:str,is_validation_set:bool,validation_WSI_IDs) -> bool:
        assert len(validation_WSI_IDs) > 0
        # eg. patient_044_node_4.tif_xy_38555_14340_512x512.png
        img_name = os.path.split(img_path)[1].split('_')
        # print(img_name)
        patient_ID = int(img_name[1])
        # print("patient_ID",patient_ID)
        node_ID = int(img_name[3][:-4]) # TODO: fix this error, name should not include .tif after node ID
        if is_validation_set:
            for tuple_patient_node in validation_WSI_IDs:
                # print(tuple_patient_node)
                patient, node = tuple_patient_node
                # print(patient, node)
                if patient_ID ==  patient and node_ID == node:
                    # print("found" , patient, node)
                    try:
                        img = Image.open(img_path)
                        if img.size[0] > 0 and img.size[1] > 0:
                            print(f"image {img_path} is validation")
                            return True
                    except:
                        print(f"image {img_path} is corrupted")
                        return False
            return False
        else:
            for tuple_patient_node in validation_WSI_IDs:
                patient, node = tuple_patient_node
                if patient_ID == patient and node_ID == node:
                    return False
            try:
                img = Image.open(img_path)
                if img.size[0] > 0 and img.size[1] > 0:
                    print(f"image {img_path} is train")
                    return True
            except:
                print(f"image {img_path} is corrupted")
                return False
        raise "should not reach here"



