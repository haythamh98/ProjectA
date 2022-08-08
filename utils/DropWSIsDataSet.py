from torch.utils.data import IterableDataset
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
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            validation_WSI_IDs = None,
            is_validation = False
    ):
        self.validation_WSI_IDs = validation_WSI_IDs
        self.is_validation = is_validation

        self.is_valid_function = lambda img_path: is_valid(img_path,self.is_validation)

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
    def is_valid(img_path:str,is_validation_set:bool,validation_WSI_IDs):
        if is_validation_set:
            patient_ID = os.path





        try:
            img = Image.open(image_path)
            if img.size[0] > 0 and img.size[1] > 0:
                return True
        except:
            print(f"image {image_path} is corrupted")

        return False

