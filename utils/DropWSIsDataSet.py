from typing import Callable, Any, Optional, Iterable

from PIL import Image
from random import shuffle
import random
from WSI_Tools.PatchExtractor_Tools.PatchExtractor_config import PYTORCH_IMAGE_DATASET_PATH
import torch
import os


def class_to_idx(classname):
    if 'NEGATIVE' in classname:
        return 0
    elif 'MACRO' in classname:
        return 1
    elif 'MICRO' in classname:
        return 2
    elif 'ITC' in classname:
        return 3
    else:
        print(classname)
        raise "wtf"



class Camelyon17IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 image_classes_root_path: str = PYTORCH_IMAGE_DATASET_PATH,
                 transform: Optional[Callable] = None,
                 # target_transform: Optional[Callable] = None, # might be needed for now
                 negative_patches_ratio: float = 0.7,  # TODO: move to parameters
                 validation_WSI_IDs=Optional[Iterable],  ##### array of tuples (patient_ID, node_ID) for validation
                 is_validation=False
                 ):
        super(Camelyon17IterableDataset).__init__()
        self.root = image_classes_root_path
        self.transform = transform
        self.validation_WSI_IDs = validation_WSI_IDs
        self.is_validation = is_validation
        self.negative_patches_ratio = negative_patches_ratio

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # if worker_info is None:  # single-process data loading, return the full iterator
        if True:
            print("one worker mode")
            return iter(
                self.Camelyon17Iterator(root=self.root,
                                        WSI_skip_list=self.validation_WSI_IDs,
                                        transform=self.transform,
                                        negative_patches_ratio=self.negative_patches_ratio,
                                        is_validation=self.is_validation,
                                        )
            )
        else:  # in a worker process
            # TODO
            pass
        raise "NotImplementedError"

    # TODO: later: extend for multi workers
    class Camelyon17Iterator:
        ''' Iterator class '''

        def __init__(self, root: str, WSI_skip_list, transform, negative_patches_ratio: float, is_validation):
            assert 0.1 <= negative_patches_ratio <= 0.9
            self.root = root
            self.negative_patches_ratio = negative_patches_ratio
            self.WSI_skip_list = WSI_skip_list
            self.transform = transform
            self.is_validation = is_validation
            negative_patches_ratio = int(negative_patches_ratio * 10)
            # one means extract Non negative patch,  zero : extract random negative patch
            self.types_extract_list = ([0] * negative_patches_ratio + [1] * (10 - negative_patches_ratio))
            random.shuffle(self.types_extract_list)

            assert os.path.isdir(os.path.join(root, 'NEGATIVE'))
            assert len(os.listdir(root)) > 1  # there is at least one more tag other than negative

            self.negative_patches_dir = os.path.join(self.root, 'NEGATIVE')
            self.other_tags_files_names = os.listdir(self.root)
            # print("self.other_tags_files_names",self.other_tags_files_names)
            self.other_tags_files_names.remove('NEGATIVE')
            to_remove = []  # remove emty dirs
            for dir_name in self.other_tags_files_names:
                if len(os.listdir(os.path.join(self.root, dir_name))) < 5:  # TODO: make default threshold parameter
                    to_remove.append(dir_name)
            for tag_to_remove in to_remove:
                self.other_tags_files_names.remove(tag_to_remove)

            self.types_extract_list_idx = 0
            print(f"supported tags other than negative are {self.other_tags_files_names}")

            self.negative_dir_iterator = iter(os.scandir(self.negative_patches_dir))

        def __iter__(self):
            return self

        def can_be_used(self, patient_ID, node_ID):
            wsi_ID = tuple((patient_ID, node_ID))
            if self.is_validation:
                if wsi_ID in self.WSI_skip_list:
                    # print(f"validation mode -> WSI {wsi_ID} is in {self.WSI_skip_list} --> valid")
                    return True
                else:
                    # print(f"validation mode -> WSI {wsi_ID} is NOT in {self.WSI_skip_list} --> not valid")
                    return False
            else:  # trainset
                if wsi_ID in self.WSI_skip_list:
                    # print(f"train mode -> WSI {wsi_ID} is in {self.WSI_skip_list} --> not valid")
                    return False
                else:
                    # print(f"train mode -> WSI {wsi_ID} is NOT in {self.WSI_skip_list} --> valid")
                    return True

        #### TODO: important, skip files in skiplist,
        def __next__(self):
            ''''Returns the next value from team object's lists '''

            # print("before os.walk")
            negative_dir_iterator = self.negative_dir_iterator
            # print("after os.walk")
            img, tag = None, None
            while True:
                self.types_extract_list_idx = self.types_extract_list_idx % len(self.types_extract_list)
                # print(self.types_extract_list_idx)
                if self.types_extract_list[self.types_extract_list_idx] == 0:  # yield one negative patch
                    # old: pil_img_name = next(negative_dir_iterator).name  # TODO: ask david... we cant randomize this
                    # new
                    negative_sub_dir = random.choice(os.listdir(self.negative_patches_dir))
                    negative_sub_dir_path = os.path.join(self.negative_patches_dir, negative_sub_dir)
                    pil_img_name = random.choice(os.listdir(negative_sub_dir_path))
                    # print(pil_img_name)
                    img_name = pil_img_name.split('_')
                    # print(img_name)
                    patient_ID = int(img_name[1])
                    # print("patient_ID",patient_ID)
                    node_ID = int(img_name[3][:-4])  #
                    # print("pil_img_name,",pil_img_name)
                    pil_img_path = os.path.join(os.path.join(negative_sub_dir_path, pil_img_name))
                    # print("pil_img_path",pil_img_path)
                    if not self.can_be_used(patient_ID, node_ID):
                        continue
                    try:
                        img = Image.open(pil_img_path).convert("RGB")
                        if not (img.size[0] > 0 and img.size[1] > 0):
                            # bad image
                            continue
                    except:
                        print(f"image {pil_img_path} is corrupted")

                    tag = 'NEGATIVE'
                else:
                    tag_dir_name = random.choice(self.other_tags_files_names)
                    pil_img_name = random.choice(os.listdir(os.path.join(self.root, tag_dir_name)))
                    pil_img_path = os.path.join(os.path.join(self.root, tag_dir_name, pil_img_name))
                    img_name = pil_img_name.split('_')
                    # print(img_name)
                    patient_ID = int(img_name[1])
                    # print("patient_ID",patient_ID)
                    node_ID = int(img_name[3][:-4])
                    # print("pil_img_name,",pil_img_name)
                    # print("pil_img_path",pil_img_path)
                    if not self.can_be_used(patient_ID, node_ID):
                        continue
                    try:
                        img = Image.open(pil_img_path).convert("RGB")
                        if not (img.size[0] > 0 and img.size[1] > 0):
                            # bad image
                            continue
                    except:
                        print(f"image {pil_img_path} is corrupted")
                    tag = tag_dir_name

                self.types_extract_list_idx += 1

                if self.transform is not None:
                    img = self.transform(img)

                return img, class_to_idx(tag)


"""
This Class is deprecated 
DatasetFolder -> uses map-style dataset, its not practicle for our dataset 


class DropWSIsDataSet(DatasetFolder):


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
                            # print(f"image {img_path} is validation")
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

"""
