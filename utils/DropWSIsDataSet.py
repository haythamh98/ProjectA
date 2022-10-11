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
                 target_transform: Optional[Callable] = None,  # might be needed for now
                 negative_patches_ratio: float = 0.7,  # TODO: move to parameters
                 validation_WSI_IDs=Optional[Iterable],  ##### array of tuples (patient_ID, node_ID) for validation
                 is_validation=False
                 ):
        super(Camelyon17IterableDataset).__init__()
        self.target_transform = target_transform
        self.root = image_classes_root_path
        self.transform = transform
        self.validation_WSI_IDs = validation_WSI_IDs
        self.is_validation = is_validation
        self.negative_patches_ratio = negative_patches_ratio

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # if worker_info is None:  # single-process data loading, return the full iterator
        if True:
            # print("one worker mode")
            return iter(
                self.Camelyon17Iterator(root=self.root,
                                        WSI_skip_list=self.validation_WSI_IDs,
                                        transform=self.transform,
                                        target_transform=self.target_transform,
                                        negative_patches_ratio=self.negative_patches_ratio,
                                        is_validation=self.is_validation,
                                        )
            )
        else:  # in a worker process
            pass
            # TODO
        raise "NotImplementedError"

    # TODO: later: extend for multi workers
    class Camelyon17Iterator:
        ''' Iterator class '''

        def __init__(self, root: str, WSI_skip_list, transform, target_transform, negative_patches_ratio: float,
                     is_validation):
            self.target_transform = target_transform
            assert 0.1 <= negative_patches_ratio <= 0.9
            self.root = root
            self.negative_patches_ratio = negative_patches_ratio
            self.WSI_skip_list = WSI_skip_list
            self.transform = transform
            self.is_validation = is_validation
            negative_patches_ratio = int(negative_patches_ratio * 10)
            # one means extract Non negative patch,  zero : extract random negative patch
            self.types_extract_list =  ([0] * negative_patches_ratio + [1] * (10 - negative_patches_ratio))
            random.shuffle(self.types_extract_list)

            assert os.path.isdir(os.path.join(root, 'NEGATIVE'))
            assert len(os.listdir(root)) > 1  # there is at least one more tag other than negative

            self.negative_patches_dir = os.path.join(self.root, 'NEGATIVE')
            self.other_tags_files_names = os.listdir(self.root)
            # print("self.other_tags_files_names",self.other_tags_files_names)
            self.other_tags_files_names.remove('NEGATIVE')
            to_remove = []  # remove emty dirs
            for dir_name in self.other_tags_files_names:
                if len(os.listdir(os.path.join(self.root, dir_name))) < 1:  # TODO: make default threshold parameter
                    to_remove.append(dir_name)
            for tag_to_remove in to_remove:
                self.other_tags_files_names.remove(tag_to_remove)

            self.types_extract_list_idx = 0
            # print(f"supported tags other than negative are {self.other_tags_files_names}")

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
            # negative_dir_iterator = self.negative_dir_iterator
            # print("after os.walk")
            img, tag = None, None
            while True:
                self.types_extract_list_idx = self.types_extract_list_idx % len(self.types_extract_list)
                #print(self.types_extract_list_idx)
                work_dir = ""
                if self.types_extract_list[self.types_extract_list_idx] == 0:  # yield one negative patch
                    work_dir = self.negative_patches_dir
                    tag = 'NEGATIVE'
                else:
                    # print(self.other_tags_files_names)
                    tag_dir_name = random.choice(self.other_tags_files_names)
                    tag = tag_dir_name

                    work_dir = os.path.join(self.root, tag_dir_name)
                    # print("work dir " + work_dir)

                all_dir_patients = os.listdir(work_dir)

                # print("looking for tag = " + tag)
                def condition(x):  # patient_9_node_0
                    x = x.split("_")
                    p_id = int(x[1])
                    n_id = int(x[3][0])
                    if self.is_validation:
                        if (p_id, n_id) in self.WSI_skip_list:
                            return True
                        return False
                    else:
                        if (p_id, n_id) in self.WSI_skip_list:
                            return False
                        return True

                # print("before filter " + str(all_dir_patients))
                filtered_options = [x for x in all_dir_patients if condition(x)]
                # print("after filter " + filtered_options)
                # import time

                choice_dir = random.choice(filtered_options)

                # print("choice dir name p " + str(choice_dir))
                # time.sleep(1)
                # print("dir lookign at " + os.path.join(work_dir,choice_dir)  )
                # time.sleep(1)
                # print("options " )
                # print(str(os.listdir(os.path.join(work_dir,choice_dir))))
                pil_img_name = random.choice(os.listdir(os.path.join(work_dir, choice_dir)))

                #print("selected name ", pil_img_name)
                img_name = pil_img_name.split('_')
                # print(img_name)
                patient_ID = int(img_name[1])
                # print("patient_ID",patient_ID)
                node_ID = int(img_name[3][:-4])
                # print("pil_img_name,",pil_img_name)
                # print("seach path = " + os.path.join(all_dir_patients, choice_dir, pil_img_name))
                pil_img_path = os.path.join(os.path.join(work_dir, choice_dir, pil_img_name))
                try:
                    img = Image.open(pil_img_path).convert("RGB")
                    if not (img.size[0] > 0 and img.size[1] > 0):
                        # bad image
                        continue
                except:
                    # print(f"image {pil_img_path} is corrupted")
                    continue
                self.types_extract_list_idx += 1
                if self.transform is not None:
                    img = self.transform(img)
                if self.target_transform is None:
                    return img, class_to_idx(tag)
                else:
                    return img, self.target_transform(class_to_idx(tag))
