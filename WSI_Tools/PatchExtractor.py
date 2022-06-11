import logging
import os.path, sys
from enum import Enum

import openslide
import torch
import logging
from typing import Iterator
from . import AnnotationHandler
from torch import Tensor
from torchvision import datasets, transforms, models

"""
    BIG TODOS:
        1) different itr classes
        2) 
        3) 
"""
class ExtractType(Enum):
    tumor_only=0
    normal_only=1

class PatchExtractor:
    def __init__(self, wsi_path: str, xml_path: str = '', size: tuple = (512,512), patches_in_batch: int = 64,  overlap: int = 0,
                 wsi_level: int = 0,extract_type: ExtractType = ExtractType.normal_only, logger=None):
        if logger is not None:
            raise NotImplementedError
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        self.logger = logging.getLogger(name="PatchExtractor")


        if wsi_level != 0:
            raise NotImplementedError

        self.wsi_path = wsi_path
        self.WSI_ID = int(os.path.split(wsi_path)[1][-7:-4])
        full_file_name = os.path.split(wsi_path)[1]
        self.WSI_type = full_file_name[0:full_file_name.index('_')]
        self.logger.debug(f"image name {full_file_name} type={self.WSI_type} ID={self.WSI_ID}")
        self.wsi_level_to_extract = wsi_level
        self.wsi = openslide.open_slide(wsi_path)

        self.patches_in_batch = patches_in_batch

        self.extract_type = extract_type

        self.xml_path = xml_path
        if xml_path != '':
            self.annotation_classifier = AnnotationHandler.AnnotationHandler(xml_path)

        self.x, self.y = 0, 0  # 13371, 147906  # TODO
        self.tensor_size = size
        self.x_step = size[0] - overlap
        self.y_step = size[1] - overlap
        self.x_end, self.y_end = self.wsi.level_dimensions[wsi_level][0], self.wsi.level_dimensions[wsi_level][1]

        self.Done = False

    def __iter__(self) -> Iterator[Tensor]:
        assert self.extract_type == ExtractType.normal_only or (self.extract_type != ExtractType.normal_only and self.xml_path != '')
        while not self.Done:
            transform = transforms.Compose([transforms.ToTensor()])

            result_tensor = torch.zeros(size=(self.patches_in_batch, 3, *self.tensor_size))
            # self.logger.debug(f"tensor shape {result_tensor.shape}")

            in_tensor_count = 0
            while in_tensor_count < self.patches_in_batch:
                if self.x_end < self.x + self.x_step:
                    # go to new row
                    if self.y_end < self.y + self.y_step:  # done iterating
                        self.x, self.y = 0, 0
                        self.logger.info(f"Finished slide, returning {in_tensor_count} valid results")
                        break
                    else:  # start next line
                        self.x = 0
                        self.y += self.y_step
                ###############################
                cur_sample = self.wsi.read_region(location=(self.x, self.y),
                                                  level=self.wsi_level_to_extract, size=self.tensor_size)
                tmp_tensor = transform(cur_sample.convert("RGB"))
                # self.logger.debug(f"tensor to fil shape {tmp_tensor.shape}")
                if self.huristic(tmp_tensor) >= 0.7:  # TODO move to config.py
                    sample_has_tumor = False
                    if self.extract_type == ExtractType.tumor_only:
                        sample_has_tumor = self.annotation_classifier.rectangle_has_tumor(x1=self.x, y1=self.y,
                                                                                          x2=self.x + self.x_step, y2=self.y,
                                                                                          x3=self.x + self.x_step,
                                                                                          y3=self.y + self.y_step,
                                                                                          x4=self.x, y4=self.y + self.y_step)
                    if (self.extract_type == ExtractType.tumor_only and sample_has_tumor) or self.extract_type == ExtractType.normal_only:  # in case only looking for tumors in WSI
                        tmp_tensor = torch.unsqueeze(tmp_tensor, 0)
                        result_tensor[in_tensor_count] = torch.clone(tmp_tensor)
                        in_tensor_count += 1
                ###############################
                self.x += self.x_step
            if in_tensor_count != self.patches_in_batch:
                self.Done = True
            self.logger.debug(f"before yielding, x={self.x}, y= {self.y}")
            yield result_tensor, in_tensor_count  # tensor, number_of_valid_recs

    def resetITR(self):
        self.x = 0
        self.y = 0

    def extract_path_batches_to_tensors(self, output_path_dir):
        '''
        >>> t = torch.tensor([1., 2.])
        >>> torch.save(t, 'tensor.pt')
        >>> torch.load('tensor.pt')
        tensor([1., 2.])
        '''
        for i, (batch, samples_count) in enumerate(self):
            type_name = 'with_tumor' if self.extract_type == ExtractType.tumor_only else 'without_tumor'
            out_tensor_path = os.path.join(output_path_dir,
                                           f'{self.WSI_type}_{self.WSI_ID}_{type_name}_{i}.pt')
            if samples_count != self.patches_in_batch:
                batch = batch[:samples_count]
            torch.save(batch, out_tensor_path)
            print(f"file {out_tensor_path} was done with {samples_count}")


    def huristic(self, tmp_tensor):
        red_channel = tmp_tensor[0]
        # print(torch.mean(red_channel))
        return torch.mean(red_channel)
