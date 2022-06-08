import logging
import os.path,sys
import openslide
import torch
import logging
from typing import Iterator
from . import AnnotationHandler
from torch import Tensor
from torchvision import datasets, transforms, models


class PatchExtractor:
    def __init__(self, wsi_path: str, xml_path: str, size: tuple, patches_in_batch: int, tumor: bool, overlap: int = 0,
                 wsi_level: int = 0):
        if wsi_level != 0:
            raise NotImplementedError

        self.xml_path = xml_path
        self.wsi_path = wsi_path
        self.WSI_ID = int(os.path.split(wsi_path)[1][-7:-4])

        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        self.logger = logging.getLogger(name="PatchExtractor")
        self.logger.debug(f"WSI_ID {self.WSI_ID}")

        self.wsi_level_to_extract = wsi_level
        self.wsi = openslide.open_slide(wsi_path)

        self.patches_in_batch = patches_in_batch
        self.tumor = tumor
        self.annotation_classifier = AnnotationHandler.AnnotationHandler(xml_path)

        self.x, self.y = 13371, 147906 # TODO
        self.tensor_size = size
        self.x_step = size[0] - overlap
        self.y_step = size[1] - overlap
        self.x_end, self.y_end = 13371 + 600, 147906  # self.wsi.level_dimensions[wsi_level][0], self.wsi.level_dimensions[wsi_level][1]

        self.Done = False

    def __iter__(self) -> Iterator[Tensor]:
        if self.Done:
            return

        transform = transforms.Compose([transforms.ToTensor()])

        result_tensor = torch.zeros(size=(self.patches_in_batch, 4, *self.tensor_size))
        self.logger.debug(f"tensor shape {result_tensor.shape}")

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
            tmp_tensor = transform(cur_sample)

            if self.huristic(tmp_tensor) >= 0.9:  # TODO move to config.py
                sample_has_tumor = self.annotation_classifier.rectangle_has_tumor(x1=self.x, y1=self.y,
                                                                                  x2=self.x + self.x_step, y2=self.y,
                                                                                  x3=self.x + self.x_step,
                                                                                  y3=self.y + self.y_step,
                                                                                  x4=self.x, y4=self.y + self.y_step)
                if self.tumor == sample_has_tumor:
                    tmp_tensor = torch.unsqueeze(tmp_tensor,0)
                    result_tensor[in_tensor_count] = torch.clone(tmp_tensor)
                    in_tensor_count += 1
            ###############################
            self.x += self.x_step
        self.logger.debug(f"returning result_tensor with {in_tensor_count} valid results")
        if in_tensor_count != self.patches_in_batch:
            self.Done = True
        yield result_tensor, in_tensor_count  # tensor, number_of_valid_recs

    def huristic(self, tmp_tensor):
        return 1
