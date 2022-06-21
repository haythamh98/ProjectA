import matplotlib

matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt

import logging
import os.path, sys
from enum import Enum
import cv2
import openslide
import torch
import logging

# from matplotlib import pyplot as plt
from typing import Iterator
from . import AnnotationHandler
from torch import Tensor
from torchvision import datasets, transforms, models
from PIL import Image, ImageOps
import numpy as np
from .PatchExtractor_config import *
from xml.dom import minidom
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.affinity import scale

"""
    BIG TODOS:
        1) different itr classes
        2) 
        3) 
"""


class ExtractType(Enum):
    tumor_only = 0
    normal_only = 1


class PatchExtractor:
    def __init__(self, wsi_path: str, xml_path: str = '', size: tuple = (512, 512), patches_in_batch: int = 64,
                 overlap: int = 0,
                 wsi_level: int = 0, extract_type: ExtractType = ExtractType.normal_only, logger=None):
        if logger is not None:
            raise NotImplementedError
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
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

        self.x, self.y = 0, 0  # 29975, 84826  # TODO 0,204194#
        self.tensor_size = size
        self.x_step = size[0] - overlap
        self.y_step = size[1] - overlap
        self.x_end, self.y_end = self.wsi.level_dimensions[wsi_level][0], self.wsi.level_dimensions[wsi_level][1]

        self.contours = []
        self.init_wsi_contours(True)

        self.Done = False

    def __iter__(self) -> Iterator[Tensor]:
        assert self.extract_type == ExtractType.normal_only or (
                self.extract_type != ExtractType.normal_only and self.xml_path != '')
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
                        self.Done = True
                        break
                    else:  # start next line
                        self.x = 0
                        self.y += self.y_step
                ###############################
                cur_sample = self.wsi.read_region(location=(self.x, self.y),
                                                  level=self.wsi_level_to_extract, size=self.tensor_size)
                tmp_tensor = transform(cur_sample.convert("RGB"))
                if self.is_valid_patch(x1=self.x, y1=self.y,
                                       x2=self.x + self.x_step,
                                       y2=self.y,
                                       x3=self.x + self.x_step,
                                       y3=self.y + self.y_step,
                                       x4=self.x,
                                       y4=self.y + self.y_step):
                    # stam11.show()
                    sample_has_tumor = False
                    if self.extract_type == ExtractType.tumor_only:
                        sample_has_tumor = self.annotation_classifier.rectangle_has_tumor(x1=self.x, y1=self.y,
                                                                                          x2=self.x + self.x_step,
                                                                                          y2=self.y,
                                                                                          x3=self.x + self.x_step,
                                                                                          y3=self.y + self.y_step,
                                                                                          x4=self.x,
                                                                                          y4=self.y + self.y_step)
                    if (
                            self.extract_type == ExtractType.tumor_only and sample_has_tumor) \
                            or self.extract_type == ExtractType.normal_only:  # in case only looking for tumors in WSI
                        self.logger.debug(
                            f"adding x,y = ({self.x},{self.y})  sample num={in_tensor_count}")
                        # print(f"adding x,y = ({self.x},{self.y})  sample num={in_tensor_count}")
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
        self.Done = False
        self.x = 0
        self.y = 0

    def extract_path_batches_to_tensors(self, output_path_dir):

        if not os.path.isdir(output_path_dir):
            try:
                os.mkdir(output_path_dir, 777)
            except:
                self.logger.critical(f"Cannot create dir {output_path_dir}")

        for i, (batch, samples_count) in enumerate(self):
            type_name = 'with_tumor' if self.extract_type == ExtractType.tumor_only else 'without_tumor'
            out_tensor_path = os.path.join(output_path_dir,
                                           f'{self.WSI_type}_{self.WSI_ID}_{type_name}_{i}.pt')
            if samples_count != self.patches_in_batch:
                batch = batch[:samples_count]
            torch.save(batch, out_tensor_path)
            print(f"file {out_tensor_path} was done with {samples_count}")

    def is_valid_patch(self, x1: float, y1: float, x2: float, y2: float,
                       x3: float, y3: float, x4: float, y4: float):
        poly = Polygon([Point(x1, y1), Point(x2, y2), Point(x3, y3), Point(x4, y4)])
        for i, cont in enumerate(self.contours):
            if poly.intersects(cont):
                x, y = poly.exterior.xy
                plt.plot(x, y)
                x, y = cont.exterior.xy
                plt.plot(x, y)
                ax = plt.gca()
                ax.invert_yaxis()  # TODO check this
                # plt.show()
                # print(f'{poly} found in {i}')
                return True
        return False

    def init_wsi_contours(self, visualize=False):
        xxx, yyy = self.wsi.level_dimensions[0][0] / 100, self.wsi.level_dimensions[0][1] / 100
        im = self.wsi.get_thumbnail(size=(xxx, yyy))
        img_gray = ImageOps.grayscale(im)
        blur = cv2.GaussianBlur(np.array(img_gray), IMG_CONTOUR_BLUR_KERNEL_SIZE, 0)
        # apply binary thresholding
        ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_OTSU)
        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        # draw contours on the original image
        image_copy = np.array(im.copy())
        valid_contours = []

        for cont in contours:
            if (xxx + yyy) * 0.9 > len(cont) > IMG_CONTOUR_MIN_SIZE:
                valid_contours += [cont]

        x_scale_ratio = self.wsi.level_dimensions[0][0] / xxx
        y_scale_ratio = self.wsi.level_dimensions[0][1] / yyy

        for contour in valid_contours:
            cur_polygon = []
            for point in contour:
                x = (float(point[0][0]) * x_scale_ratio)
                y = (float(point[0][1]) * y_scale_ratio)
                cur_polygon.append(Point(x, y))

            polly = Polygon(cur_polygon)
            # polly = scale(polly, yfact=-1, origin=(1, 0))
            self.contours.append(polly)  # (36864,87898)
            x, y = polly.exterior.xy
            plt.plot(np.array(x), np.array(y))
            ax = plt.gca()
            # ax.invert_yaxis()  # TODO check this
            # plt.show()
        # visualize = True
        if visualize:
            cv2.drawContours(image=image_copy, contours=valid_contours, contourIdx=-1, color=(0, 255, 0), thickness=1,
                             lineType=cv2.LINE_AA)
            # see the results
            #cv2.imshow("5ara", image_copy)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            full_wsi_name = os.path.split(self.wsi_path)[1][:-4]
            print(full_wsi_name)
            cv2.imwrite(os.path.join(os.path.split(self.wsi_path)[0],full_wsi_name + '.jpg'), image_copy)
