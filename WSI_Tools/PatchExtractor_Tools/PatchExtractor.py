from enum import Enum

import matplotlib

matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt

import os.path, sys
import cv2
import openslide
import logging

# from matplotlib import pyplot as plt
from WSI_Tools.PatchExtractor_Tools.AnnotationHandler import XMLAnnotationHandler
from PIL import ImageOps
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from PatchExtractor_config import *
from abc import ABC


# BIG TODO:
#   1) merge micro&macro together?

class PatchTag(Enum):
    NONE = -1
    NEGATIVE = 0
    MACRO = 1
    MICRO = 2
    ITC = 3

class PatchExtractor:
    def __init__(self,
                 wsi_path: str = wsi_dir_path,
                 xml_dir_path: str = annotation_xml_dir_path,
                 tag_csv_file_path : str = tag_csv_file_path,
                 patch_size: tuple = DEFAULT_PATCH_SIZE,
                 patch_overlap: tuple = DEFAULT_PATCH_OVERLAP,
                 negative_output_dir : str = NEGATIVE_OUTPUT_DIR,
                 macro_output_dir : str = MACRO_OUTPUT_DIR,
                 micro_output_dir : str =MICRO_OUTPUT_DIR ,
                 itc_output_dir : str =ITC_OUTPUT_DIR ,
                 wsi_level: int = DEFAULT_EXTRACTION_LEVEL,
                 down_scaled_image_annotated_boundaries_output_dir_path: str = DOWN_SCALLED_IMAGE_ANNOTATED_CONTOURS_OUTPUT_DIR_PATH,
                 logger= PATCH_EXTRACTORS_DEFAULT_LOGGER
                 ):


        if not DISABLE_LOGGER:
            if logger is not None:
                self.logger = logger
            else:
                logging.basicConfig(stream=sys.stdout, level=logging.INFO)
                self.logger = logging.getLogger(name="PatchExtractor_Tools")

        self.tag_csv_file_path = tag_csv_file_path
        self.xml_dir_path = xml_dir_path
        self.analyze_wsi_path(wsi_path)
        self.read_tag()

        if wsi_level != 0:
            raise NotImplementedError
        self.wsi_level_to_extract = wsi_level

        expected_xml_path = self.generate_expected_xml_path()
        self.tumor_classifier = XMLAnnotationHandler(expected_xml_path)  # return enum

        self.contours = self.generate_contours_around_wsi(down_scaled_image_annotated_boundaries_output_dir_path)

        self.itc_output_dir = itc_output_dir
        self.micro_output_dir = micro_output_dir
        self.macro_output_dir = macro_output_dir
        self.negative_output_dir = negative_output_dir
        self.patch_overlap = patch_overlap
        self.patch_size = patch_size

        self.wsi = openslide.open_slide(self.wsi_path)


    def analyze_wsi_path(self, wsi_path):
        self.wsi_path = wsi_path
        self.wsi_name = os.path.split(wsi_path)[1]
        print("self.wsi_name = os.path.split(wsi_path)[1][:-4]",self.wsi_name)
        self.patient_ID = int(self.wsi_name.split('_')[1])
        self.patient_node_ID = int(self.wsi_name.split('_')[3][0])
        self.wsi_Center_ID = -1
        if 'Center' in wsi_path: # TODO: check Center or center
            self.wsi_Center_ID = wsi_path[wsi_path.find('Center')+7:wsi_path.find('Center')+8]
            print('self.wsi_Center_ID',self.wsi_Center_ID)

    def read_tag(self):
        with open(self.tag_csv_file_path) as csv_file:
            lines = csv_file.readlines()
            for line in lines:
                if line.startswith(self.wsi_name):
                    line = line.split(',')
                    self.wsi_tag = line[-1]
                    print(' self.wsi_tag ', self.wsi_tag)
                    return
        raise Exception(f"TAG NOT FOUND for {self.wsi_name}")

    def generate_expected_xml_path(self):
        return os.path.join(self.xml_dir_path,self.wsi_name[:-3] + 'xml')

    def generate_contours_around_wsi(self, down_scaled_image_annotated_boundries_output_dir_path):
        xxx, yyy = self.wsi.level_dimensions[0][0] / DOWN_SAMPLE_RATE_FOR_GENERATING_CONTOUR_IMAGE, \
                   self.wsi.level_dimensions[0][1] / DOWN_SAMPLE_RATE_FOR_GENERATING_CONTOUR_IMAGE
        im = self.wsi.get_thumbnail(size=(xxx, yyy))
        img_gray = ImageOps.grayscale(im)
        blur = cv2.GaussianBlur(np.array(img_gray), IMG_CONTOUR_BLUR_KERNEL_SIZE, 0)
        # apply binary thresholding
        ret, thresh = cv2.threshold(blur, CV2_THRESH_FOR_EDGES, 255, cv2.THRESH_OTSU)
        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        # draw contours on the original image
        image_copy = np.array(im.copy())
        valid_contours = []
        # filter contours
        for cont in contours:
            if (xxx + yyy) * 0.9 > len(cont) > IMG_CONTOUR_MIN_NUM_POINTS:
                # TODO: filter vertical noise lines from wsi
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


        cv2.drawContours(image=image_copy, contours=valid_contours, contourIdx=-1, color=(0, 255, 0), thickness=1,
                         lineType=cv2.LINE_AA)
        cv2.imwrite(os.path.join(down_scaled_image_annotated_boundries_output_dir_path, self.wsi_name + '.jpg'), image_copy)

        return valid_contours

    # TODO: for now: all patches in node with ITC are considered itc
    # tag = micro/macro is ignored, and we calculate the max width of the metastasis
    def classify_metastasis_polygon(self,polygon : Polygon):
        for contour in self.contours:
            if polygon.intersects(contour):
                if self.wsi_tag == 'itc':
                    return PatchTag.ITC
                return self.tumor_classifier.get_polygon_metastasis(polygon)
        return PatchTag.NONE

    def start_extract(self):
        pass



"""
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
"""