import time

#import matplotlib
#matplotlib.use('GTK3Agg')
#import matplotlib.pyplot as plt

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
from WSI_Tools.PatchExtractor_Tools.PatchExtractor_config import *
from abc import ABC
import threading


# BIG TODO:
#   1) merge micro&macro together?

# more code under the class

class PatchExtractor:
    def __init__(self,
                 wsi_path: str,
                 xml_dir_path: str = annotation_xml_dir_path,
                 tag_csv_file_path: str = tag_csv_file_path,
                 patch_size: tuple = DEFAULT_PATCH_SIZE,
                 patch_overlap: tuple = DEFAULT_PATCH_OVERLAP,
                 negative_output_dir: str = NEGATIVE_OUTPUT_DIR,
                 macro_output_dir: str = MACRO_OUTPUT_DIR,
                 micro_output_dir: str = MICRO_OUTPUT_DIR,
                 itc_output_dir: str = ITC_OUTPUT_DIR,
                 wsi_level: int = DEFAULT_EXTRACTION_LEVEL,
                 down_scaled_image_annotated_boundaries_output_dir_path: str = DOWN_SCALED_IMAGE_ANNOTATED_CONTOURS_OUTPUT_DIR_PATH,
                 logger=PATCH_EXTRACTORS_DEFAULT_LOGGER
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
        self.wsi = openslide.open_slide(self.wsi_path)
        expected_xml_path = self.generate_expected_xml_path()
        # TODO: in order to distinguish between macro and micro, must pass ratio per pixel
        # for now, only solution is to set all as micro
        self.tumor_classifier = XMLAnnotationHandler(expected_xml_path,1)  # return enum

        self.contours = []
        self.generate_contours_around_wsi(down_scaled_image_annotated_boundaries_output_dir_path)

        self.itc_output_dir = itc_output_dir
        self.micro_output_dir = micro_output_dir
        self.macro_output_dir = macro_output_dir
        self.negative_output_dir = negative_output_dir
        self.patch_overlap = patch_overlap
        self.patch_size = patch_size

    def analyze_wsi_path(self, wsi_path):
        self.wsi_path = wsi_path
        self.wsi_name = os.path.split(wsi_path)[1]
        self.patient_ID = int(self.wsi_name.split('_')[1])
        self.patient_node_ID = int(self.wsi_name.split('_')[3][0])
        self.wsi_Center_ID = -1
        if 'center' in wsi_path:
            self.wsi_Center_ID = int(wsi_path[wsi_path.find('center') + 7:wsi_path.find('center') + 8])

        assert 0 <= self.wsi_Center_ID < N_DATA_CENTERS

    def read_tag(self):
        if self.tag_csv_file_path is None:
            return  # extract for test
        with open(self.tag_csv_file_path) as csv_file:
            lines = csv_file.readlines()
            for line in lines:
                if line.startswith(self.wsi_name):
                    line = line.split(',')
                    self.wsi_tag = line[-1][:-1]  # in conversion \n is added, yes, so bad
                    return
        raise Exception(f"TAG NOT FOUND for {self.wsi_name}")

    def generate_expected_xml_path(self):
        return os.path.join(self.xml_dir_path, self.wsi_name[:-3] + 'xml')

    def generate_contours_around_wsi(self, down_scaled_image_annotated_boundries_output_dir_path):
        xxx, yyy = self.wsi.level_dimensions[0][0] / DOWN_SAMPLE_RATE_FOR_GENERATING_CONTOUR_IMAGE[self.wsi_Center_ID], \
                   self.wsi.level_dimensions[0][1] / DOWN_SAMPLE_RATE_FOR_GENERATING_CONTOUR_IMAGE[self.wsi_Center_ID]
        im = self.wsi.get_thumbnail(size=(xxx, yyy))
        img_gray = ImageOps.grayscale(im)
        img_gray = np.array(img_gray)
        img_gray[img_gray > BLACK_COLOR_THRESH_TO_IGNORE[self.wsi_Center_ID]] = 0
        blur = cv2.GaussianBlur(np.array(img_gray), IMG_CONTOUR_BLUR_KERNEL_SIZE[self.wsi_Center_ID], 0)
        # apply binary thresholding
        if self.wsi_Center_ID != 2:
          ret, thresh = cv2.threshold(blur, CV2_THRESH_FOR_EDGES[self.wsi_Center_ID], 255, cv2.THRESH_OTSU)
        else:
          ret, thresh = cv2.threshold(blur, CV2_THRESH_FOR_EDGES[self.wsi_Center_ID], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        # draw contours on the original image
        image_copy = np.array(im.copy())
        valid_contours = []
        # filter contours

        x_scale_ratio = self.wsi.level_dimensions[0][0] / xxx
        y_scale_ratio = self.wsi.level_dimensions[0][1] / yyy

        for cont in contours:
            # ignore if too big (to ignore boundaries of the wsi)
            if len(cont) > (xxx + yyy) * 0.9 or len(cont) < IMG_CONTOUR_MIN_NUM_POINTS[self.wsi_Center_ID]:
                continue
            # filter vertical/horizontal noise lines from wsi scanner
            polly = None
            # try:
            cur_polygon = []
            for point in cont:
                x = float(point[0][0])
                y = float(point[0][1])
                cur_polygon.append(Point(x, y))
            try:
                polly = Polygon(cur_polygon)
                box = polly.minimum_rotated_rectangle
                x, y = box.exterior.coords.xy
                axis = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
                if axis[0] == 0 or axis[1] == 0 or axis[1] / axis[0] > DROP_WSI_SCANNER_NOISE_LINE_THRESH[
                    self.wsi_Center_ID] or axis[0] / \
                        axis[1] > DROP_WSI_SCANNER_NOISE_LINE_THRESH[self.wsi_Center_ID]:
                    # print("Dropping scanner noise")
                    # TODO: some noise still pass, check better condition
                    continue
            except:
                # the polygon doesn't form a "valid" shape
                print("generate_contours_around_wsi: Exception: the polygon doesn't form a valid shape")
                continue

            # if passed all filters, append to real contours
            valid_contours += [cont]

        for contour in valid_contours:
            cur_polygon = []
            for point in contour:
                x = (float(point[0][0]) * x_scale_ratio)
                y = (float(point[0][1]) * y_scale_ratio)
                cur_polygon.append(Point(x, y))
            polly = Polygon(cur_polygon)
            self.contours.append(polly)

            x, y = polly.exterior.xy
            #plt.plot(np.array(x), np.array(y))
            #ax = plt.gca()

        # draw wsi contours to output image
        cv2.drawContours(image=image_copy, contours=valid_contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                         lineType=cv2.LINE_AA)
        # draw annotation contours to output image
        annotation_contours = []
        for poly in self.tumor_classifier.polygons:
            x, y = poly.exterior.xy
            xs = np.array(x) / DOWN_SAMPLE_RATE_FOR_GENERATING_CONTOUR_IMAGE[self.wsi_Center_ID]
            ys = np.array(y) / DOWN_SAMPLE_RATE_FOR_GENERATING_CONTOUR_IMAGE[self.wsi_Center_ID]
            xs = xs.astype(np.int32)
            ys = ys.astype(np.int32)
            # naive
            tmp = []
            for x,y in zip(xs,ys):
                tmp.append([x,y])
            annotation_contours.append(np.array(tmp))


        cv2.drawContours(image=image_copy, contours=annotation_contours, contourIdx=-1, color=(255, 0, 0), thickness=2,
                         lineType=cv2.LINE_AA)
        # save image
        cv2.imwrite(os.path.join(down_scaled_image_annotated_boundries_output_dir_path, self.wsi_name + '.jpg'),
                    image_copy)

    # TODO: for now: all patches in node with ITC are considered itc
    # tag = micro/macro tag is ignored, and we calculate the max width of the metastasis, since WSI can include both micro/macro
    def classify_metastasis_polygon(self, polygon: Polygon):
        for contour in self.contours:
            if polygon.intersects(contour):
                if self.wsi_tag == 'itc':
                    return PatchTag.ITC
                return self.tumor_classifier.get_polygon_metastasis(polygon)
        return PatchTag.NONE

    def start_extract(self):
        x_step = self.patch_size[0] - self.patch_overlap[0]
        y_step = self.patch_size[1] - self.patch_overlap[1]

        def parallel_polygon_extraction(poly, fn_index):
            print(f"started function {fn_index}")
            # for i, poly in enumerate(self.contours):
            #    print(f"Dealing with poly num {i} out of {len(self.contours)}")
            x_arr, y_arr = poly.exterior.coords.xy
            x_start, y_start = int(min(x_arr)), int(min(y_arr))
            x_stop, y_stop = int(max(x_arr)), int(max(y_arr))
            for y in range(y_start, y_stop, y_step):
                for x in range(x_start, x_stop, x_step):
                    patch_poly = None
                    cur_sample = None
                    try:
                        patch_poly = Polygon(
                            [Point(x, y), Point(x + x_step, y), Point(x + x_step, y + y_step), Point(x, y + y_step)])
                        if not patch_poly.intersects(poly):
                            continue
                        cur_sample = self.wsi.read_region(location=(x, y),
                                                          level=self.wsi_level_to_extract,
                                                          size=self.patch_size)
                    except:
                        print("bad intersection in polygons")
                    patch_tag = self.classify_metastasis_polygon(patch_poly)

                    output_PIL_img_name = f'{self.wsi_name}_xy_{x}_{y}_{self.patch_size[0]}x{self.patch_size[1]}.png'
                    output_PIL_img_full_path = ''
                    if patch_tag == PatchTag.NEGATIVE:
                        output_PIL_img_full_path = os.path.join(self.negative_output_dir , output_PIL_img_name)
                    elif patch_tag == PatchTag.MACRO:
                        output_PIL_img_full_path = os.path.join(self.macro_output_dir , output_PIL_img_name)
                    elif patch_tag == PatchTag.MICRO:
                        output_PIL_img_full_path = os.path.join(self.micro_output_dir , output_PIL_img_name)
                    elif patch_tag == PatchTag.ITC:
                        output_PIL_img_full_path = os.path.join(self.itc_output_dir , output_PIL_img_name)
                    else:
                        raise Exception("Patch has no tag")

                    cur_sample.save(output_PIL_img_full_path)

            print(f"done function {fn_index}")

        threads = []
        i = 0
        while i < len(self.contours):
            while len(threads) < MAX_EXTRACTION_THREADS and i < len(self.contours):
                t = threading.Thread(target=parallel_polygon_extraction, args=(self.contours[i], i))
                t.start()
                threads.append(t)
                i += 1
            while len(threads) == MAX_EXTRACTION_THREADS:
                time.sleep(THREAD_POOLING_TIME_SEC)
                for thread in threads:
                    if thread.is_alive():
                        continue
                    else:
                        thread.join()
                        threads.remove(thread)

        print(f"all threads launched, waiting for remaining {len(threads)} threads")
        for remaining_thread in threads:
            remaining_thread.join()

# rose: draw_contours_only=True will show extracted contours in
# PatchExtractor_config.DOWN_SCALED_IMAGE_ANNOTATED_CONTOURS_OUTPUT_DIR_PATH
def iterate_camelyon17_files_extract_patches(draw_contours_only=False):
    for root,dirs,files in os.walk(os.path.join(NETWORK_DATABASE,'Camelyon17')):
        for filename in files:
            if filename.endswith('.tif'):  # Found WSI
                if check_in_skip_list(filename):
                    continue  # skip the file
                file_path = os.path.join(root,filename)
                print("in " + file_path)
                try:
                    ext = PatchExtractor(file_path)
                    if not draw_contours_only:
                        ext.start_extract()
                except:
                    print(f"Error when extracting from {file_path}, To debug remove try except block and run, Do this only if you know what you are doing")



from WSI_Tools.PatchExtractor_Tools.wsi_utils import form_wsi_path_by_ID
from utils.Baseline_config import INTERESTING_WSI_IDS
rose = [(4, 4), (9, 1), (10, 4), (12, 0), (15, 1), (15, 2), (16, 1), (17, 1), (17, 2), (17, 4), (20, 2), (20, 4), (21, 3), (22, 4), (24, 1), (24, 2), (34, 3), (36, 3), (38, 2), (39, 1), (40, 2), (41, 0), (42, 3), (44, 4), (45, 1), (46, 3), (46, 4), (48, 1), (51, 2), (52, 1), (60, 3), (61, 4), (62, 2), (64, 0), (66, 2), (67, 4), (68, 1), (72, 0), (73, 1), (75, 4), (80, 1), (81, 4), (86, 0), (86, 4), (87, 0), (88, 1), (89, 3), (92, 1), (96, 0), (99, 4)]
extra_negative_slides = [(0,0),(3,0),(22,2),(30,1),(38,3),(45,2),(51,3),(64,1),(77,0),(84,4),(93,0)]
def iterate_camelyon17_interesting_files_extract_patches(draw_contours_only=False):
    for file_idx in rose+extra_negative_slides: # TODO: get list from rose
        filepath = form_wsi_path_by_ID(file_idx)
        if check_in_skip_list(os.path.split(filepath)[-1]):
            continue  # skip the file
        print("in " + os.path.split(filepath)[-1])
        try:
            ext = PatchExtractor(filepath)
            if not draw_contours_only:
                ext.start_extract()
        except:
            print(f"Error when extracting from {filepath}, To debug remove try except block and run, Do this only if you know what you are doing")


