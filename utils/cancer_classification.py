# import the necessary packages
import sys

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from PIL import Image
import openslide
sys.path.append('../')
from WSI_Tools.PatchExtractor_Tools.wsi_utils import form_wsi_path_by_ID


def calculate_ratio(patient, node, diameter, patch_size):

    whole_slide = openslide.open_slide(form_wsi_path_by_ID((int(patient), int(node))))
    while_slide_dims = whole_slide.level_dimensions[0]
    x_resolution = whole_slide.properties['tiff.XResolution']
    x_resolution_mm = float(x_resolution)/10
    # print(whole_slide.properties['tiff.XResolution'],whole_slide.properties['tiff.ResolutionUnit'] )
    real_diameter = (diameter * patch_size)/x_resolution_mm
    print(real_diameter)
    if real_diameter < 0.2:  #0.0078740157:  # 0.2mm in inches
        print("ITC")

    elif real_diameter > 0.2 and real_diameter < 2: #0.0787401575:
        print("MICRO")

    else:
        print("MACRO")
    return



def cancer_classification(image_path):
    # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True,
    #                 help="path to the input image")
    #
    # args = vars(ap.parse_args())

    # load the image, convert it to grayscale, and blur it slightly
    patient = image_path.split("_")[2]
    node = image_path.split("_")[4]


    image = cv2.imread(image_path)
    # print(type(image[0][0][0]))
    img = Image.fromarray(image)
    img.save('input_img.png')

    # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
    width, height = image.shape[0],  image.shape[1]
    for x in range(0, width):
        for y in range(0, height):
            if image[x, y, 0] >= 250 and image[x, y, 1] >= 250 and image[x, y, 2] >= 250:
                image[x, y] = [0, 0, 0].copy()
            if image[x, y, 0] >= 250 and image[x, y, 1] <= 10 and image[x, y, 2] <= 10:
                image[x, y] = [0, 0, 255].copy()


    img = Image.fromarray(image)
    img.save('flip_colors.png')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    img = Image.fromarray(gray)
    img.save('flip_colors.png')
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(gray, kernel, iterations=1)
    edged = cv2.erode(edged, kernel, iterations=1)
    edged = cv2.Canny(edged, 50, 100)
    img = Image.fromarray(edged)
    img.save('my3.png')

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable

    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None
    # loop over the contours individually
    for c in cnts:
        area = cv2.contourArea(c)
        diameter_in_pixels = np.sqrt(4*area/np.pi)

        calculate_ratio(patient, node, diameter_in_pixels, 2*256)


#python cancer_classification.py --image "/databases/hawahaitam/heatmaps/heatmap_patient_9_node_1_img.png"


if __name__ == "__main__":
    image_path = "/databases/hawahaitam/heatmaps/heatmap_patient_38_node_2_img.png"
    cancer_classification(image_path)