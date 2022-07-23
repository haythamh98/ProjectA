import os
from os import path

import PIL
import cv2
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# from openslide.deepzoom import DeepZoomGenerator
import torch
from torchvision import transforms, models


import openslide
# from CLAM.wsi_core.WholeSlideImage import WholeSlideImage


from WSI_Tools.PatchExtractor_Tools.AnnotationHandler import XMLAnnotationHandler
from WSI_Tools.PatchExtractor_Tools.PatchExtractor import PatchTag,PatchExtractor

from sklearn.manifold import TSNE
from WSI_Tools.PatchExtractor_Tools.PatchExtractor_config import *

def big_file_Walkaround():
    PIL.Image.MAX_IMAGE_PIXELS = 933120000  # 21630025728 * 10

def do_knn():
    pass



if __name__ == '__main__':
    print("Hi")
    # big_file_Walkaround()
    # for windows only
    # os.add_dll_directory(r"C:\Users\haytham\Downloads\openslide-win64-20171122\openslide-win64-20171122\bin")

    for i in range(4,5):
        wsi_path = rf'/home/hawahaitam/data/Camelyon17/training/center_2/patient_044/patient_044_node_{i}.tif'
        ext = PatchExtractor(wsi_path)
        print("in", wsi_path)
        ext.start_extract()


    exit()


    from utils.Dataset import *
    for x in camelyon17_ds:
        print(x)
        break







    exit()
    wsi_path = r'/home/hawahaitam/data/Camelyon17/training/center_0/patient_005/patient_005_node_4.tif'
    ext = PatchExtractor(wsi_path)
    ext.start_extract()

