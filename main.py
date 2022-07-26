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
from utils.Dataset import *
from Models.PT_Resnet50_KNN import *
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

# rose: draw_contours_only=True will show extracted contours in
# PatchExtractor_config.DOWN_SCALED_IMAGE_ANNOTATED_CONTOURS_OUTPUT_DIR_PATH
def iterate_camelyon17_files_extract_patches(draw_contours_only=False):
    for root,dirs,files in os.walk(os.path.join(LOCAL_DATABASE,'Camelyon17')):
        for filename in files:
            if filename.endswith('.tif'):  # Found WSI
                file_path = os.path.join(root,filename)
                print("in " + file_path)
                try:
                    ext = PatchExtractor(file_path)
                    if not draw_contours_only:
                        ext.start_extract()
                except:
                    print(f"Error when extracting from {file_path}, To debug remove try except block and run, Do this only if you know what you are doing")


if __name__ == '__main__':
    print("Hi")
    # Rose: some functions that will help you
    # iterate_camelyon17_files_extract_patches(draw_contours_only=True)
    # draw_random_samples()
    forw,y = get_random_samples_resnet50_forward(3)
    toknn = (forw[32:],y[32:]) # must stack , or change innnir implementation of init_Knn_model
    X_test = forw[:32]
    y_test = y[:32]
    init_Knn_model(toknn,1)
    y_pred = knn_model.predict(X_test)
    miss_classifications = y_test - y_pred
    print(f"miss classified {torch.count_nonzero(miss_classifications)} out of {len(y_test)}")
    accu = 1 - torch.count_nonzero(miss_classifications) / len(y_test)
    print(f"Accuracy {accu} ")
    
    exit()
    for i in range(0,5):
        wsi_path = rf'/home/hawahaitam/data/Camelyon17/training/center_0/patient_014/patient_014_node_{i}.tif'
        ext = PatchExtractor(wsi_path)
        print("in", wsi_path)
        # ext.start_extract()
    for i in range(0,5):
        wsi_path = rf'/home/hawahaitam/data/Camelyon17/training/center_0/patient_007/patient_007_node_{i}.tif'
        ext = PatchExtractor(wsi_path)
        print("in", wsi_path)
        ext.start_extract()
    for i in range(0,2):
        wsi_path = rf'/home/hawahaitam/data/Camelyon17/training/center_0/patient_015/patient_015_node_{i}.tif'
        ext = PatchExtractor(wsi_path)
        print("in", wsi_path)
        # ext.start_extract()

    exit()


    from utils.Dataset import *
    for x in camelyon17_ds:
        print(x)
        break







    exit()
    wsi_path = r'/home/hawahaitam/data/Camelyon17/training/center_0/patient_005/patient_005_node_4.tif'
    ext = PatchExtractor(wsi_path)
    ext.start_extract()

