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
from WSI_Tools.PatchExtractor_Tools.PatchExtractor import PatchTag, PatchExtractor, \
    iterate_camelyon17_files_extract_patches
from Baseline import *
from sklearn.manifold import TSNE
from WSI_Tools.PatchExtractor_Tools.PatchExtractor_config import *



def big_file_Walkaround():
    PIL.Image.MAX_IMAGE_PIXELS = 933120000  # 21630025728 * 10




if __name__ == '__main__':
    print("Hi")
    # Rose: some functions that will help you
    # iterate_camelyon17_files_extract_patches(draw_contours_only=True) in PatchExtractor
    # draw_random_samples() in Dataset
    # knn_sanity_check() in PT_Resnet50_KNN

    # TODO:
    # 1) fix extraction parameters for at least 2 centers (center 0 is done)
    # 2) run contours extraction to check all images , if bad image add wsi name to skiplist as in example, in patchExtractor_config file
    # 3) run extract slides (you might need to delete macro file, TODO: make sure micro has patches)
    # 4) you might need to use GPU for faster processing

    iterate_camelyon17_files_extract_patches(draw_contours_only=False)
    # knn_sanity_check()

    # projectA_run_baseline_heatmap_build([1,2])



