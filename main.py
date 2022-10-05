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
from utils.Dataset_config import *


def big_file_Walkaround():
    PIL.Image.MAX_IMAGE_PIXELS = 933120000  # 21630025728 * 10




if __name__ == '__main__':
    print("Hi")

    #from WSI_Tools.PatchExtractor_Tools.PatchExtractor import *
    #iterate_camelyon17_interesting_files_extract_patches(draw_contours_only=False)
    #from utils.Folder_spliter import split_folder_to_subfolders
    #split_folder_to_subfolders(MACRO_OUTPUT_DIR)
    #split_folder_to_subfolders(NEGATIVE_OUTPUT_DIR)

    import Final_solution
    Final_solution.run_final_train_model()

    #itera = iter(ds)
    #lll = list([next(itera),next(itera),next(itera)])
    #print(lll)

    print("bye")


    #
    # Dataset.init2222_ds_dl(validation_WSI_IDs,use_dummy_ds=True)

    # projectA_run_baseline_heatmap_build([1,2])



