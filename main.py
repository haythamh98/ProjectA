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
from torchvision.transforms import ColorJitter

import openslide
# from CLAM.wsi_core.WholeSlideImage import WholeSlideImage


from WSI_Tools.PatchExtractor_Tools.AnnotationHandler import XMLAnnotationHandler
from WSI_Tools.PatchExtractor_Tools.PatchExtractor import PatchTag,PatchExtractor

from sklearn.manifold import TSNE


def big_file_Walkaround():
    PIL.Image.MAX_IMAGE_PIXELS = 933120000  # 21630025728 * 10


if __name__ == '__main__':
    print("Hi")
    # big_file_Walkaround()
    # for windows only
    # os.add_dll_directory(r"C:\Users\haytham\Downloads\openslide-win64-20171122\openslide-win64-20171122\bin")

