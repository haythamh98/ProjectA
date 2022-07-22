import os

from enum import Enum


class PatchTag(Enum):
    NONE = -1
    NEGATIVE = 0
    MACRO = 1
    MICRO = 2
    ITC = 3


# Common
MY_NAME = 'hawahaitam'  # '@staff.technion.ac.il'
NETWORK_DATABASE = os.path.join('/', 'databases', MY_NAME)
LOCAL_DATABASE = os.path.join('/', 'data', MY_NAME)

# Logger
DISABLE_LOGGER = True
PATCH_EXTRACTORS_DEFAULT_LOGGER = None

# Contours
DOWN_SCALLED_IMAGE_ANNOTATED_CONTOURS_OUTPUT_DIR_PATH = os.path.join(LOCAL_DATABASE, 'Extractor_Contours_test_22_7')
DOWN_SAMPLE_RATE_FOR_GENERATING_CONTOUR_IMAGE = 100
IMG_CONTOUR_BLUR_KERNEL_SIZE = (11, 11)
CV2_THRESH_FOR_EDGES = 100
IMG_CONTOUR_MIN_NUM_POINTS = 50
BLACK_COLOR_THRESH_TO_IGNORE = 210
DROP_WSI_SCANNER_NOISE_LINE_THRESH = 8

# annotations
annotation_xml_dir_path = os.path.join(LOCAL_DATABASE, 'Annotations')
tag_csv_file_path = os.path.join(LOCAL_DATABASE, 'Tag', 'stage_labels.csv')

# extractor parameters
DEFAULT_EXTRACTION_LEVEL = 0  # anything else is not implemented (for now)
DEFAULT_PATCH_SIZE = (512, 512)
DEFAULT_PATCH_OVERLAP = (128, 128)  # X,Y
MAX_EXTRACTION_THREADS = 8
THREAD_POOLING_TIME_SEC = 10

# extraction dirs
NEGATIVE_OUTPUT_DIR = os.path.join(LOCAL_DATABASE, 'Pytorch_Dataset', 'NEGATIVE')
MACRO_OUTPUT_DIR = os.path.join(LOCAL_DATABASE, 'Pytorch_Dataset', 'MACRO')
MICRO_OUTPUT_DIR = os.path.join(LOCAL_DATABASE, 'Pytorch_Dataset', 'MICRO')
ITC_OUTPUT_DIR = os.path.join(LOCAL_DATABASE, 'Pytorch_Dataset', 'ITC')
