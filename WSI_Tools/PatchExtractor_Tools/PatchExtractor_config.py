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
N_DATA_CENTERS = 5

# Logger
DISABLE_LOGGER = True
PATCH_EXTRACTORS_DEFAULT_LOGGER = None

# Contours
DOWN_SCALED_IMAGE_ANNOTATED_CONTOURS_OUTPUT_DIR_PATH = os.path.join(NETWORK_DATABASE, 'Extractor_Contours_test_27_07')
if not os.path.isdir(DOWN_SCALED_IMAGE_ANNOTATED_CONTOURS_OUTPUT_DIR_PATH):
    try:
        os.mkdir(DOWN_SCALED_IMAGE_ANNOTATED_CONTOURS_OUTPUT_DIR_PATH,mode=777)
    except:
        print(f"Error creating dir {DOWN_SCALED_IMAGE_ANNOTATED_CONTOURS_OUTPUT_DIR_PATH}")
        raise


# hyper-parameters
# all params should be center specific, So must use list of n_centers size, one for each center
DOWN_SAMPLE_RATE_FOR_GENERATING_CONTOUR_IMAGE = [100, 100, 100, 100, 100, 100]
IMG_CONTOUR_BLUR_KERNEL_SIZE = [(11, 11), (15, 15), (9, 9), (11, 11), (11, 11), (11, 11)]  # TODO
CV2_THRESH_FOR_EDGES = [100, 120, 100, 100, 100, 100]  # TODO
IMG_CONTOUR_MIN_NUM_POINTS = [50, 55, 50, 50, 50, 50]
BLACK_COLOR_THRESH_TO_IGNORE = [210, 220, 230, 210, 210]  # TODO
DROP_WSI_SCANNER_NOISE_LINE_THRESH = [8, 6, 8, 8, 8]#####

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
PYTORCH_IMAGE_DATASET_PATH =  os.path.join(LOCAL_DATABASE, 'Pytorch_Dataset')
NEGATIVE_OUTPUT_DIR = os.path.join(PYTORCH_IMAGE_DATASET_PATH, 'NEGATIVE')
MACRO_OUTPUT_DIR = os.path.join(PYTORCH_IMAGE_DATASET_PATH, 'MACRO')
MICRO_OUTPUT_DIR = os.path.join(PYTORCH_IMAGE_DATASET_PATH, 'MICRO')
ITC_OUTPUT_DIR = os.path.join(PYTORCH_IMAGE_DATASET_PATH, 'ITC')

# TODO: rose look at the extracted contours images first, then manually disable bad slides by adding their name here
# you can find them at DOWN_SCALED_IMAGE_ANNOTATED_CONTOURS_OUTPUT_DIR_PATH, remember to pass the right variable to
# iterate_camelyon17_files_extract_patches(draw_contours_only=True)

# Temp skip list
BAD_FILES_SKIP_LIST = [
    "patient_xxx_node_y",
    "patient_004_node_3",
    "patient_005_node_3", ####
    "patient_009_node_1",
    "patient_011_node_3",
    "patient_012_node_1", ####
    "patient_013_node_0", ####
    "patient_013_node_3", ####
    "patient_018_node_2", ####
    
    
    
]
def check_in_skip_list(filename):
    for name in BAD_FILES_SKIP_LIST:
        if name in filename:
            return True
    return False
