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
DOWN_SCALED_IMAGE_ANNOTATED_CONTOURS_OUTPUT_DIR_PATH = os.path.join(NETWORK_DATABASE, 'Extractor_Contours_test_4_10')
if not os.path.isdir(DOWN_SCALED_IMAGE_ANNOTATED_CONTOURS_OUTPUT_DIR_PATH):
    try:
        os.mkdir(DOWN_SCALED_IMAGE_ANNOTATED_CONTOURS_OUTPUT_DIR_PATH, mode=777)
    except:
        print(f"Error creating dir {DOWN_SCALED_IMAGE_ANNOTATED_CONTOURS_OUTPUT_DIR_PATH}")
        raise

# all params should be center specific, So must use list of n_centers size, one for each center
DOWN_SAMPLE_RATE_FOR_GENERATING_CONTOUR_IMAGE = [100, 100, 100, 100, 100]
IMG_CONTOUR_BLUR_KERNEL_SIZE = [(11, 11), (15, 15), (7, 7), (11, 11), (17, 17)]  # TODO
CV2_THRESH_FOR_EDGES = [100, 120, 150, 100, 30]  # TODO
IMG_CONTOUR_MIN_NUM_POINTS = [50, 55, 50, 50, 50]
BLACK_COLOR_THRESH_TO_IGNORE = [210, 220, 254, 210, 210]  # TODO
DROP_WSI_SCANNER_NOISE_LINE_THRESH = [8, 6, 8, 8, 8]  #####

# annotations
annotation_xml_dir_path = os.path.join(LOCAL_DATABASE, 'Annotations')
tag_csv_file_path = os.path.join(LOCAL_DATABASE, 'Tag', 'stage_labels.csv')

# extractor parameters
DEFAULT_EXTRACTION_LEVEL = 0  # anything else is not implemented (for now)
DEFAULT_PATCH_SIZE = (256, 256)
DEFAULT_PATCH_OVERLAP = (64, 64)  # X,Y
MAX_EXTRACTION_THREADS = 8
THREAD_POOLING_TIME_SEC = 10

# extraction dirs
PYTORCH_IMAGE_DATASET_PATH = os.path.join(LOCAL_DATABASE, 'Pytorch_Dataset')
DUMMY_PYTORCH_IMAGE_DATASET_PATH = os.path.join(LOCAL_DATABASE, 'Pytorch_Dataset_dummy')
NEGATIVE_OUTPUT_DIR = os.path.join(PYTORCH_IMAGE_DATASET_PATH, 'NEGATIVE')
MACRO_OUTPUT_DIR = os.path.join(PYTORCH_IMAGE_DATASET_PATH, 'MACRO')
MICRO_OUTPUT_DIR = os.path.join(PYTORCH_IMAGE_DATASET_PATH, 'MACRO')
ITC_OUTPUT_DIR = os.path.join(PYTORCH_IMAGE_DATASET_PATH, 'MACRO')
# TODO
# MICRO_OUTPUT_DIR = os.path.join(PYTORCH_IMAGE_DATASET_PATH, 'MICRO')
# ITC_OUTPUT_DIR = os.path.join(PYTORCH_IMAGE_DATASET_PATH, 'ITC')

# TODO: rose look at the extracted contours images first, then manually disable bad slides by adding their name here
# you can find them at DOWN_SCALED_IMAGE_ANNOTATED_CONTOURS_OUTPUT_DIR_PATH, remember to pass the right variable to
# iterate_camelyon17_files_extract_patches(draw_contours_only=True)

# Temp skip list
BAD_FILES_SKIP_LIST = [
    "patient_xxx_node_y",
    "patient_004_node_3",
    "patient_005_node_3",  ####
    "patient_009_node_1",
    "patient_011_node_3",
    "patient_012_node_1",  ####
    "patient_013_node_0",  ####
    "patient_013_node_3",  ####
    "patient_018_node_2",  ####
    "patient_019_node_0",
    "patient_019_node_2",
    "patient_020_node_2",
    "patient_022_node_2",
    "patient_024_node_4",
    "patient_026_node_4",
    "patient_027_node_0",
    "patient_027_node_2",
    "patient_029_node_1",
    "patient_034_node_1",
    "patient_035_node_2",
    "patient_035_node_3",
    "patient_036_node_1",
    "patient_036_node_2",
    "patient_039_node_1",
    "patient_099_node_4",
]


def check_in_skip_list(filename):
    for name in BAD_FILES_SKIP_LIST:
        if name in filename:
            return True
    return False


annotated_wsi = [(4, 4), (9, 1), (10, 4), (12, 0), (15, 1), (15, 2), (16, 1), (17, 1), (17, 2), (17, 4), (20, 2),
                 (20, 4), (21, 3), (22, 4), (24, 1), (24, 2), (34, 3), (36, 3), (38, 2), (39, 1), (40, 2), (41, 0),
                 (42, 3), (44, 4), (45, 1), (46, 3), (46, 4), (48, 1), (51, 2), (52, 1), (60, 3), (61, 4), (62, 2),
                 (64, 0), (66, 2), (67, 4), (68, 1), (72, 0), (73, 1), (75, 4), (80, 1), (81, 4), (86, 0), (86, 4),
                 (87, 0), (88, 1), (89, 3), (92, 1), (96, 0),
                 #  (99, 4)
                 ]
extra_negative_slides = [(0, 0), (3, 0), (22, 2), (30, 1), (38, 3), (45, 2), (51, 3), (64, 1), (77, 0), (84, 4),
                         (93, 0), (11, 2), (11, 3), (18, 0), (22, 2), (49, 4), (57, 0), (57, 2), (57, 3), (58, 1),
                         (58, 0), (58, 2), (58, 3), (58, 4), (78, 0), (78, 1), (78, 2), (78, 3), (78, 4), (95, 4),
                         (95, 3), (95, 2)
                         ]

interesting_wsis = annotated_wsi + extra_negative_slides
