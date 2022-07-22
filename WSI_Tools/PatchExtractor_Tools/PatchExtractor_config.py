import os



# Logger
DISABLE_LOGGER = True
PATCH_EXTRACTORS_DEFAULT_LOGGER = None

# Contours
DOWN_SCALLED_IMAGE_ANNOTATED_CONTOURS_OUTPUT_DIR_PATH = r''
DOWN_SAMPLE_RATE_FOR_GENERATING_CONTOUR_IMAGE = 100
IMG_CONTOUR_BLUR_KERNEL_SIZE = (7, 7)
CV2_THRESH_FOR_EDGES = 100
IMG_CONTOUR_MIN_NUM_POINTS = 50

# annotations
annotation_xml_dir_path = r''
wsi_dir_path = r''
tag_csv_file_path = r''


# extractor parameters
DEFAULT_EXTRACTION_LEVEL = 0  # anything else is not implemented (for now)
DEFAULT_PATCH_SIZE = (512,512)
DEFAULT_PATCH_OVERLAP = (128,128)  # X,Y

# extraction dirs
NEGATIVE_OUTPUT_DIR = r''
MACRO_OUTPUT_DIR= r''
MICRO_OUTPUT_DIR= r''
ITC_OUTPUT_DIR= r''
