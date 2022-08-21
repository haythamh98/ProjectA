from WSI_Tools.PatchExtractor_Tools.PatchExtractor_config import *
KNN_K = 7
BASELINE_N_BATCHES_FOR_KNN = 200



# TODO: TEMP_EXTRACTION_PATH_FOR_TEST_WSI  = os.path.join('/','home',MY_NAME,'temp_dir')

INTERESTING_WSI_IDS = [
    # center 0
    tuple((9, 1)),
    tuple((10, 4)),
    tuple((12, 0)),
    tuple((15, 1)),
    tuple((15, 2)),
    tuple((16, 1)),
    tuple((17, 1)),
    tuple((17, 4)),
    # center 1
    tuple((20, 2)),
    tuple((20, 4)),
    tuple((21, 3)),
    tuple((22, 4)),
    tuple((24, 1)),
    tuple((24, 2)),
    tuple((34, 3)),
    tuple((36, 3)),
    tuple((38, 2)),
    tuple((39, 1)),
    # center 2
    tuple((40, 2)),  # SLIDE TAG: itc
    tuple((41, 0)),  # SLIDE TAG: itc
    tuple((42, 3)),  #
    tuple((44, 4)),  #
    tuple((45, 1)),  # SLIDE TAG: micro
    tuple((46, 3)),  #
    tuple((46, 4)),  #
    tuple((48, 1)),  # SLIDE TAG: micro
]