import os.path

import torchvision
from matplotlib import pyplot as plt

from PIL.Image import Image

from utils.Baseline_config import *
from Models import PT_Resnet50_KNN
from WSI_Tools.PatchExtractor_Tools.PatchExtractor_config import *
from utils import Dataset
import torch
from utils import Dataset
from utils.Heatmap import heatmap
from utils.Heatmap import *
from utils.Dataset_config import *
import logging
from datetime import datetime
from WSI_Tools.PatchExtractor_Tools.PatchExtractor import PatchTag, PatchExtractor
from WSI_Tools.PatchExtractor_Tools.wsi_utils import form_wsi_path_by_ID
from utils.Baseline_config import *
from utils.Baseline_config import TEMP_EXTRACTION_PATH_FOR_TEST_WSI

############# logger
date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")

logger = logging.getLogger(__name__)
logging.basicConfig(filename=rf"../logging/projectA_baseline_{date}",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)


def get_random_samples_resnet50_forward_all_tags(nBatches: int = 1):
    if PT_Resnet50_KNN.pt_resnet50_model_cut is None:
        print("initinnngg resnet50")
        PT_Resnet50_KNN.init_pre_trained_resnet50_model()
    BATCH_SIZE = 64  # for now

    # print("camelyon17_ds")
    # print(camelyon17_ds.class_to_idx)

    result_tensor = torch.zeros(size=(nBatches * BATCH_SIZE, 2048))
    tags_tensor = torch.zeros(nBatches * BATCH_SIZE)
    with torch.no_grad():
        # class_idx_samples_counter = [0] * 4
        for i, Xy in zip(range(nBatches), Dataset.camelyon17_dl):
            # print(f"extracted {i} samples out of {nBatches}")
            # data = next(iter(data_loader))
            X, y = Xy[0], Xy[1]  # X,y shape[0] == BATCH_SIZE
            result_tensor[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = PT_Resnet50_KNN.pt_resnet50_model_cut.forward(
                X).squeeze()
            tags_tensor[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = y
    return result_tensor, tags_tensor


def projectA_run_baseline():
    global logger
    PT_Resnet50_KNN.init_pre_trained_resnet50_model()
    logger.debug("hello debug mode")

    interesting_slides = [
        tuple((0, 0))
    ]
    for interesting_slide in interesting_slides:
        logger.info(f"using all dataset for train, then validating {interesting_slide}")
        logger.debug("init dataset & dataloader")
        Dataset.init3333_ds(validation_WSI_IDs=[interesting_slide], use_dummy_ds=False, only_train_set=True)
        logger.debug("done init dataset & dataloader")
        train_n_batches = BASELINE_N_BATCHES_FOR_KNN
        logger.debug(f"using {train_n_batches * BATCH_SIZE} patches for trainset")
        resnt50_output_tensor = torch.zeros(size=(train_n_batches * BATCH_SIZE, 2048))
        tags_tensor = torch.zeros(train_n_batches * BATCH_SIZE)
        with torch.no_grad():
            for i, xxx in zip(range(train_n_batches), Dataset.camelyon17_train_dl):
                if 1 % 10 == 0:
                    logger.debug(f"processing batch num {i} out of {train_n_batches}")
                imges_tensort, tags = xxx[0], xxx[1]
                resnt50_output_tensor[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = \
                    PT_Resnet50_KNN.pt_resnet50_model_cut.forward(imges_tensort).squeeze()
                tags_tensor[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = tags
        logger.debug(f"Init-ing KNN now, with k={KNN_K}")
        to_knn_train_fit = [resnt50_output_tensor,
                            tags_tensor]  # must stack , or change innnir implementation of init_Knn_model
        PT_Resnet50_KNN.init_Knn_model(train_dataset=to_knn_train_fit, n_neighbors=KNN_K)
        print("KNN_K", KNN_K)
        TEMP_EXTRACTION_PATH_FOR_TEST_WSI = os.path.join('/', 'databases', 'hawahaitam', 'temp_dir', 'temp_dir')
        logger.debug(f"Done, now extracting full WSI to {TEMP_EXTRACTION_PATH_FOR_TEST_WSI}, clearing it first")
        try:
            import shutil
            # TODO shutil.rmtree(TEMP_EXTRACTION_PATH_FOR_TEST_WSI)
            # os.mkdir(TEMP_EXTRACTION_PATH_FOR_TEST_WSI)
        except:
            TEMP_EXTRACTION_PATH_FOR_TEST_WSI = os.path.join(os.path.split(TEMP_EXTRACTION_PATH_FOR_TEST_WSI)[0],
                                                             f'stam_{datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")}')
            logger.debug(f"error deleting dir, try to create new at {TEMP_EXTRACTION_PATH_FOR_TEST_WSI}")
            os.mkdir(TEMP_EXTRACTION_PATH_FOR_TEST_WSI)
            # if doesnt work then die
        # try:
        logger.debug(f"start extract")
        print(f"start extract")
        extr = PatchExtractor(
            wsi_path=form_wsi_path_by_ID(interesting_slide),
            xml_dir_path='hell',  # so it wont use annotation, and classify all as negative
            patch_size=DEFAULT_PATCH_SIZE,
            patch_overlap=(0, 0),
            negative_output_dir=TEMP_EXTRACTION_PATH_FOR_TEST_WSI,
            macro_output_dir=TEMP_EXTRACTION_PATH_FOR_TEST_WSI,
            micro_output_dir=TEMP_EXTRACTION_PATH_FOR_TEST_WSI,
            itc_output_dir=TEMP_EXTRACTION_PATH_FOR_TEST_WSI,
            down_scaled_image_annotated_boundaries_output_dir_path=DOWN_SCALED_IMAGE_ANNOTATED_CONTOURS_OUTPUT_DIR_PATH,
            logger=PATCH_EXTRACTORS_DEFAULT_LOGGER,
        )
        # TODO extr.start_extract()
        logger.debug(f"done extract")

        heatmap_tensor = torch.zeros((3,extr.wsi.dimensions[0] // 512, extr.wsi.dimensions[1] // 512))
        one_wsi_patches_names = os.listdir(TEMP_EXTRACTION_PATH_FOR_TEST_WSI)
        to_tensor_transform = transforms.ToTensor()

        for x in range(0, len(one_wsi_patches_names), BATCH_SIZE):
            print(f"in batch num {x} out of {len(one_wsi_patches_names)} ")
            tmp_tensor = torch.zeros([64, 3, 512, 512])
            for y in range(0, BATCH_SIZE):
                img = Image.open(os.path.join(TEMP_EXTRACTION_PATH_FOR_TEST_WSI, one_wsi_patches_names[x + y])).convert(
                    'RGB')
                tmp_tensor[y] = to_tensor_transform(img)
                img.close()
            forward = PT_Resnet50_KNN.pt_resnet50_model_cut(tmp_tensor).squeeze()
            predictions = PT_Resnet50_KNN.knn_predict(forward)
            del forward
            del tmp_tensor
            print("predic", predictions)
            for y in range(0, BATCH_SIZE):
                # patient_000_node_0.tif_xy_95924_90156_512x512.png
                img_name = one_wsi_patches_names[x + y].split('_')
                i = int(img_name[5]) // 512
                j = int(img_name[6]) // 512
                predi = int(predictions[y])
                if predi != 0:
                    heatmap_tensor[predi-1][i][j] = 1


        heatmap(heatmap_tensor)

        # torch.save(heatmap_tensor, r"../heatmap.pt")

        # TODO: last 64-


def projectA_run_baseline_for_patches_only(validation_WSI_IDs, train_n_batches, test_n_batches, use_dummy_ds=False):
    Dataset.init3333_ds(validation_WSI_IDs, use_dummy_ds=use_dummy_ds)
    PT_Resnet50_KNN.init_pre_trained_resnet50_model()

    resnt50_output_tensor = torch.zeros(size=(train_n_batches * BATCH_SIZE, 2048))
    tags_tensor = torch.zeros(train_n_batches * BATCH_SIZE)
    print("initing samples for knn")
    with torch.no_grad():
        for i, xxx in zip(range(train_n_batches), Dataset.camelyon17_train_dl):
            print(f"processing train batch num {i} out of {train_n_batches}")
            imges_tensort, tags = xxx[0], xxx[1]
            resnt50_output_tensor[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = \
                PT_Resnet50_KNN.pt_resnet50_model_cut.forward(imges_tensort).squeeze()
            tags_tensor[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = tags

    print("initing KNN")
    to_knn_train_fit = [resnt50_output_tensor,
                        tags_tensor]  # must stack , or change innnir implementation of init_Knn_model
    PT_Resnet50_KNN.init_Knn_model(train_dataset=to_knn_train_fit, n_neighbors=7)

    TEST_resnt50_output_tensor = torch.zeros(size=(test_n_batches * BATCH_SIZE, 2048))
    TEST_tags_tensor = torch.zeros(test_n_batches * BATCH_SIZE)
    print("initing test samples for knn")
    with torch.no_grad():
        for i, xxx in zip(range(test_n_batches), Dataset.camelyon17_validation_dl):
            print(f"processing train batch num {i} out of {test_n_batches}")
            imges_tensort, tags = xxx[0], xxx[1]
            TEST_resnt50_output_tensor[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = \
                PT_Resnet50_KNN.pt_resnet50_model_cut.forward(imges_tensort).squeeze()
            TEST_tags_tensor[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = tags

    PT_Resnet50_KNN.knn_validate(TEST_resnt50_output_tensor, TEST_tags_tensor, use_print=True)
