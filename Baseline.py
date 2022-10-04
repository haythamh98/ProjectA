import os.path
import sys

import torchvision
from matplotlib import pyplot as plt

from PIL.Image import Image
from shapely.geometry import Polygon, Point
import shutil
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

############# logger
date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")

logger = logging.getLogger(__name__)
try:
    logging.basicConfig(filename=rf"../logging/projectA_baseline_{date}",
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
except:
    pass

# logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)
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


    # train on allset/[interesting_slide] then validate on [interesting_slide] then export results,heatmap
    for interesting_slide in INTERESTING_WSI_IDS[13:]:  # TODO:::::
        logger.info(f"using all dataset for train, then validating {interesting_slide}")
        logger.debug("init dataset & dataloader")
        Dataset.init3333_ds(validation_WSI_IDs=[interesting_slide], use_dummy_ds=False, only_train_set=True)
        logger.debug("done init dataset & dataloader")
        logger.debug(f"using {BASELINE_N_BATCHES_FOR_KNN * BATCH_SIZE} patches for trainset")

        # train data for KNN : resnet50 forward
        resnt50_output_tensor = torch.zeros(size=(BASELINE_N_BATCHES_FOR_KNN * BATCH_SIZE, 2048))
        tags_tensor = torch.zeros(BASELINE_N_BATCHES_FOR_KNN * BATCH_SIZE)
        with torch.no_grad():
            for i, xxx in zip(range(BASELINE_N_BATCHES_FOR_KNN), Dataset.camelyon17_train_dl):
                if i % 10 == 0:  # print every 10
                    logger.debug(f"processing batch num {i} out of {BASELINE_N_BATCHES_FOR_KNN}")
                imges_tensort, tags = xxx[0], xxx[1]
                resnt50_output_tensor[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = \
                    PT_Resnet50_KNN.pt_resnet50_model_cut.forward(imges_tensort).squeeze()
                tags_tensor[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = tags

        # init knn
        logger.debug(f"Init-ing KNN now, with k={KNN_K}")
        to_knn_train_fit = [resnt50_output_tensor,
                            tags_tensor]  # must stack , or change innnir implementation of init_Knn_model
        PT_Resnet50_KNN.init_Knn_model(train_dataset=to_knn_train_fit, n_neighbors=KNN_K)

        TEMP_EXTRACTION_PATH_FOR_TEST_WSI = os.path.join('/', 'databases', 'hawahaitam', 'temp_dir', 'temp_dir')
        logger.debug(f"Done, now extracting full WSI to {TEMP_EXTRACTION_PATH_FOR_TEST_WSI}, clearing it first")
        try:
            shutil.rmtree(TEMP_EXTRACTION_PATH_FOR_TEST_WSI)
            os.mkdir(TEMP_EXTRACTION_PATH_FOR_TEST_WSI)
        except:
            TEMP_EXTRACTION_PATH_FOR_TEST_WSI = os.path.join(os.path.split(TEMP_EXTRACTION_PATH_FOR_TEST_WSI)[0],
                                                             f'stam_{datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")}')
            logger.debug(f"error deleting dir, try to create new at {TEMP_EXTRACTION_PATH_FOR_TEST_WSI}")
            os.mkdir(TEMP_EXTRACTION_PATH_FOR_TEST_WSI)
            # if doesnt work then die
        # extract [interesting_slide] without overlap
        logger.debug(f"start extract")
        print(f"start extract of slide {interesting_slide}")
        extr = PatchExtractor(
            wsi_path=form_wsi_path_by_ID(interesting_slide),
            patch_size=DEFAULT_PATCH_SIZE,
            patch_overlap=(0, 0),
            negative_output_dir=TEMP_EXTRACTION_PATH_FOR_TEST_WSI,
            macro_output_dir=TEMP_EXTRACTION_PATH_FOR_TEST_WSI,
            micro_output_dir=TEMP_EXTRACTION_PATH_FOR_TEST_WSI,
            itc_output_dir=TEMP_EXTRACTION_PATH_FOR_TEST_WSI,
            down_scaled_image_annotated_boundaries_output_dir_path=DOWN_SCALED_IMAGE_ANNOTATED_CONTOURS_OUTPUT_DIR_PATH,
            logger=PATCH_EXTRACTORS_DEFAULT_LOGGER,
        )
        extr.start_extract()
        logger.debug(f"done extract")

        # build heatmap, resnet forward + knn predict
        heatmap_tensor = torch.zeros((3, (extr.wsi.dimensions[0] // 512) + 1, (extr.wsi.dimensions[1] // 512) + 1))
        pred_heatmap_tensor = torch.zeros(((extr.wsi.dimensions[0] // 512) + 1, (extr.wsi.dimensions[1] // 512) + 1))
        one_wsi_patches_names = os.listdir(TEMP_EXTRACTION_PATH_FOR_TEST_WSI)
        to_tensor_transform = transforms.ToTensor()

        for x in range(0, len(one_wsi_patches_names), BATCH_SIZE):
            print(f"in batch num {x//BATCH_SIZE} out of {len(one_wsi_patches_names)//BATCH_SIZE} ")
            tmp_tensor = torch.zeros([64, 3, 512, 512])
            for y in range(0, min(BATCH_SIZE, len(one_wsi_patches_names) - x - 1)):
                img = Image.open(os.path.join(TEMP_EXTRACTION_PATH_FOR_TEST_WSI, one_wsi_patches_names[x + y])).convert(
                    'RGB')
                tmp_tensor[y] = to_tensor_transform(img)
                img.close()
            forward = PT_Resnet50_KNN.pt_resnet50_model_cut(tmp_tensor).squeeze()
            predictions = PT_Resnet50_KNN.knn_predict(forward)
            del forward  # TODO: CHECK HIGH MEMORY USAGE
            del tmp_tensor
            # print("predic", predictions)
            for y in range(0, min(BATCH_SIZE, len(one_wsi_patches_names) - x - 1)):
                # patient_000_node_0.tif_xy_95924_90156_512x512.png
                img_name = one_wsi_patches_names[x + y].split('_')
                i = int(img_name[5]) // 512
                j = int(img_name[6]) // 512
                predi = int(predictions[y])
                pred_heatmap_tensor[i][j] = int(predictions[y])
                if predi != 0:
                    # print("ij,",i,j)
                    heatmap_tensor[predi - 1][i][j] = 1
                    # RED: MACRO/MICRO
                    # BLUE: ITC
                else:
                    # WHITE: NEGATIVE
                    heatmap_tensor[0][i][j] = 1
                    heatmap_tensor[1][i][j] = 1
                    heatmap_tensor[2][i][j] = 1

        # evaluate results
        negative_as_negative = 0
        itc_as_itc = 0
        macro_as_macro = 0
        negative_as_macro = 0
        negative_as_itc = 0
        itc_as_negative = 0
        itc_as_macro = 0
        macro_as_negative = 0
        macro_as_itc = 0

        for i in range(pred_heatmap_tensor.shape[0]):
            for j in range(pred_heatmap_tensor.shape[1]):
                patch_poly = Polygon(
                    [Point(i * 512, j * 512), Point(i * 512 + 512, j * 512), Point(i * 512 + 512, j * 512 + 512),
                     Point(i * 512, j * 512 + 512)])

                patch_tag = extr.classify_metastasis_polygon(patch_poly)
                if patch_tag == PatchTag.NEGATIVE:
                    if pred_heatmap_tensor[i][j] == 0:
                        negative_as_negative += 1
                    elif pred_heatmap_tensor[i][j] == 1 or pred_heatmap_tensor[i][j] == 2:
                        negative_as_macro += 1
                    elif pred_heatmap_tensor[i][j] == 3:
                        negative_as_itc += 1
                elif patch_tag == PatchTag.MACRO or patch_tag == PatchTag.MICRO:
                    if pred_heatmap_tensor[i][j] == 0:
                        macro_as_negative += 1
                    elif pred_heatmap_tensor[i][j] == 1 or pred_heatmap_tensor[i][j] == 2:
                        macro_as_macro += 1
                    elif pred_heatmap_tensor[i][j] == 3:
                        macro_as_itc += 1
                elif patch_tag == PatchTag.ITC:
                    if pred_heatmap_tensor[i][j] == 0:
                        itc_as_negative += 1
                    elif pred_heatmap_tensor[i][j] == 1 or pred_heatmap_tensor[i][j] == 2:
                        itc_as_macro += 1
                    elif pred_heatmap_tensor[i][j] == 3:
                        itc_as_itc += 1

        logger.info(f"negative_as_negative = {negative_as_negative}")
        logger.info(f"itc_as_itc = {itc_as_itc}")
        logger.info(f"macro_as_macro = {macro_as_macro}")
        logger.info(f"miss classify negative_as_macro = {negative_as_macro}")
        logger.info(f"miss classify negative_as_itc = {negative_as_itc}")
        logger.info(f"miss classify itc_as_negative = {itc_as_negative}")
        logger.info(f"miss classify itc_as_macro = {itc_as_macro}")
        logger.info(f"miss classify macro_as_negative = {macro_as_negative}")
        logger.info(f"miss classify macro_as_itc = {macro_as_itc}")

        heatmap_pil_img = heatmap(heatmap_tensor, show=False)
        heatmap_pil_img.save(rf"../heatmap_patient_{extr.patient_ID}_node_{extr.patient_node_ID}_img.png")
        torch.save(heatmap_tensor, rf"../heatmap_patient_{extr.patient_ID}_node_{extr.patient_node_ID}.pt")


def projectA_run_baseline_for_patches_only(validation_WSI_IDs, BASELINE_N_BATCHES_FOR_KNN, test_n_batches,
                                           use_dummy_ds=False):
    Dataset.init3333_ds(validation_WSI_IDs, use_dummy_ds=use_dummy_ds)
    PT_Resnet50_KNN.init_pre_trained_resnet50_model()

    resnt50_output_tensor = torch.zeros(size=(BASELINE_N_BATCHES_FOR_KNN * BATCH_SIZE, 2048))
    tags_tensor = torch.zeros(BASELINE_N_BATCHES_FOR_KNN * BATCH_SIZE)
    print("initing samples for knn")
    with torch.no_grad():
        for i, xxx in zip(range(BASELINE_N_BATCHES_FOR_KNN), Dataset.camelyon17_train_dl):
            print(f"processing train batch num {i} out of {BASELINE_N_BATCHES_FOR_KNN}")
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
