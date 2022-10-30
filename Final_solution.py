import random
from collections import OrderedDict
import torch
import torch.nn as nn
from PIL.Image import Image
from WSI_Tools.PatchExtractor_Tools.PatchExtractor import *
from WSI_Tools.PatchExtractor_Tools.PatchExtractor_config import *
from utils import Dataset
from torchvision import models
from utils.Dataset_config import *
from utils.Final_solution_config import *
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix
from utils.Evaluation_metrics import *
from utils.Heatmap import heatmap
from utils.summaries import TensorboardSummary
import logging

model = None
optimizer = None


def init_final_model():
    global model, optimizer
    model = models.resnet18(pretrained=True)
    model.eval()
    # print(list(model.children())[:])
    # model = torch.nn.Sequential(*(list(model.children())[:-1]))  # strips off last linear layer
    for params in model.parameters():
        params.requires_grad_ = False
    # add a new final layer
    nr_filters = model.fc.in_features  # number of input features of last layer
    model.fc = nn.Linear(nr_filters, out_features=1)
    # print("RESTENT18!*" , list(model.children())[:])
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=adam_lr, betas=adam_betas, eps=adam_eps, weight_decay=adam_weight_decay,
                     amsgrad=adam_amsgrad)


def get_random_validation_idx():
    from WSI_Tools.PatchExtractor_Tools.PatchExtractor_config import interesting_wsis
    size_ = len(interesting_wsis)
    return random.choices(interesting_wsis, weights=None, cum_weights=None, k=int(0.25 * size_) + 1)


def import_model(path: str = ""):  # TODO: rose set default
    global model
    model = None  # TODO: rose import model by path


def test_wsi(path: str = None, wsi_tuple: tuple = None, patch_size=None, patches_overlap=None):
    global model
    assert model is not None

    if wsi_tuple is not None:
        path = form_wsi_path_by_ID(wsi_tuple)
    extr = PatchExtractor(
        wsi_path=path,
        patch_size=DEFAULT_PATCH_SIZE if patch_size is None else patch_size,
        patch_overlap=(0, 0) if patches_overlap is None else patches_overlap,
        negative_output_dir=TEMP_EXTRACTION_PATH_FOR_TEST_WSI,
        macro_output_dir=TEMP_EXTRACTION_PATH_FOR_TEST_WSI,
        micro_output_dir=TEMP_EXTRACTION_PATH_FOR_TEST_WSI,
        itc_output_dir=TEMP_EXTRACTION_PATH_FOR_TEST_WSI,
        down_scaled_image_annotated_boundaries_output_dir_path=DOWN_SCALED_IMAGE_ANNOTATED_CONTOURS_OUTPUT_DIR_PATH,
        logger=PATCH_EXTRACTORS_DEFAULT_LOGGER,
    )
    print(f"start extract {path}, contour image = {DOWN_SCALED_IMAGE_ANNOTATED_CONTOURS_OUTPUT_DIR_PATH}")
    extr.start_extract()
    print("done extract")
    # build heatmap, resnet forward + knn predict
    heatmap_tensor = torch.zeros((3, (extr.wsi.dimensions[0] // 512) + 1, (extr.wsi.dimensions[1] // 512) + 1))
    pred_heatmap_tensor = torch.zeros(((extr.wsi.dimensions[0] // 512) + 1, (extr.wsi.dimensions[1] // 512) + 1))
    one_wsi_patches_names = os.listdir(TEMP_EXTRACTION_PATH_FOR_TEST_WSI)
    to_tensor_transform = transforms.ToTensor()
    sf_layer = nn.Sigmoid()

    for x in range(0, len(one_wsi_patches_names), BATCH_SIZE):
        print(f"in batch num {x // BATCH_SIZE} out of {len(one_wsi_patches_names) // BATCH_SIZE} ")
        tmp_tensor = torch.zeros([64, 3, *patch_size], device=device)
        for y in range(0, min(BATCH_SIZE, len(one_wsi_patches_names) - x - 1)):
            img = Image.open(os.path.join(TEMP_EXTRACTION_PATH_FOR_TEST_WSI, one_wsi_patches_names[x + y])).convert(
                'RGB')
            tmp_tensor[y] = to_tensor_transform(img)
            img.close()

        x_batch = tmp_tensor

        yhat = sf_layer(model(x_batch))
        y_temp = torch.clone(yhat)
        y_temp[y_temp >= 0.5] = 1  # is tumor
        y_temp[y_temp < 0.5] = 0  # is negative

        for y in range(0, min(BATCH_SIZE, len(one_wsi_patches_names) - x - 1)):
            # patient_000_node_0.tif_xy_95924_90156_512x512.png
            img_name = one_wsi_patches_names[x + y].split('_')
            i = int(img_name[5]) // 512
            j = int(img_name[6]) // 512
            predi = int(y_temp[y])
            pred_heatmap_tensor[i][j] = int(y_temp[y])
            if predi != 0:
                heatmap_tensor[predi - 1][i][j] = 1  # only red is metastasis, no other color should appear
            else:
                # WHITE: NEGATIVE
                heatmap_tensor[0][i][j] = 1
                heatmap_tensor[1][i][j] = 1
                heatmap_tensor[2][i][j] = 1

        del y_temp  # TODO: CHECK HIGH MEMORY USAGE
        del tmp_tensor

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
    print(f"negative_as_negative = {negative_as_negative}")
    print(f"itc_as_itc = {itc_as_itc}")
    print(f"macro_as_macro = {macro_as_macro}")
    print(f"miss classify negative_as_macro = {negative_as_macro}")
    print(f"miss classify negative_as_itc = {negative_as_itc}")
    print(f"miss classify itc_as_negative = {itc_as_negative}")
    print(f"miss classify itc_as_macro = {itc_as_macro}")
    print(f"miss classify macro_as_negative = {macro_as_negative}")
    print(f"miss classify macro_as_itc = {macro_as_itc}")
    heatmap_pil_img = heatmap(heatmap_tensor, show=False)
    heatmap_pil_img.save(rf"../heatmap_patient_{extr.patient_ID}_node_{extr.patient_node_ID}_img.png")
    torch.save(heatmap_tensor, rf"../heatmap_patient_{extr.patient_ID}_node_{extr.patient_node_ID}.pt")


def run_final_train_model():
    # TODO: rose save best model
    global model
    if model is None:
        init_final_model()
    validation_set = get_random_validation_idx()
    print("Validation set:")
    print(validation_set)
    camelyon17_train_ds, camelyon17_train_dl, camelyon17_validation_ds, camelyon17_validation_dl = Dataset.init_ds_final_solution(
        validation_WSI_IDs=validation_set,
        use_dummy_ds=False,
        only_train_set=False,
        negative_patches_ratio_train=0.7,
        negative_patches_ratio_validation=0.7,
    )

    # TODO temp
    sf_layer = nn.Sigmoid()
    # evaluator = Evaluator
    best_model_wts = None
    losses = []
    val_losses = []
    epoch_train_losses = []
    epoch_test_losses = []
    epoch_test_miss_classification_precentage = []
    tumor_accuracy_by_epoch = []
    negative_accuracy_by_epoch = []
    log_dir = "/home/hawahaitam/results/evaluation_metrics_log/"
    summary = TensorboardSummary(log_dir)
    writer = summary.create_summary()
    metrics_log_file = open("/home/hawahaitam/results/metrics_log.txt", 'w')
    metrics_log_file.close()
    logger_name = "metrics_app"
    logger1 = logging.getLogger(logger_name)
    logger1.propagate = False
    logger1.setLevel(logging.WARNING)
    formatter1 = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    timestr = time.strftime("%Y%m%d-%H%M")
    filehandler1 = logging.FileHandler(f"/home/hawahaitam/results/metrics_log_{timestr}.txt")
    filehandler1.setFormatter(formatter1)
    logger1.addHandler(filehandler1)

    def train_step(x, y):
        # make prediction
        yhat = model(x)
        yhat = sf_layer(yhat)
        # compute loss
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # optimizer.cleargrads()
        return loss

    for epoch in range(n_epochs):
        epoch_loss = 0

        # enter train mode
        model.train()
        print(f"Starting train Epoch {epoch + 1}")
        # total images we train at each epoch is n_batches_train*model_input_batch_size
        for i, data in tqdm(zip(range(n_batches_train), camelyon17_train_dl),
                            total=n_batches_train):  # iterate ove batches
            x_batch, y_batch = data
            x_batch = x_batch.to(device)  # move to gpu
            y_batch = y_batch.unsqueeze(1).float()  # convert target to same nn output shape
            y_batch = y_batch.to(device)  # move to gpu

            loss = train_step(x_batch, y_batch)
            # print("loss = " , loss)
            epoch_loss += loss / (n_batches_train * model_input_batch_size)
            losses.append(loss)

        epoch_train_losses.append(epoch_loss.item())
        print('\nEpoch : {}, train loss : {}'.format(epoch + 1, epoch_loss))

        # model to eval mode
        model.eval()
        # validation doesnt requires gradient
        with torch.no_grad():
            cum_loss = 0
            # TODO: rose use those after eval loop
            curr_n_negative_samples = 0
            curr_n_metastasis_samples = 0
            curr_n_correct_metastasis_classification = 0
            curr_n_correct_negative_classification = 0
            total_tn = 0
            total_tp = 0
            total_fn = 0
            total_fp = 0
            roc_curve = 0
            for i, data_ in tqdm(zip(range(n_batches_validation), camelyon17_validation_dl),
                                 total=n_batches_validation):
                x_batch, y_batch = data_
                x_batch = x_batch.to(device)
                y_batch = y_batch.unsqueeze(1).float()  # convert target to same nn output shape
                y_batch = y_batch.to(device)

                yhat = sf_layer(model(x_batch))
                roc_curve += AUC_of_the_ROC(yhat.cpu(), y_batch.cpu())

                y_temp = torch.clone(yhat)
                y_temp[y_temp >= 0.5] = 1  # is tumor
                y_temp[y_temp < 0.5] = 0  # is negative

                curr_batch_n_negative_samples = torch.sum((y_batch == 0)).item()

                curr_batch_n_metastasis_samples = torch.sum((y_batch == 1)).item()
                curr_batch_n_correct_metastasis_classification = torch.sum((y_temp == y_batch) * (y_temp == 1)).item()
                curr_batch_n_correct_negative_classification = torch.sum((y_temp == y_batch) * (y_temp == 0)).item()

                curr_n_negative_samples += curr_batch_n_negative_samples
                curr_n_metastasis_samples += curr_batch_n_metastasis_samples
                curr_n_correct_metastasis_classification += curr_batch_n_correct_metastasis_classification
                curr_n_correct_negative_classification += curr_batch_n_correct_negative_classification

                del y_temp
                val_loss = loss_fn(yhat, y_batch)
                cum_loss += val_loss / (n_batches_validation * model_input_batch_size)
                val_losses.append(val_loss.item())
            # total_tn = curr_batch_n_correct_negative_classification
            total_tn = curr_n_correct_negative_classification
            # total_tp = curr_batch_n_correct_metastasis_classification
            total_tp = curr_n_correct_metastasis_classification

            total_fn = curr_n_negative_samples - total_tn
            total_fp = curr_n_metastasis_samples - total_tp
            # acc = evaluator.accuracy(total_tp, total_tn, total_fp, total_fn)
            #
            # prec = evaltutor.precision(total_tp, total_fp)
            acc = accuracy(total_tp, total_tn, total_fp, total_fn)
            balanced_acc = balanced_accuracy(total_tp, total_tn, total_fp, total_fn)
            prec = precision(total_tp, total_fp)
            rec = recall(total_tp, total_fn)
            f1 = f1_score(total_tp, total_fp, total_fn)
            roc_curve = roc_curve / n_batches_validation
            total_miss_classified_patches = (curr_n_negative_samples - curr_n_correct_negative_classification) + (
                    curr_n_metastasis_samples - curr_n_correct_metastasis_classification)
            negative_accuracy_by_epoch.append(curr_n_correct_negative_classification / curr_n_negative_samples)
            tumor_accuracy_by_epoch.append(curr_n_correct_metastasis_classification / curr_n_metastasis_samples)
            # writer.add_scalar('Precision', prec, epoch)
            # writer.add_scalar('Acc', acc, epoch)

            logger1.info(
                'Epoch: %s , Acc: %s , Precision:%s , Recall: %s , F1_score: %s , Balanced_accuracy: %s , Area_under_curve_ROC: %s }',
                epoch, acc, prec, rec, f1, balanced_acc, roc_curve)

            epoch_test_losses.append(cum_loss.item())
            epoch_test_miss_classification_precentage.append(
                total_miss_classified_patches / (n_batches_validation * model_input_batch_size))
            print('Epoch : {}, validation loss : {}'.format(epoch + 1, cum_loss))
            print("---------------------------------------------------------------------------------------")
            print(
                f"Model miss classified {total_miss_classified_patches} out of {(n_batches_validation * model_input_batch_size)}")
            print(f"Model classification accuracy {1 - epoch_test_miss_classification_precentage[-1]} %")
            print(f"Tumor classification accuracy {tumor_accuracy_by_epoch[-1]} %")
            print(f"Negative classification accuracy {negative_accuracy_by_epoch[-1]} %")
            print("---------------------------------------------------------------------------------------")
            best_loss = min(epoch_test_losses)

            # save best model
            if cum_loss <= best_loss:
                best_model_wts = model.state_dict()
                timestr = time.strftime("%Y%m%d-%H%M%S")
                out_filepath = os.path.join(best_models_directory, curr_model_name, f"best_model_{timestr}.pt")
                torch.save(best_model_wts, out_filepath)
            # early stopping
            early_stopping_counter = 0
            if cum_loss > best_loss:
                early_stopping_counter += 1

            if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
                print("/nTerminating: early stopping")
                break  # terminate training
        # Disabled: change validation and train
        # new_validation_set = get_random_validation_idx()
        # camelyon17_train_ds.validation_WSI_IDs = new_validation_set
        # camelyon17_validation_ds.validation_WSI_IDs = new_validation_set

    # print final statistics
    print("epoch_test_losses", epoch_test_losses)
    print("epoch_train_losses", epoch_train_losses)
    print("epoch_test_miss_classification_precentage", epoch_test_miss_classification_precentage)
    model_accu = [1 - x for x in epoch_test_miss_classification_precentage]
    print("model accuracy list by epoch", model_accu)
    print("model negative accuracy list by epoch", negative_accuracy_by_epoch)
    print("model tumor accuracy list by epoch", tumor_accuracy_by_epoch)
    filehandler1.close()
