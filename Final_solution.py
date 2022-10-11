import random
from collections import OrderedDict
import torch
import torch.nn as nn
from utils import Dataset
from torchvision import models
from utils.Dataset_config import *
from utils.Final_solution_config import *
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix
from utils.Evaluation_metrics import *
from utils.summaries import TensorboardSummary
import logging




model = None
optimizer = None

def init_final_model():
    global model,optimizer
    model = models.resnet18(pretrained=True)
    model.eval()
    # print(list(model.children())[:])
    # model = torch.nn.Sequential(*(list(model.children())[:-1]))  # strips off last linear layer
    for params in model.parameters():
        params.requires_grad_ = False
    # add a new final layer
    nr_filters = model.fc.in_features  # number of input features of last layer
    model.fc = nn.Linear(nr_filters ,out_features=1)
    # print("RESTENT18!*" , list(model.children())[:])
    model = model.to(device)
    optimizer = Adam(model.parameters(),lr=adam_lr, betas=adam_betas, eps=adam_eps, weight_decay=adam_weight_decay, amsgrad=adam_amsgrad)




def get_random_validation_idx():
    from WSI_Tools.PatchExtractor_Tools.PatchExtractor_config import interesting_wsis
    size_ = len(interesting_wsis)
    return random.choices(interesting_wsis, weights=None, cum_weights=None, k=int(0.20*size_)+1)


def run_final_train_model():
    # TODO: rose save best model
    global model
    if model is None:
        init_final_model()
    camelyon17_train_ds, camelyon17_train_dl, camelyon17_validation_ds, camelyon17_validation_dl = Dataset.init_ds_final_solution(
        validation_WSI_IDs=get_random_validation_idx(),
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
    filehandler1 = logging.FileHandler("/home/hawahaitam/results/metrics_log.txt")
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
        print(f"Starting train Epoch {epoch+1}")
        # total images we train at each epoch is n_batches_train*model_input_batch_size
        for i, data in tqdm(zip(range(n_batches_train),camelyon17_train_dl), total=n_batches_train):  # iterate ove batches
            x_batch, y_batch = data
            x_batch = x_batch.to(device)  # move to gpu
            y_batch = y_batch.unsqueeze(1).float()  # convert target to same nn output shape
            y_batch = y_batch.to(device)  # move to gpu

            loss = train_step(x_batch, y_batch)
            # print("loss = " , loss)
            epoch_loss += loss / (n_batches_train*model_input_batch_size)
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
            for i , data_ in tqdm(zip(range(n_batches_validation), camelyon17_validation_dl), total=n_batches_validation):
                x_batch, y_batch = data_
                x_batch = x_batch.to(device)
                y_batch = y_batch.unsqueeze(1).float()  # convert target to same nn output shape
                y_batch = y_batch.to(device)

                yhat = sf_layer(model(x_batch))
                roc_curve += AUC_of_the_ROC(yhat.cpu(), y_batch.cpu())


                y_temp = torch.clone(yhat)
                y_temp[y_temp >= 0.5] = 1 # is tumor
                y_temp[y_temp < 0.5] = 0 # is negative

                curr_batch_n_negative_samples = torch.sum((y_batch==0)).item()

                curr_batch_n_metastasis_samples = torch.sum((y_batch==1)).item()
                curr_batch_n_correct_metastasis_classification =  torch.sum((y_temp==y_batch) * (y_temp==1)).item()
                curr_batch_n_correct_negative_classification =  torch.sum((y_temp==y_batch) * (y_temp==0)).item()

                curr_n_negative_samples += curr_batch_n_negative_samples
                curr_n_metastasis_samples += curr_batch_n_metastasis_samples
                curr_n_correct_metastasis_classification += curr_batch_n_correct_metastasis_classification
                curr_n_correct_negative_classification += curr_batch_n_correct_negative_classification

                del y_temp
                val_loss = loss_fn(yhat, y_batch)
                cum_loss += val_loss / (n_batches_validation*model_input_batch_size)
                val_losses.append(val_loss.item())
            total_tn = curr_batch_n_correct_negative_classification
            total_tp = curr_batch_n_correct_metastasis_classification
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
            roc_curve = roc_curve/n_batches_validation
            total_miss_classified_patches = (curr_n_negative_samples - curr_n_correct_negative_classification) + (curr_n_metastasis_samples-curr_n_correct_metastasis_classification)
            negative_accuracy_by_epoch.append(curr_n_correct_negative_classification/curr_n_negative_samples)
            tumor_accuracy_by_epoch.append(curr_n_correct_metastasis_classification/curr_n_metastasis_samples)
            # writer.add_scalar('Precision', prec, epoch)
            # writer.add_scalar('Acc', acc, epoch)

            logger1.warning('Epoch: %s , Acc: %s , Precision:%s , Recall: %s , F1_score: %s , Balanced_accuracy: %s , Area_under_curve_ROC: %s }' ,
                            epoch, acc, prec, rec, f1, balanced_acc, roc_curve)

            epoch_test_losses.append(cum_loss.item())
            epoch_test_miss_classification_precentage.append(total_miss_classified_patches/(n_batches_validation*model_input_batch_size))
            print('Epoch : {}, validation loss : {}'.format(epoch + 1, cum_loss))
            print("---------------------------------------------------------------------------------------")
            print(f"Model miss classified {total_miss_classified_patches} out of {(n_batches_validation*model_input_batch_size)}")
            print(f"Model classification accuracy {1-epoch_test_miss_classification_precentage[-1]} %")
            print(f"Tumor classification accuracy {tumor_accuracy_by_epoch[-1]} %")
            print(f"Negative classification accuracy {negative_accuracy_by_epoch[-1]} %")
            print("---------------------------------------------------------------------------------------")
            best_loss = min(epoch_test_losses)

            # save best model
            if cum_loss <= best_loss:
                best_model_wts = model.state_dict()
                timestr = time.strftime("%Y%m%d-%H%M%S")
                out_filepath = os.path.join(best_models_directory,curr_model_name,f"best_model_{timestr}.pt")
                torch.save(best_model_wts, out_filepath)
            # early stopping
            early_stopping_counter = 0
            if cum_loss > best_loss:
                early_stopping_counter += 1

            if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
                print("/nTerminating: early stopping")
                break  # terminate training
        # change validation and train
        new_validation_set = get_random_validation_idx()
        camelyon17_train_ds.validation_WSI_IDs = new_validation_set
        camelyon17_validation_ds.validation_WSI_IDs = new_validation_set





    # print final statistics
    print("epoch_test_losses",epoch_test_losses)
    print("epoch_train_losses",epoch_train_losses)
    print("epoch_test_miss_classification_precentage",epoch_test_miss_classification_precentage)
    model_accu = [1-x for x in epoch_test_miss_classification_precentage]
    print("model accuracy list by epoch",model_accu)
    print("model negative accuracy list by epoch",negative_accuracy_by_epoch)
    print("model tumor accuracy list by epoch",tumor_accuracy_by_epoch)
    filehandler1.close()
