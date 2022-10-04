import random

import torch
import torch.nn as nn
from utils import Dataset
from torchvision import models
from utils.Dataset_config import *
from utils.Final_solution_config import *
from tqdm import tqdm


model = None

def init_final_model():
    global model
    model = models.resnet18(pretrained=False)
    model.eval()
    print(list(model.children())[:])
    # model = torch.nn.Sequential(*(list(model.children())[:-1]))  # strips off last linear layer
    for params in model.parameters():
        params.requires_grad_ = False
    # add a new final layer
    nr_filters = model.fc.in_features  # number of input features of last layer
    model.fc = nn.Linear(nr_filters, 1)
    print("RESTENT18!*" , list(model.children())[:])
    model = model.to(device)

def get_random_validation_idx():
    from WSI_Tools.PatchExtractor_Tools.PatchExtractor_config import interesting_wsis
    size_ = len(interesting_wsis)
    return random.choices(interesting_wsis, weights=None, cum_weights=None, k=int(0.20*size_)+1)


def run_final_train_model():
    global model
    if model is None:
        init_final_model()
    camelyon17_train_ds, camelyon17_train_dl, camelyon17_validation_ds, camelyon17_validation_dl = Dataset.init_ds_final_solution(
        validation_WSI_IDs=get_random_validation_idx(),
        use_dummy_ds=True,
        only_train_set=True,
        negative_patches_ratio_train=0.7,
        negative_patches_ratio_validation=0.7,
    )

    losses = []
    val_losses = []
    epoch_train_losses = []
    epoch_test_losses = []

    def train_step(x, y):
        # make prediction
        yhat = model(x)
        # enter train mode
        model.train()
        # compute loss
        loss = loss_fn(yhat, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # optimizer.cleargrads()

        return loss
    for epoch in range(n_epochs):
        epoch_loss = 0
        # total images we train at each epoch is n_batches_train*model_input_batch_size
        for i, data in tqdm(zip(range(n_batches_train),camelyon17_train_dl), total=n_batches_train):  # iterate ove batches
            x_batch, y_batch = data
            x_batch = x_batch.to(device)  # move to gpu
            y_batch = y_batch.unsqueeze(1).float()  # convert target to same nn output shape
            y_batch = y_batch.to(device)  # move to gpu

            loss = train_step(x_batch, y_batch)
            epoch_loss += loss / len(n_batches_train*model_input_batch_size)
            losses.append(loss)

        epoch_train_losses.append(epoch_loss)
        print('\nEpoch : {}, train loss : {}'.format(epoch + 1, epoch_loss))

        # validation doesnt requires gradient
        with torch.no_grad():
            cum_loss = 0
            for i , x_batch, y_batch in zip(range(n_batches_validation),camelyon17_validation_dl):
                print(f"validating batch num {i+1} out of {n_batches_validation} ")
                x_batch = x_batch.to(device)
                y_batch = y_batch.unsqueeze(1).float()  # convert target to same nn output shape
                y_batch = y_batch.to(device)

                # model to eval mode
                model.eval()

                yhat = model(x_batch)
                val_loss = loss_fn(yhat, y_batch)
                cum_loss += loss / len(n_batches_validation*model_input_batch_size)
                val_losses.append(val_loss.item())

            epoch_test_losses.append(cum_loss)
            print('Epoch : {}, val loss : {}'.format(epoch + 1, cum_loss))

            best_loss = min(epoch_test_losses)

            # save best model
            if cum_loss <= best_loss:
                best_model_wts = model.state_dict()

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