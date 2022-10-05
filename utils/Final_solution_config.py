import torch
from utils.Dataset_config import BATCH_SIZE
from torch.optim import Adam

n_epochs = 50
early_stopping_tolerance = 5
early_stopping_threshold = -1 # TODO: = 0.03
################## Adam
adam_lr = 0.0001
adam_betas = (0.9, 0.999)
adam_eps = 1e-08
adam_weight_decay = 0
adam_amsgrad = False






model_input_batch_size = BATCH_SIZE
n_batches_train = 2
n_batches_validation = 1

loss_fn = torch.nn.BCELoss()
