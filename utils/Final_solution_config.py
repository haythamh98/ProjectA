import os
import torch
from utils.Dataset_config import BATCH_SIZE
from torch.optim import Adam

n_epochs = 100  # 20000 step is n_batches_train=200 and 100 epochs
early_stopping_tolerance = n_epochs  # //10
early_stopping_threshold = -1  # TODO: = 0.03
################## Adam
adam_lr = 0.001
adam_betas = (0.9, 0.999)
adam_eps = 1e-08
adam_weight_decay = 0
adam_amsgrad = False

best_models_directory = os.path.join(f"/", "home", "hawahaitam", "results", "best_models")
curr_model_name = "resnet18"

model_input_batch_size = BATCH_SIZE  # Don't touch
n_batches_train = 200  # *64
n_batches_validation = 100  # *64
# validate_every_n_virtual_epochs = 200 # TODO: for debug put this = 1

loss_fn = torch.nn.BCELoss()

TEMP_EXTRACTION_PATH_FOR_TEST_WSI = os.path.join('/', 'home', 'hawahaitam', 'temp_dir')
