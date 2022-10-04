import torch
from utils.Dataset_config import BATCH_SIZE
from torch.optim import Adam

n_epochs = 10
early_stopping_tolerance = 3
early_stopping_threshold = 0.03

model_input_batch_size = BATCH_SIZE
n_batches_train = 16
n_batches_validation = 4

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = Adam(lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)