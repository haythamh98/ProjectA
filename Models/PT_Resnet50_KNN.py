import torch
import torchvision.datasets
from sklearn.neighbors import KNeighborsClassifier
from torchvision import models
from utils.Dataset_config import *
from utils.Dataset import *

knn_model = None
pt_resnet50_model_cut = None


# TODO: must get output of resnet50 (layer -2) and tag
# TODO: i didnt debug here
def init_Knn_model(
        dataset,  # iterable of tuples (X,y)
        n_neighbors: int
):
    global knn_model
    if knn_model is None:
        knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(dataset[:, 0], dataset[:, 1])
    '''
    # Predict on dataset which model has not seen before
    y_pred = knn.predict(X_test)
    miss_classifications = y_test - y_pred
    print(f"miss classified {torch.count_nonzero(miss_classifications)} out of {len(y_test)}")
    accu = 1 - torch.count_nonzero(miss_classifications) / len(y_test)
    print(f"Accuracy {accu} ")
    '''


def init_pre_trained_resnet50_model():
    global pt_resnet50_model_cut
    resnet50 = models.resnet50(pretrained=True)
    resnet50.eval()
    pt_resnet50_model_cut = torch.nn.Sequential(*(list(resnet50.children())[:-1]))  # strips off last linear layer


def get_random_samples_resnet50_forward(nBatches : int = 1):
    global pt_resnet50_model_cut
    if pt_resnet50_model_cut is None:
        init_pre_trained_resnet50_model()

    result_tensor = torch.zeros(size=(nBatches*BATCH_SIZE, 2048))
    tags_tensor = torch.zeros(nBatches*BATCH_SIZE)
    with torch.no_grad():
        for i, Xy in zip(range(nBatches),camelyon17_dl):
            X, y = Xy[0], Xy[1]  # X,y shape[0] == BATCH_SIZE
            result_tensor[i*BATCH_SIZE: (i+1) * BATCH_SIZE] = pt_resnet50_model_cut.forward(X).squeeze()
            tags_tensor[i*BATCH_SIZE: (i+1) * BATCH_SIZE] = y
    return result_tensor, tags_tensor
