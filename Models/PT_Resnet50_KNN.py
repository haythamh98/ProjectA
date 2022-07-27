import torch
import torchvision.datasets
from sklearn.neighbors import KNeighborsClassifier
from torchvision import models
from utils.Dataset_config import *
from utils.Dataset import *

knn_model = None
pt_resnet50_model_cut = None



'''
rose
1) Dataset usage example and forward into resnet50 model in "get_random_samples_resnet50_forward(nBatches : int = 1)"
    a) the example above also includes combining the results of different forward iterations
2) for knn module you might need to read documentation, can be found in sklearn documentation
    
'''




def init_Knn_model(
        train_dataset,  # iterable, dim0 = X, dim1 = y
        n_neighbors: int
):
    global knn_model
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(train_dataset[0], train_dataset[1])

def knn_predict(X):
    global knn_model
    assert knn_model is not None  # must init first
    return knn_model.predict(X)

def knn_validate(X_test,y_test, use_print=False):
    global knn_model
    assert knn_model is not None  # must init first
    y_pred = knn_model.predict(X_test)
    miss_classifications = y_test - y_pred
    accu = 1 - torch.count_nonzero(miss_classifications) / len(y_test)
    if use_print:
        print(f"knn miss classified {torch.count_nonzero(miss_classifications)} out of {len(y_test)}")
        print(f"Accuracy {accu} ")
    return accu

def knn_sanity_check():
    forw,y = get_random_samples_resnet50_forward(2)
    to_knn_train_fit = [forw[32:],y[32:]] # must stack , or change innnir implementation of init_Knn_model
    X_test = forw[:32]
    y_test = y[:32]

    init_Knn_model(dataset=to_knn_train_fit,n_neighbors=1)
    print("validate on same trainset, with k=0 should have 100% accuracy")
    knn_validate(to_knn_train_fit[0],to_knn_train_fit[1],use_print=True)
    print("now test set ... ")
    knn_validate(X_test,y_test,use_print=True)

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
