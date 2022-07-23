import torch
import torchvision.datasets
from sklearn.neighbors import KNeighborsClassifier
from torchvision import models

knn_model = None
pt_resnet50_model_cut = None


def init_Knn_model(
        dataset: torchvision.datasets.VisionDataset,  # or any subclass
        n_neighbors: int
):
    global knn_model
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    return
    # Predict on dataset which model has not seen before
    y_pred = knn.predict(X_test)
    miss_classifications = y_test - y_pred
    print(f"miss classified {torch.count_nonzero(miss_classifications)} out of {len(y_test)}")
    accu = 1 - torch.count_nonzero(miss_classifications) / len(y_test)
    print(f"Accuracy {accu} ")


def init_pre_trained_resnet50_model():
    global pt_resnet50_model_cut
    resnet50 = models.resnet50(pretrained=True)
    resnet50.eval()
    pt_resnet50_model_cut = torch.nn.Sequential(*(list(resnet50.children())[:-1]))  # strips off last linear layer


def resnet50_forward(batch_tensor):
    with torch.no_grad():
        forward = pt_resnet50_model_cut.forward(batch_tensor)
        return forward
