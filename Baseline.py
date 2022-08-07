from utils.Baseline_config import *
from Models import PT_Resnet50_KNN
from utils import Dataset
import torch


def get_random_samples_resnet50_forward_all_tags(nBatches : int = 1):
    if PT_Resnet50_KNN.pt_resnet50_model_cut is None:
        print("initinnngg")
        PT_Resnet50_KNN.init_pre_trained_resnet50_model()
    BATCH_SIZE = 64  # for now

    # print("camelyon17_ds")
    # print(camelyon17_ds.class_to_idx)

    result_tensor = torch.zeros(size=(nBatches*BATCH_SIZE, 2048))
    tags_tensor = torch.zeros(nBatches*BATCH_SIZE)
    with torch.no_grad():
        # class_idx_samples_counter = [0] * 4
        for i, Xy in zip(range(nBatches),Dataset.camelyon17_dl):
            print(f"extracted {i} samples out of {nBatches}")
            # data = next(iter(data_loader))
            X, y = Xy[0], Xy[1]  # X,y shape[0] == BATCH_SIZE
            result_tensor[i*BATCH_SIZE: (i+1) * BATCH_SIZE] = PT_Resnet50_KNN.pt_resnet50_model_cut.forward(X).squeeze()
            tags_tensor[i*BATCH_SIZE: (i+1) * BATCH_SIZE] = y
    return result_tensor, tags_tensor


"""
This function receive indices of validation patch
it trains KNN after resnet50(-1) output and classify all WSI
input:
1) array of tuples, (patient_ID,node_ID)
TODO: implement WSIs_idx 
"""
def projectA_run_baseline_heatmap_build(WSIs_idx):
    global knn_model
    assert len(WSIs_idx) > 0
    print("initing ds,dl")
    Dataset.init_ds_dl()
    print("initing ds,dl")
    train_data_tensor, train_tags_tensor = get_random_samples_resnet50_forward_all_tags(50)
    test_data_tensor, test_tags_tensor = get_random_samples_resnet50_forward_all_tags(50)
    print("initing KNN")
    PT_Resnet50_KNN.init_Knn_model(dataset=[train_data_tensor,train_tags_tensor],n_neighbors= 7)
    PT_Resnet50_KNN.knn_validate(test_data_tensor,test_tags_tensor,use_print=True)