from WSI_Tools.PatchExtractor_Tools.PatchExtractor_config import *

def n_to_3digit_str(n):
    return '0' * (3-len(str(n))) + str(n)

def form_wsi_path_by_ID(WSI_ID):
    camelyon17_train_dir = os.path.join(NETWORK_DATABASE,'Camelyon17','training')
    center_ID = -1
    if WSI_ID[0] in range(0,20):
        center_ID = 0
    elif WSI_ID[0] in range(20,40):
        center_ID = 1
    elif WSI_ID[0] in range(40,60):
        center_ID = 2
    elif WSI_ID[0] in range(60,80):
        center_ID = 3
    elif WSI_ID[0] in range(80,100):
        center_ID = 4
    # patient_005_node_0.tif
    return os.path.join(camelyon17_train_dir,f'center_{center_ID}',f'patient_{n_to_3digit_str(WSI_ID[0])}',f'patient_{n_to_3digit_str(WSI_ID[0])}_node_{WSI_ID[1]}.tif')
