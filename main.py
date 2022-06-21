import os, sys
from ctypes import cdll
from os import path

import PIL
import cv2
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from openslide.deepzoom import DeepZoomGenerator
import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import ColorJitter

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from ctypes.util import find_library

import openslide
# from CLAM.wsi_core.WholeSlideImage import WholeSlideImage


import xml.dom.minidom
from xml.dom import minidom

from WSI_Tools.AnnotationHandler import AnnotationHandler
from WSI_Tools.PatchExtractor import PatchExtractor, ExtractType

from sklearn.manifold import TSNE


# from sklearn.manifold import MDS


def big_file_Walkaround():
    PIL.Image.MAX_IMAGE_PIXELS = 933120000  # 21630025728 * 10


def extract_tiles_in_rectangle(wsi, level: int, tile_size: tuple, outsize: tuple, start_rec: tuple, end_rec: tuple,
                               out_path,
                               out_name):
    h, w = wsi.level_dimensions[level - 1][0], wsi.level_dimensions[level - 1][1]

    tile_decrease_ratio_x = 1  # tile_size[0] / outsize[0]
    tile_decrease_ratio_y = 1  # tile_size[1] / outsize[1]

    start_x, start_y = start_rec
    end_x, end_y = end_rec
    assert end_x <= h and end_y <= w  # stam

    pic_index = 0
    for i in range(start_x // tile_size[0], end_x // tile_size[0]):
        for j in range(start_y // tile_size[1], end_y // tile_size[1]):
            x, y = i * tile_size[0], j * tile_size[1]
            sample = wsi.read_region(location=(x, y), level=level,
                                     size=(
                                         tile_size[0] // tile_decrease_ratio_x, tile_size[1] // tile_decrease_ratio_y))
            sample.save(path.join(out_path, f'{out_name}_{pic_index}'), 'PNG')
            print(pic_index)
            pic_index += 1
    return pic_index


def extract_samples():
    file_path = os.path.join('/', 'home', 'administrator', 'temp', 'tumor_009.tif')
    out_file_path_w_tumor = os.path.join('/', 'home', 'administrator', 'temp', 'tumor_9_with_tumor')
    out_file_path_wo_tumor = os.path.join('/', 'home', 'administrator', 'temp', 'tumor_9_without_tumor')
    out_file_path_temp = os.path.join('/', 'home', 'administrator', 'temp', 'temp')
    '''
    if os.path.isdir(out_file_path_w_tumor):
        os.rmdir(out_file_path_w_tumor)
    if os.path.isdir(out_file_path_wo_tumor):
        os.rmdir(out_file_path_wo_tumor)
    os.mkdir(out_file_path_wo_tumor, mode=777)
    os.mkdir(out_file_path_w_tumor, mode=777)
    '''
    wsi = openslide.open_slide(file_path)
    # show_image_RGBA(wsi.get_thumbnail(size=(512,512)))

    tile_size = 512
    tile_decrease_ratio = 1
    extract_tiles_in_rectangle(wsi, level=1, tile_size=(512, 512), outsize=(512, 512), start_rec=(0, 0),
                               end_rec=(600, 600), out_path=out_file_path_temp, out_name="hi")


def extract_features():
    resnet50 = models.resnet50(pretrained=True)
    resnet50.eval()
    my_model = torch.nn.Sequential(*(list(resnet50.children())[:-1]))  # strips off last linear layer

    preprocess = transforms.Compose([transforms.ToTensor(), ])
    ssize = 99
    outputs = torch.zeros(size=(2 * ssize, 2048))
    file_dir = os.path.join('dataset', 'normal')
    file_dir1 = os.path.join('dataset', 'tumor')
    for i, img in enumerate(os.listdir(file_dir)):
        if i == outputs.shape[0] / 2:
            break
        image = Image.open(os.path.join(file_dir, img))  # .convert('RGB')
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = torch.squeeze(my_model(input_batch)).unsqueeze(0)
            outputs[i] = output.clone()
            print(i)
    for i, img in enumerate(os.listdir(file_dir1)):
        if i == outputs.shape[0] / 2:
            break
        image = Image.open(os.path.join(file_dir1, img))  # .convert('RGB')
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        # print(input_batch.shape)
        with torch.no_grad():
            output = torch.squeeze(my_model(input_batch)).unsqueeze(0)
            outputs[ssize + i] = output.clone()
            print(i)

    for i in range(10):
        X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(outputs)

        print(X_embedded)
        plt.scatter(X_embedded[:ssize, 0], X_embedded[:ssize, 1], color='#88c999')
        plt.scatter(X_embedded[ssize:, 0], X_embedded[ssize:, 1], color='hotpink')

        plt.legend(["Normal", "Tumor"])
        plt.show()


def extract_forward_tensors(wsi_file_path, xml_file_path, output_path_dir, tumor: bool):
    '''
    >>> t = torch.tensor([1., 2.])
    >>> torch.save(t, 'tensor.pt')
    >>> torch.load('tensor.pt')
    tensor([1., 2.])
    '''
    resnet50 = models.resnet50(pretrained=True)
    resnet50.eval()
    my_model = torch.nn.Sequential(*(list(resnet50.children())[:-1]))  # strips off last linear layer

    extractor = PatchExtractor(wsi_path=wsi_file_path, xml_path=xml_file_path, patches_in_batch=64, size=(512, 512),
                               tumor=tumor, overlap=0, wsi_level=0)
    for i, (batch, samples_count) in enumerate(extractor):
        transform = transforms.ToPILImage()

        # img = transform(batch[0]).convert("RGB")
        # img.show()
        with torch.no_grad():
            forward = my_model(batch)
            type_name = 'with_tumor' if tumor else 'without_tumor'
            out_tensor_path = os.path.join(output_path_dir,
                                           f'{extractor.WSI_type}_{extractor.WSI_ID}_{type_name}_{i}.pt')
            if samples_count != 64:
                forward = forward[:samples_count]
            torch.save(forward, out_tensor_path)
            print(f"file {out_tensor_path} was done with {samples_count}")


def apply_TSNE(tensor_dir):
    with_tumor = torch.randn(0)  # torch.zeros(size=(64,2048))
    wo_tumor = torch.randn(0)  # torch.zeros(size=(64,2048))
    for tensor_file_name in os.listdir(tensor_dir):
        loaded_tensor = torch.load(os.path.join(tensor_dir, tensor_file_name))
        loaded_tensor = torch.squeeze(loaded_tensor)
        data_has_tumor = ('with_tumor' in tensor_file_name)
        print(f"loading file {tensor_file_name} laoded shape {loaded_tensor.shape}")
        if data_has_tumor:
            if with_tumor is None:
                with_tumor = torch.clone(loaded_tensor)
            else:
                with_tumor = torch.cat((with_tumor, loaded_tensor), 0)
        else:
            if wo_tumor is None:
                wo_tumor = torch.clone(loaded_tensor)
            else:
                wo_tumor = torch.cat((wo_tumor, loaded_tensor), 0)
    total_tensor = None
    if wo_tumor is None:
        total_tensor = with_tumor
    elif with_tumor is None:
        total_tensor = wo_tumor
    else:
        total_tensor = torch.cat((wo_tumor, with_tumor), 0)
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(total_tensor)
    tumor_count = with_tumor.shape[0]
    no_tumor_count = wo_tumor.shape[0]
    if no_tumor_count > 0:
        plt.scatter(X_embedded[:no_tumor_count, 0], X_embedded[:no_tumor_count, 1], color='#88c999')
    if tumor_count > 0:
        plt.scatter(X_embedded[no_tumor_count:, 0], X_embedded[no_tumor_count:, 1], color='hotpink')
    plt.legend(["Normal", "Tumor"])
    plt.show()


def int_to_3digit_str(n):
    return "0" * (3 - len(str(n))) + str(n)


def tensor_file_resnet50_forward(path, num_samples=64 * 2):
    resnet50 = models.resnet50(pretrained=True)
    resnet50.eval()
    transforms = torch.nn.Sequential(  # TODO
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
    )
    my_model = torch.nn.Sequential(*(list(resnet50.children())[:-1]))  # strips off last linear layer
    samples_out_count = 0
    results = []
    for filename in os.listdir(path):
        batch_tensor = torch.load(os.path.join(path, filename))
        with torch.no_grad():
            transformed_tensor = transforms(batch_tensor)
            forward = my_model(transformed_tensor)
            results += [forward]
            samples_out_count += batch_tensor.shape[0]
            if samples_out_count >= num_samples:
                break
    return results


def extract_batches(wsi_path, out_dir):
    xml_file_path = wsi_path[:-4] + '.xml'
    extractor = PatchExtractor(wsi_path=wsi_path, xml_path=xml_file_path, patches_in_batch=64,
                               size=(512, 512), extract_type=ExtractType.normal_only, overlap=0, wsi_level=0)
    current_out_dir = os.path.join(out_dir, extractor.WSI_type + '_' + int_to_3digit_str(extractor.WSI_ID) + '_normal')
    if not os.path.isdir(current_out_dir):
        os.mkdir(current_out_dir)
    extractor.extract_path_batches_to_tensors(current_out_dir)
    if 'tumor' in extractor.WSI_type:
        extractor.resetITR()
        extractor.extract_type = ExtractType.tumor_only
        current_out_dir = os.path.join(out_dir,
                                       extractor.WSI_type + '_' + int_to_3digit_str(extractor.WSI_ID) + '_tumor')
        if not os.path.isdir(current_out_dir):
            os.mkdir(current_out_dir)
        extractor.extract_path_batches_to_tensors(current_out_dir)

    '''
    file_path = os.path.join('/', 'databases', 'hawahaitam', 'tumor_005.tif')
    extract_batches(file_path, out_dir_path)
    file_path = os.path.join('/', 'databases', 'hawahaitam', 'tumor_006.tif')
    extract_batches(file_path, out_dir_path)
    file_path = os.path.join('/', 'databases', 'hawahaitam', 'tumor_007.tif')
    extract_batches(file_path, out_dir_path)
    file_path = os.path.join('/', 'databases', 'hawahaitam', 'tumor_001.tif')
    extract_batches(file_path, out_dir_path)
    file_path = os.path.join('/', 'databases', 'hawahaitam', 'normal_001.tif')
    extract_batches(file_path, out_dir_path)
    file_path = os.path.join('/', 'databases', 'hawahaitam', 'normal_002.tif')
    extract_batches(file_path, out_dir_path)
    file_path = os.path.join('/', 'databases', 'hawahaitam', 'normal_003.tif')
    extract_batches(file_path, out_dir_path)
    file_path = os.path.join('/', 'databases', 'hawahaitam', 'normal_004.tif')
    extract_batches(file_path, out_dir_path)
    file_path = os.path.join('/', 'databases', 'hawahaitam', 'normal_005.tif')
    extract_batches(file_path, out_dir_path)
    
    '''


def dir_to_forward_tensor(dir_path, nSamples):
    res_list = tensor_file_resnet50_forward(dir_path, num_samples=nSamples)
    res_tensor = torch.cat(res_list, dim=0)
    torch.save(res_tensor,
               os.path.join('/', 'databases', 'hawahaitam', os.path.split(dir_path)[1] + '_forwarded_tensor.pt'))


def from_forward_dir_to_tsne_tensor(forward_dir, output_path):
    samples_to_take_from_each_tumor_forward_tensor = 64 * 100
    samples_to_take_from_each_normal_forward_tensor = 64 * 20

    normal_total_tensor = None
    tumor_total_tensor = None
    for forward_tensor_file_name in os.listdir(forward_dir):
        val = torch.load(os.path.join(forward_dir, forward_tensor_file_name))
        val = torch.squeeze(val)
        val = val[torch.randperm(val.size()[0])]

        if '_normal' in forward_tensor_file_name:
            val = val[:min(samples_to_take_from_each_normal_forward_tensor, val.shape[0])]
            print(val.shape)
            if normal_total_tensor is None:
                normal_total_tensor = torch.clone(val)
            else:
                normal_total_tensor = torch.cat((normal_total_tensor, val), dim=0)

        if '_tumor' in forward_tensor_file_name:
            val = val[:min(samples_to_take_from_each_tumor_forward_tensor, val.shape[0])]
            print(val.shape)
            if tumor_total_tensor is None:
                tumor_total_tensor = torch.clone(val)
            else:
                tumor_total_tensor = torch.cat((tumor_total_tensor, val), dim=0)

    total_tensor = torch.cat((normal_total_tensor, tumor_total_tensor))
    label_tensor = torch.cat((torch.zeros(normal_total_tensor.shape[0]), torch.ones(tumor_total_tensor.shape[0])))  # normal: 0, tumor: 1
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(total_tensor)

    labels_path = os.path.join(os.path.split(output_path)[0], os.path.split(output_path)[1][:-3] + '_labels.pt')
    print("labels_path", labels_path)
    torch.save(X_embedded, output_path)
    torch.save(label_tensor, labels_path)

    tumor_count = tumor_total_tensor.shape[0]  # total_tensor.shape[0]
    normal_count = normal_total_tensor.shape[0]

    if normal_count > 0:
        plt.scatter(X_embedded[:normal_count, 0], X_embedded[:normal_count, 1], color='#88c999')
        # plt.scatter(X_embedded[:normal_count - normal_test_count, 0], X_embedded[:normal_count - normal_test_count, 1],
        #            color='#88c999')
        # plt.scatter(X_embedded[normal_count - normal_test_count:normal_count, 0],
        #            X_embedded[normal_count - normal_test_count:normal_count, 1], color='#000000')
    if tumor_count > 0:
        plt.scatter(X_embedded[normal_count:, 0], X_embedded[normal_count:, 1], color='hotpink')
        # plt.scatter(X_embedded[normal_count:-tumor_test_count, 0], X_embedded[normal_count:-tumor_test_count, 1],
        #            color='hotpink')
        # plt.scatter(X_embedded[-tumor_test_count:, 0], X_embedded[-tumor_test_count:, 1], color='#0000ff')
    plt.legend(["Normal", "Tumor"])
    # plt.legend(["Normal", "Normal predicted", "Tumor", "Tumor predicted"])
    plt.show()
    return output_path, labels_path


def from_tnse_tensor_to_knn_module(knn_input_tensor_path, knn_labels_tensor_path, n_neighbors=7):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split

    knn_input_tensor = torch.load(knn_input_tensor_path)
    knn_labels_tensor = torch.load(knn_labels_tensor_path)

    #print(knn_labels_tensor)
    # Create feature and target arrays
    shuffle_idx = torch.randperm(knn_input_tensor.shape[0])
    #print(shuffle_idx)

    X = knn_input_tensor[shuffle_idx]
    y = knn_labels_tensor[shuffle_idx]

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    knn.fit(X_train, y_train)

    # Predict on dataset which model has not seen before
    y_pred = knn.predict(X_test)
    miss_classifications = y_test - y_pred
    print(f"miss classified {torch.count_nonzero(miss_classifications)} out of {len(y_test)}")
    accu = 1 - torch.count_nonzero(miss_classifications) / len(y_test)
    print(f"Accuracy {accu} ")
    return knn


if __name__ == '__main__':
    print("Hi")
    # big_file_Walkaround()
    # for windows only
    # os.add_dll_directory(r"C:\Users\haytham\Downloads\openslide-win64-20171122\openslide-win64-20171122\bin")

    # knn + tsne
    # tsne get output
    forward_dir = os.path.join('/', 'databases', 'hawahaitam', 'forwarded')
    tsne_output_path = os.path.join('/', 'databases', 'hawahaitam', 'tsne', 'a_t6_n1_out.pt')
    output_path, labels_path = from_forward_dir_to_tsne_tensor(forward_dir, tsne_output_path)
    # output_path = r"/databases/hawahaitam/tsne/test_1.pt"
    # labels_path = r"/databases/hawahaitam/tsne/test_1_labels.pt"
    # iterate over K
    for k in [1, 5, 10, 15, 20, 25, 30,50,100,150,300,500,800]:
        print(f"Knn with k={k}")
        knn = from_tnse_tensor_to_knn_module(knn_input_tensor_path=output_path, knn_labels_tensor_path=labels_path)
        # try to predict all WSI
        continue
        forward_test_wsi_normal001 = os.path.join('/', 'databases', 'hawahaitam', 'forwarded_test','normal_001_normal_forwarded_tensor.pt')
        forward_test_wsi_tumor006_t = os.path.join('/', 'databases', 'hawahaitam', 'forwarded_test','tumor_006_tumor_forwarded_tensor.pt')
        forward_test_wsi_tumor006_n = os.path.join('/', 'databases', 'hawahaitam', 'forwarded_test','tumor_006_normal_forwarded_tensor.pt')

        test_wsi = torch.load(forward_test_wsi_normal001)
        y_pred = knn.predict(test_wsi)
        print(f"Accuracy for normal001 {np.count_nonzero(y_pred)/len(y_pred)}")

        test_wsi = torch.load(forward_test_wsi_tumor006_n)
        y_pred = knn.predict(test_wsi)
        print(f"Accuracy for tumor006 normal {np.count_nonzero(y_pred)/len(y_pred)}")

        test_wsi = torch.load(forward_test_wsi_tumor006_t)
        y_pred = knn.predict(test_wsi)
        print(f"Accuracy for tumor006 tumor {1 - np.count_nonzero(y_pred)/len(y_pred)}")





























    exit()
    dir_path = os.path.join('/', 'data', 'hawahaitam')
    for tensor_dir_name in os.listdir(dir_path):
        print(f"start {tensor_dir_name}")
        full_dir_name = os.path.join(dir_path, tensor_dir_name)
        dir_to_forward_tensor(full_dir_name, 64 * 1000)

    exit()
    tensor_batches_dir = os.path.join('/', 'data', 'hawahaitam')
    dataset_transforms = torch.nn.Sequential(
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
    )

    # print(resnet50)

    samples_from_wsi_count = 64 * 30

    res_dict = dict()

    for filename in os.listdir(tensor_batches_dir):
        tensor_out_name = os.path.join(tensor_batches_dir, filename)
        res_dict[filename] = tensor_file_resnet50_forward(tensor_out_name, num_samples=samples_from_wsi_count)

    ## take out tumor 016 for test

    # tumor_016_tumor_test = res_dict.pop('tumor_016_tumor', None)
    # tumor_016_normal_test = res_dict.pop('tumor_016_normal', None)
    # tumor_test__tensor = torch.cat(tumor_016_tumor_test, dim=0)
    # normal_test__tensor = torch.cat(tumor_016_normal_test, dim=0)

    normal_total_tensor = None
    tumor_total_tensor = None
    for key, val in res_dict.items():
        if '_normal' in key:
            if normal_total_tensor is None:
                normal_total_tensor = torch.cat(val, dim=0)
            else:
                to_add = torch.cat(val, dim=0)
                print(to_add.shape)
                normal_total_tensor = torch.cat((normal_total_tensor, to_add), dim=0)
        if '_tumor' in key:
            if tumor_total_tensor is None:
                tumor_total_tensor = torch.cat(val, dim=0)
            else:
                to_add = torch.cat(val, dim=0)
                print(to_add.shape)
                tumor_total_tensor = torch.cat((tumor_total_tensor, to_add), dim=0)

    # tumor_total_tensor = torch.cat((tumor_total_tensor, tumor_test__tensor), dim=0)
    # normal_total_tensor = torch.cat((normal_total_tensor, normal_test__tensor), dim=0)

    tumor_total_tensor = tumor_total_tensor.squeeze()
    normal_total_tensor = normal_total_tensor.squeeze()
    out_tensor_path = os.path.join('/', 'databases', 'hawahaitam', 'tumor_tensor_FILE_1.pt')
    torch.save(tumor_total_tensor, out_tensor_path)
    out_tensor_path = os.path.join('/', 'databases', 'hawahaitam', 'normal_tensor_FILE_2.pt')
    torch.save(normal_total_tensor, out_tensor_path)

    print("normal_total_tensor", normal_total_tensor.shape)
    print("tumor_total_tensor", tumor_total_tensor.shape)
    exit()

    # normal-train normal test , tumor train, ...
    total_tensor = torch.cat((normal_total_tensor, tumor_total_tensor))
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(total_tensor)
    tumor_count = tumor_total_tensor.shape[0]  # total_tensor.shape[0]
    normal_count = normal_total_tensor.shape[0]
    normal_test_count = normal_test__tensor.shape[0]
    tumor_test_count = tumor_test__tensor.shape[0]
    print("normal_test_count", normal_test_count)
    print("tumor_test_count", tumor_test_count)
    if normal_count > 0:
        plt.scatter(X_embedded[:normal_count - normal_test_count, 0], X_embedded[:normal_count - normal_test_count, 1],
                    color='#88c999')
        plt.scatter(X_embedded[normal_count - normal_test_count:normal_count, 0],
                    X_embedded[normal_count - normal_test_count:normal_count, 1], color='#000000')
    if tumor_count > 0:
        plt.scatter(X_embedded[normal_count:-tumor_test_count, 0], X_embedded[normal_count:-tumor_test_count, 1],
                    color='hotpink')
        plt.scatter(X_embedded[-tumor_test_count:, 0], X_embedded[-tumor_test_count:, 1], color='#0000ff')
    plt.legend(["Normal", "Normal predicted", "Tumor", "Tumor predicted"])
    plt.show()

    exit()

    from multiprocessing import Process, Manager


    def worker_job(wsi_filename, dict_to_fill, num_of_normal_samples, num_of_tumor_samples=0):
        resnet50 = models.resnet50(pretrained=True)
        # model :torch.nn.Module = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

        resnet50.eval()
        print(resnet50)

        my_model = torch.nn.Sequential(*(list(resnet50.children())[:-1]))  # strips off last linear layer
        print('hello', wsi_filename)
        tumor_samples_out = 0
        dir_name = os.path.join(tensor_batches_dir, wsi_filename + '_tumor')
        for filename in os.listdir(dir_name):
            print(filename)
            batch_tensor = torch.load(os.path.join(dir_name, filename))
            with torch.no_grad():
                print("in with")
                forward = my_model(batch_tensor)
                print("forward")
                if wsi_filename + '_tumor' not in dict_to_fill:
                    dict_to_fill[wsi_filename + '_tumor'] = [forward]
                else:
                    dict_to_fill[wsi_filename + '_tumor'] += [forward]
                print(tumor_samples_out)
                tumor_samples_out += 64
                if tumor_samples_out >= num_of_tumor_samples:
                    break

        normal_samples_out = 0
        dir_name = os.path.join(tensor_batches_dir, wsi_filename + '_normal')
        for filename in os.listdir(dir_name):
            batch_tensor = torch.load(os.path.join(dir_name, filename))
            with torch.no_grad():
                forward = my_model(batch_tensor)
                if wsi_filename + '_normal' not in dict_to_fill:
                    dict_to_fill[wsi_filename + '_normal'] = [forward]
                else:
                    dict_to_fill[wsi_filename + '_normal'] += [forward]
                normal_samples_out += 64
                if normal_samples_out >= num_of_normal_samples:
                    break
        print(f"done extracting {normal_samples_out} normal samples, {tumor_samples_out} tumor samples")


    normal_samples_from_wsi_count = 64
    tumor_samples_from_wsi_count = 64

    with Manager() as manager:
        forward_final_result = manager.dict()  # key=img_name, val = [forward list]
        jobs = [
            Process(target=worker_job, args=(
                'tumor_009', forward_final_result, normal_samples_from_wsi_count, tumor_samples_from_wsi_count)),
            Process(target=worker_job, args=(
                'tumor_001', forward_final_result, normal_samples_from_wsi_count, tumor_samples_from_wsi_count)),
            Process(target=worker_job, args=(
                'tumor_002', forward_final_result, normal_samples_from_wsi_count, tumor_samples_from_wsi_count)),
            Process(target=worker_job, args=(
                'tumor_003', forward_final_result, normal_samples_from_wsi_count, tumor_samples_from_wsi_count)),
            Process(target=worker_job, args=(
                'tumor_016', forward_final_result, normal_samples_from_wsi_count, tumor_samples_from_wsi_count)),
            Process(target=worker_job, args=('normal_001', forward_final_result, normal_samples_from_wsi_count, 0)),
            Process(target=worker_job, args=('normal_002', forward_final_result, normal_samples_from_wsi_count, 0))
        ]

        for p in jobs:
            p.start()
            break
        print("proccesses lnched")
        for p in jobs:
            p.join()
            break
            print(f"proccess {p} finished")
        print(forward_final_result)

        import json

        with open(os.path.join(tensor_batches_dir, 'dict_exmaple.txt'), 'w') as convert_file:
            convert_file.write(json.dumps(forward_final_result))
    exit()

    ################################################ normal 001 #############################################
    file_path = os.path.join('/', 'databases', 'hawahaitam', 'normal_001.tif')
    xml_file_path = ''  # os.path.join('/', 'databases', 'hawahaitam', 'tumor_001.xml')
    out_dir_path = os.path.join('/', 'data', 'hawahaitam', 'tumor_001_normal')
    extractor = PatchExtractor(wsi_path=file_path, xml_path=xml_file_path, patches_in_batch=64,
                               size=(512, 512), extract_type=ExtractType.normal_only, overlap=0, wsi_level=0)
    extractor.extract_path_batches_to_tensors(out_dir_path)
    ################################################ normal 002 #############################################
    file_path = os.path.join('/', 'databases', 'hawahaitam', 'normal_002.tif')
    xml_file_path = ''  # os.path.join('/', 'databases', 'hawahaitam', 'tumor_001.xml')
    out_dir_path = os.path.join('/', 'data', 'hawahaitam', 'tumor_002_normal')
    extractor = PatchExtractor(wsi_path=file_path, xml_path=xml_file_path, patches_in_batch=64,
                               size=(512, 512), extract_type=ExtractType.normal_only, overlap=0, wsi_level=0)
    extractor.extract_path_batches_to_tensors(out_dir_path)

    exit()
    ################################################ tumor 001 #############################################
    file_path = os.path.join('/', 'databases', 'hawahaitam', 'tumor_001.tif')
    xml_file_path = os.path.join('/', 'databases', 'hawahaitam', 'tumor_001.xml')
    out_dir_path = os.path.join('/', 'data', 'hawahaitam', 'tumor_001_normal')
    extractor = PatchExtractor(wsi_path=file_path, xml_path=xml_file_path, patches_in_batch=64,
                               size=(512, 512), extract_type=ExtractType.normal_only, overlap=0, wsi_level=0)
    extractor.extract_path_batches_to_tensors(out_dir_path)
    extractor.resetITR()
    extractor.extract_type = ExtractType.tumor_only
    out_dir_path = os.path.join('/', 'data', 'hawahaitam', 'tumor_001_tumor')
    extractor.extract_path_batches_to_tensors(out_dir_path)

    ################################################ tumor 002 #############################################
    file_path = os.path.join('/', 'databases', 'hawahaitam', 'tumor_002.tif')
    xml_file_path = os.path.join('/', 'databases', 'hawahaitam', 'tumor_002.xml')
    out_dir_path = os.path.join('/', 'data', 'hawahaitam', 'tumor_002_normal')
    extractor = PatchExtractor(wsi_path=file_path, xml_path=xml_file_path, patches_in_batch=64,
                               size=(512, 512), extract_type=ExtractType.normal_only, overlap=0, wsi_level=0)
    extractor.extract_path_batches_to_tensors(out_dir_path)
    extractor.resetITR()
    extractor.extract_type = ExtractType.tumor_only
    out_dir_path = os.path.join('/', 'data', 'hawahaitam', 'tumor_002_tumor')
    extractor.extract_path_batches_to_tensors(out_dir_path)

    ################################################ tumor 003 #############################################
    file_path = os.path.join('/', 'databases', 'hawahaitam', 'tumor_003.tif')
    xml_file_path = os.path.join('/', 'databases', 'hawahaitam', 'tumor_003.xml')
    out_dir_path = os.path.join('/', 'data', 'hawahaitam', 'tumor_003_normal')
    extractor = PatchExtractor(wsi_path=file_path, xml_path=xml_file_path, patches_in_batch=64,
                               size=(512, 512), extract_type=ExtractType.normal_only, overlap=0, wsi_level=0)
    extractor.extract_path_batches_to_tensors(out_dir_path)
    extractor.resetITR()
    extractor.extract_type = ExtractType.tumor_only
    out_dir_path = os.path.join('/', 'data', 'hawahaitam', 'tumor_003_tumor')
    extractor.extract_path_batches_to_tensors(out_dir_path)

    ################################################ tumor 016 #############################################
    file_path = os.path.join('/', 'databases', 'hawahaitam', 'tumor_016.tif')
    xml_file_path = os.path.join('/', 'databases', 'hawahaitam', 'tumor_016.xml')
    out_dir_path = os.path.join('/', 'data', 'hawahaitam', 'tumor_016_normal')
    extractor = PatchExtractor(wsi_path=file_path, xml_path=xml_file_path, patches_in_batch=64,
                               size=(512, 512), extract_type=ExtractType.normal_only, overlap=0, wsi_level=0)
    extractor.extract_path_batches_to_tensors(out_dir_path)
    extractor.resetITR()
    extractor.extract_type = ExtractType.tumor_only
    out_dir_path = os.path.join('/', 'data', 'hawahaitam', 'tumor_016_tumor')
    extractor.extract_path_batches_to_tensors(out_dir_path)

    exit()
    im = wsi.get_thumbnail(size=(1024, 1024))
    img_gray = ImageOps.grayscale(im)

    for i in [7, 15, 25, 55, 105]:
        # tmp_img =
        blur = cv2.GaussianBlur(np.array(img_gray), (i, i), 0)
        # apply binary thresholding
        ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_OTSU)
        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        # draw contours on the original image
        image_copy = np.array(im.copy())
        valid_contours = []
        for cont in contours:
            if len(cont) > 10:
                valid_contours += [cont]
        # for con in contours:

        cv2.drawContours(image=image_copy, contours=valid_contours, contourIdx=-1, color=(0, 255, 0), thickness=1,
                         lineType=cv2.LINE_AA)

        # see the results
        cv2.imshow(f'kernel size = {i}', image_copy)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

    ret, thresh = cv2.threshold(np.array(img_gray), 0.1, 0, 1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    # img = ImageOps.grayscale()
    im.show()

    extractor = PatchExtractor(wsi_path=file_path, xml_path=xml_file_path, patches_in_batch=64,
                               size=(512, 512), extract_type=ExtractType.normal_only, overlap=0, wsi_level=0)
    extractor.extract_path_batches_to_tensors(out_dir_path)
    extractor.extract_type = ExtractType.tumor_only
    extractor.resetITR()
    extractor.extract_path_batches_to_tensors(out_dir_path)

    exit()
    file_path = os.path.join('/', 'databases', 'hawahaitam', 'tumor_009.tif')
    xml_file_path = os.path.join('/', 'databases', 'hawahaitam', 'tumor_009.xml')
    out_dir_path = os.path.join('/', 'data', 'hawahaitam', 'temp_9_6')
    extract_forward_tensors(file_path, xml_file_path, out_dir_path, tumor=True)
    print("extract_forward_tensors(file_path,xml_file_path,out_dir_path,tumor=True) ----------- DONE")
    extract_forward_tensors(file_path, xml_file_path, out_dir_path, tumor=False)
    print("extract_forward_tensors(file_path,xml_file_path,out_dir_path,tumor=False) ----------- DONE")
    apply_TSNE(out_dir_path)

    exit()
    pe = PatchExtractor(wsi_path=file_path, xml_path=xml_file_path, size=(512, 512), patches_in_batch=6, tumor=True,
                        overlap=0, wsi_level=0)
    for result, extracted_succ in pe:
        transform = transforms.ToPILImage()

        img = transform(result[0]).convert("RGB")
        img.show()
        break

    exit()

    anota = AnnotationHandler(xml_file_path)

    exit()
    wsi = openslide.open_slide(file_path)

    tu = wsi.get_thumbnail((512, 512))

    transform = transforms.Compose([transforms.PILToTensor()])

    tensor_img = transform(tu)

    show_image_RGBA(tensor_img)

    exit()
    # file_path = os.path.join('/','home', 'administrator', 'temp', 'normal_004.tif')
    # file_path = os.path.join('/', 'home', 'administrator', 'temp', 'tumor_009.tif')
    '''
    file_path = os.path.join('normal', 'normal_0.png')
    image = Image.open(file_path).convert('RGB')
    print("getting resnet")
    resnet50 = models.resnet50(pretrained=True)
    #model :torch.nn.Module = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

    resnet50.eval()
    # print(resnet50)

    my_model = torch.nn.Sequential(*list(resnet50.modules())[:-1])  # strips off last linear layer

    transform = transforms.Compose([transforms.PILToTensor()])



    tensor = torch.unsqueeze(transform(image),0)
    print(tensor.shape)
    new_tensor = torch.zeros(size=(64,3,512,512))
    for i in range(64):
        new_tensor[i] = torch.clone(tensor)
    print(new_tensor.shape)
    # exit()
    '''
    print("start")
    extract_features()
    # dataset = torchvision.datasets.ImageFolder(os.path.join(os.path.split(os.path.realpath(__file__))[0],'dataset'),transform=transforms.ToTensor())

    exit()
    print(X_embedded.shape)
    torchvision.utils.save_image(torch.from_numpy(X_embedded), os.path.join('dataset', 'tumor_ext',
                                                                            f'tumor_{X_embedded.shape[0]}_{X_embedded.shape[1]}.jpg'))
    # shoukd paint it as graph, scattr
