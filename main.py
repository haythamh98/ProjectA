import os, sys
from ctypes import cdll
from os import path

import PIL
#import cv2
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from openslide.deepzoom import DeepZoomGenerator
import torch
from torch import Tensor
from torchvision import datasets, transforms, models

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from ctypes.util import find_library

import openslide
# from CLAM.wsi_core.WholeSlideImage import WholeSlideImage


import xml.dom.minidom
from xml.dom import minidom

from WSI_Tools.AnnotationHandler import AnnotationHandler
from WSI_Tools.PatchExtractor import PatchExtractor,ExtractType


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


def extract_forward_tensors(wsi_file_path,xml_file_path,output_path_dir,tumor:bool):
    '''
    >>> t = torch.tensor([1., 2.])
    >>> torch.save(t, 'tensor.pt')
    >>> torch.load('tensor.pt')
    tensor([1., 2.])
    '''
    resnet50 = models.resnet50(pretrained=True)
    resnet50.eval()
    my_model = torch.nn.Sequential(*(list(resnet50.children())[:-1]))  # strips off last linear layer

    extractor = PatchExtractor(wsi_path=wsi_file_path, xml_path= xml_file_path, patches_in_batch= 64, size = (512, 512), tumor = tumor, overlap= 0, wsi_level= 0)
    for i,(batch,samples_count) in enumerate(extractor):
        transform = transforms.ToPILImage()

        # img = transform(batch[0]).convert("RGB")
        # img.show()
        with torch.no_grad():
            forward = my_model(batch)
            type_name = 'with_tumor' if tumor else 'without_tumor'
            out_tensor_path = os.path.join(output_path_dir,f'{extractor.WSI_type}_{extractor.WSI_ID}_{type_name}_{i}.pt')
            if samples_count != 64:
                forward = forward[:samples_count]
            torch.save(forward,out_tensor_path)
            print(f"file {out_tensor_path} was done with {samples_count}")





def apply_TSNE(tensor_dir):

    with_tumor = torch.randn(0)  # torch.zeros(size=(64,2048))
    wo_tumor = torch.randn(0)  # torch.zeros(size=(64,2048))
    for tensor_file_name in os.listdir(tensor_dir):
        loaded_tensor = torch.load(os.path.join(tensor_dir,tensor_file_name))
        loaded_tensor = torch.squeeze(loaded_tensor)
        data_has_tumor = ('with_tumor' in tensor_file_name)
        print(f"loading file {tensor_file_name} laoded shape {loaded_tensor.shape}")
        if data_has_tumor:
            if with_tumor is None:
                with_tumor = torch.clone(loaded_tensor)
            else:
                with_tumor = torch.cat((with_tumor,loaded_tensor),0)
        else:
            if wo_tumor is None:
                wo_tumor = torch.clone(loaded_tensor)
            else:
                wo_tumor = torch.cat((wo_tumor, loaded_tensor),0)
    total_tensor = None
    if wo_tumor is None:
        total_tensor = with_tumor
    elif with_tumor is None:
        total_tensor = wo_tumor
    else:
        total_tensor = torch.cat((wo_tumor,with_tumor),0)
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(total_tensor)
    tumor_count = with_tumor.shape[0]
    no_tumor_count = wo_tumor.shape[0]
    if no_tumor_count > 0:
        plt.scatter(X_embedded[:no_tumor_count, 0], X_embedded[:no_tumor_count, 1], color='#88c999')
    if tumor_count > 0:
        plt.scatter(X_embedded[no_tumor_count:, 0], X_embedded[no_tumor_count:, 1], color='hotpink')
    plt.legend(["Normal", "Tumor"])
    plt.show()





if __name__ == '__main__':
    print("Hi")
    # big_file_Walkaround()
    # os.add_dll_directory(r"C:\Users\haytham\Downloads\openslide-win64-20171122\openslide-win64-20171122\bin") # for windows only
    file_path = os.path.join('/', 'databases', 'hawahaitam', 'tumor_009.tif')
    xml_file_path = os.path.join('/', 'databases', 'hawahaitam', 'tumor_009.xml')
    out_dir_path = os.path.join('/', 'data', 'hawahaitam', 'temp_11_6')


    extractor = PatchExtractor(wsi_path=file_path, xml_path= xml_file_path, patches_in_batch= 64,
                   size = (512, 512), extract_type=ExtractType.normal_only, overlap= 0, wsi_level= 0).extract_path_batches_to_tensors(output_path_dir=out_dir_path)
    extractor.extract_path_batches_to_tensors(out_dir_path)
    extractor.extract_type = ExtractType.tumor_only
    extractor.resetITR()
    extractor.extract_path_batches_to_tensors(out_dir_path)


    exit()
    file_path = os.path.join('/', 'databases', 'hawahaitam', 'tumor_009.tif')
    xml_file_path = os.path.join('/', 'databases', 'hawahaitam', 'tumor_009.xml')
    out_dir_path = os.path.join('/', 'data', 'hawahaitam', 'temp_9_6')
    extract_forward_tensors(file_path,xml_file_path,out_dir_path,tumor=True)
    print("extract_forward_tensors(file_path,xml_file_path,out_dir_path,tumor=True) ----------- DONE")
    extract_forward_tensors(file_path,xml_file_path,out_dir_path,tumor=False)
    print("extract_forward_tensors(file_path,xml_file_path,out_dir_path,tumor=False) ----------- DONE")
    apply_TSNE(out_dir_path)


    exit()
    pe = PatchExtractor(wsi_path=file_path,xml_path=xml_file_path,size=(512,512),patches_in_batch=6,tumor=True,overlap=0,wsi_level=0)
    for result,extracted_succ in pe:
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
