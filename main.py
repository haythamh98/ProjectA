import os, sys
import time

import PIL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def big_file_Walkaround():
    PIL.Image.MAX_IMAGE_PIXELS = 21079261184

def show_image_by_path(image_path):
    img = mpimg.imread(image_path)
    imgplot = plt.imshow(img)
    plt.show()

def show_image_RGBA(image):
    imgplot = plt.imshow(image)
    plt.show()


def openslide_walkaround():  # TODO: based on your system
    os.add_dll_directory(r"C:\Users\haytham\Downloads\openslide-win64-20171122\openslide-win64-20171122\bin")



if __name__ == '__main__':
    big_file_Walkaround()
    print('import openslide')
    os.add_dll_directory(r"C:\Users\haytham\Downloads\openslide-win64-20171122\openslide-win64-20171122\bin")
    import openslide
    # from openslide.deepzoom import DeepZoomGenerator
    print('done import openslide')

    # windowsOS
    file_path = os.path.join('..', 'camelyon16', 'tumor', 'tumor_010.tif')
    #file_path = os.path.join('/','home', 'administrator', 'temp', 'tumor_001.tif')

    wsi = openslide.ImageSlide(file_path)
    print(wsi.associated_images)
    print(wsi.properties)
    h,w  = wsi.level_dimensions[0][0]//1000,wsi.level_dimensions[0][1]//1000

    # tmp_img = wsi.get_thumbnail(size=((h,w)))
    #small_img = wsi.read_region(location=(0,0),level=0,size=(97792, 221184))
    #show_image_RGBA(small_img)
    # show_image_RGBA(tmp_img)

    print('wsi.dimensions',wsi.dimensions)
    print('wsi.level_downsamples',wsi.level_downsamples)
    #exit()
    sample = wsi.read_region((1000, 1000), 0, (8, 8))
    print('hhhhhhh')
    show_image_RGBA(sample)

    # dz = DeepZoomGenerator(wsi, tile_size=256, overlap=0)
    print("level_dimensions",dz.level_dimensions)
    print("tile_count",dz.tile_count)
    print("level_count",dz.level_count)
    print("level_tiles",dz.level_tiles)
    print("get_tile_coordinates",dz.get_tile_coordinates(level=5,address=((0,0))))
    # tile = dz.get_tile(11,address=(0,0))