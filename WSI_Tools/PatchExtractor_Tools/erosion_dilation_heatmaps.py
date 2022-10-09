import numpy as np
import cv2 as cv
import math



def calculate_template_space(temp_side_length):
        return int(temp_side_length/2)


def erosion(image, template_side_length, template):
    new_image = np.zeros(image.shape, image.dtype)
    # Coordinates are provided as (y,x), where the origin is at the top left of the image
    # So always remember that (-) is used instead of (+) to iterate
    template_space = calculate_template_space(template_side_length)
    half_template = int((template_side_length - 1) / 2)

    for x in range(template_space, new_image.shape[1] - template_space):
        for y in range(template_space, new_image.shape[0] - template_space):
            minimum = 256
            for c in range(0, template_side_length):
                for d in range(0, template_side_length):
                    a = x - half_template - 1 + c
                    b = y - half_template - 1 + d
                    sub = image[b, a] - template[d, c]
                    if sub < minimum:
                        if sub > 0:
                            minimum = sub
            new_image[y, x] = int(minimum)
    return new_image


def dilation(image, template_side_length, template):
    new_image = np.zeros(image.shape, image.dtype)
    # Coordinates are provided as (y,x), where the origin is at the top left of the image
    # So always remember that (-) is used instead of (+) to iterate
    template_space = calculate_template_space(template_side_length)
    half_template = int((template_side_length - 1) / 2)

    for x in range(template_space, new_image.shape[1] - template_space):
        for y in range(template_space, new_image.shape[0] - template_space):
            maximum = 0
            for c in range(0, template_side_length):
                for d in range(0, template_side_length):
                    a = x - half_template - 1 + c
                    b = y - half_template - 1 + d
                    sub = image[b, a] - template[d, c]
                    if sub > maximum:
                        if sub > 0:
                            maximum = sub
            new_image[y, x] = int(maximum)
    return new_image




if __name__ == "__main__":
    # Median Filter
    #
    # img = cv.imread("/home/vignesh/PycharmProjects/COS791_Ass1/median_erosion_dilation_image_filters/images/rotated_fence.png", cv.IMREAD_GRAYSCALE)
    # filter_size = 7
    # new_img = median_filter(img, filter_size)
    # cv.imwrite("/home/vignesh/PycharmProjects/COS791_Ass1/median_erosion_dilation_image_filters/images/rotated_fence_" + str(filter_size) + "_.png", new_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    #

    # Erosion
    img = cv.imread("/databases/hawahaitam/heatmaps/heatmap_patient_38_node_2_img.png", cv.IMREAD_GRAYSCALE)
    
    filter_size = 3
    temp = np.zeros(img.shape, img.dtype)
    new_img = erosion(img, filter_size, temp)
    cv.imwrite("/databases/hawahaitam/heatmaps/heatmap_patient_38_node_2_img_" + "after erosion" + "_.png", new_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Dilation
    img = cv.imread("/databases/hawahaitam/heatmaps/heatmap_patient_38_node_2_img.png", cv.IMREAD_GRAYSCALE)
    filter_size = 3
    temp = np.zeros(img.shape, img.dtype)
    new_img = dilation(img, filter_size, temp)
    cv.imwrite("/databases/hawahaitam/heatmaps/heatmap_patient_38_node_2_img_" + "after dilation" + "_.png", new_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
