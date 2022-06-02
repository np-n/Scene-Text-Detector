"""
-- Create by: Netra Prasad Neupane
-- Created on: 5/18/22
"""

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageEnhance
import imutils
from skimage.color import rgb2gray
from skimage.transform import rotate
from deskew import determine_skew
import matplotlib.pyplot as plt
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)


def shadow_remove(img):
    """
    Method to remove the shadow for the given input image
    """
    # color_channels = ["blue","green","red"]
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for i,plane in enumerate(rgb_planes):
        # cv2.imshow('color-'+color_channels[i],plane)
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        # cv2.imshow('dilated'+str(i), dilated_img)
        bg_img = cv2.medianBlur(dilated_img, 21)
        # cv2.imshow('blur'+str(i), bg_img)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        # cv2.imshow('diff'+str(i), diff_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # cv2.imshow('norm'+str(i), norm_img)
        result_norm_planes.append(norm_img)
        # cv2.waitKey(0)

    shadow_removed = cv2.merge(result_norm_planes)

    return shadow_removed


def sharpen_image_v1(img):
    """
    OpenCV Method to sharpen image by filtering
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_img = cv2.filter2D(img, -1, kernel)
    # final_image = np.hstack((img,sharpened_img))
    # cv2.imshow("Final Image 1",final_image)

    return sharpened_img


def sharpen_image_v2(img):
    """
    OpenCV Method to sharpen image by adding two image and adjusting gradients
    """
    blur = cv2.GaussianBlur(img, (9,9), 10)
    sharpened_img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    final_image = np.hstack((img, sharpened_img))
    cv2.imshow("Final Image 2", final_image)

    return final_image


def sharpen_image_v3(img):
    """
    Pillow method to sharpen image
    """
    # Converting image from cv2 format to pil format
    img_cv = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cv)

    # Creating object of Sharpness class
    img_obj = ImageEnhance.Sharpness(img_pil)

    # showing resultant image
    img_enhanced = img_obj.enhance(4.0)

    # use numpy to convert the pil_image into a numpy array
    img_np = np.array(img_enhanced)

    # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that
    # the color is converted from RGB to BGR format
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # blur = cv2.GaussianBlur(img_cv, (3,3),0)
    # blur = cv2.blur(img_cv,(2,2))

    # cv2.imshow('Sharpen Image', img_cv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img_cv


def deskew_image(image):
    """
    Method to align the image according to it's skewness
    """
    grayscale = rgb2gray(image)
    angle = determine_skew(grayscale)
    rotated = rotate(image, angle, resize=True) * 255
    aligned_image = rotated.astype(np.uint8)


    return aligned_image



def enhance_image_v1(image):
    """
    Method to separate foreground pixels from background pixels using OTSU thresholding method
    :param image:
    :return:
    """
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, threshed_image = cv2.threshold(gray_image,
                            0,  # threshold value, ignored when using cv2.THRESH_OTSU
                            255,  # maximum value assigned to pixel values exceeding the threshold
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # thresholding
    final_image = cv2.merge((threshed_image, threshed_image, threshed_image))

    return final_image


def enhance_image_v2(image):
    """
    Method to enhance the image by enhancing contrast, brightness, color, sharpness and denoising
    """
    # Color conversion for the opencv image
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # Enhance contrast
    curr_con = ImageEnhance.Contrast(image)
    new_con = 1.5
    # Contrast enhanced by a factor of 1.5
    img_contrasted = curr_con.enhance(new_con)

    # Enhance brightness
    curr_bri = ImageEnhance.Brightness(img_contrasted)
    new_bri = 1.5
    # Brightness enhanced by a factor of 1.5
    img_brightened = curr_bri.enhance(new_bri)

    # Enhance color level
    curr_col = ImageEnhance.Color(img_brightened)
    new_col = 1.5
    # Color level enhanced by a factor of 1.5
    img_colored = curr_col.enhance(new_col)

    # Enhance Sharpness
    curr_sharp = ImageEnhance.Sharpness(img_colored)
    new_sharp = 1.5
    # Sharpness enhanced by a factor of 1.5
    img_sharped = curr_sharp.enhance(new_sharp)

    # denoising with fast means
    img_sharped = np.asarray(img_sharped)
    img_sharped_cv = cv2.cvtColor(img_sharped,cv2.COLOR_RGB2BGR)
    img_denoised = cv2.fastNlMeansDenoisingColored(img_sharped_cv, None, 10, 10, 7, 21)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(src=img_denoised, ddepth=-1, kernel=kernel)

    return image_sharp


def enhance_image_v3(image):
    """
    Method to enhance the image by denoising and enhancing contrast, sharpness
    """
    # For color image
    img_denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Color conversion for the opencv image
    image = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # Enhance contrast
    contrast_enhancer = ImageEnhance.Contrast(image)
    # Contrast enhanced by a factor of 1.5
    img_contrasted = contrast_enhancer.enhance(1.5)

    # Enhance Sharpness
    sharpness_enhancer = ImageEnhance.Sharpness(img_contrasted)
    # Sharpness enhanced by a factor of 1.5
    img_sharped = sharpness_enhancer.enhance(1.5)

    img_sharped = np.asarray(img_sharped)
    img_sharped_cv = cv2.cvtColor(img_sharped, cv2.COLOR_RGB2BGR)

    return img_sharped_cv

def enhance_image_v4(image):
    """
    Methods to return the image by performing different thresholding
    :param image:
    :return:
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = rgb2gray(image)

    matplotlib.rcParams['font.size'] = 9

    binary_global = image > threshold_otsu(image)

    window_size = 25
    thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.8)
    thresh_sauvola = threshold_sauvola(image, window_size=window_size)

    binary_niblack = image > thresh_niblack
    binary_sauvola = image > thresh_sauvola

    # plt.figure(figsize=(8, 7))
    # plt.subplot(2, 2, 1)
    # plt.imshow(image, cmap=plt.cm.gray)
    # plt.title('Original')
    # plt.axis('off')
    #
    # plt.subplot(2, 2, 2)
    plt.title('Global Threshold')
    plt.imshow(binary_global, cmap=plt.cm.gray)
    plt.axis('off')
    #
    # plt.subplot(2, 2, 3)
    # plt.imshow(binary_niblack, cmap=plt.cm.gray)
    # plt.title('Niblack Threshold')
    # plt.axis('off')

    # plt.subplot(2, 2, 4)
    # plt.imshow(binary_sauvola, cmap=plt.cm.gray)
    # plt.title('Sauvola Threshold')
    # plt.axis('off')

    plt.show()

def resize_image(img):
    if img.shape[0] > 400 or img.shape[0] < 400:
        img = imutils.resize(img,height=400)

    return img

