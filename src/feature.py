
"""
Feature Extraction Module

Given a gray-level image, extract feature vectors using different approaches:

1) vectorize pixels
2) extract Histogram of Oriented Gradients(HOG)
"""

import cv2
import numpy as np
from globv import uni_size


def unify_img_size(img):
    return cv2.resize(img, uni_size)

def hog(img):
    """

    :param img: input image
    :type img: cv2.image
    :return: hist: output feature vector
    :rtype: numpy.array(1, 64)
    """
    #hijacking hog with pixel features for now
    feature = np.squeeze(img.reshape(-1, img.size) / 256.0)  # using pixel features
    return feature


    # bin_n = 16  # Number of bins
    # gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    # gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    # mag, ang = cv2.cartToPolar(gx, gy)
    # bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing bin values in (0...16)
    # bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    # mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    # hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    # hist = np.hstack(hists)     # hist is a 64 bit vector
    # hist /= 1000.0
    # return hist


def pixel_vec(img):
    """

    :param img: input image
    :type img: cv2.image
    :return: feature: output feature vector
    :rtype: numpy.array(1,img.size)
    """
    feature = np.squeeze(img.reshape(-1, img.size)/256.0) # using pixel features
    return feature