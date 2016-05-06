
"""
Feature Extraction Module

Given a gray-level image, extract feature vectors using different approaches:

1) vectorize pixels
2) extract Histogram of Oriented Gradients(HOG)
"""

import cv2
import numpy as np
from utils import unify_img_size


def hog(img):
    """
    Extract hog feature from image

    :param img: input image
    :type img: cv2.image
    :return: hist: output feature vector
    :rtype: numpy.array(1, 64)
    """

    u_img = unify_img_size(img)
    hog = cv2.HOGDescriptor("../models/hog.xml")
    feature = hog.compute(u_img)
    return np.squeeze(feature.reshape(-1, len(feature)))


def pixel_vec(img):
    """
    Extract pixel feature from image

    :param img: input image
    :type img: cv2.image
    :return: feature: output feature vector
    :rtype: numpy.array(1,img.size)
    """

    u_img = unify_img_size(img)
    feature = np.squeeze(u_img.reshape(-1, u_img.size)/256.0) # using pixel features
    return feature