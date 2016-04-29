
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
    """

    :param img: input image
    :type img: cv2.image
    :return: image with a uniform size
    :rtype: cv2.image
    """

    return cv2.resize(img, uni_size)


def hog(img):
    """

    :param img: input image
    :type img: cv2.image
    :return: hist: output feature vector
    :rtype: numpy.array(1, 64)
    """

    u_img = unify_img_size(img)
    hog = cv2.HOGDescriptor("../models/hog.xml")
    #cv2.imshow("haha", u_img)
    #cv2.waitKey(0)

    feature = hog.compute(u_img)
    return np.squeeze(feature.reshape(-1, len(feature)))


def pixel_vec(img):
    """

    :param img: input image
    :type img: cv2.image
    :return: feature: output feature vector
    :rtype: numpy.array(1,img.size)
    """

    u_img = unify_img_size(img)
    feature = np.squeeze(u_img.reshape(-1, img.size)/256.0) # using pixel features
    return feature