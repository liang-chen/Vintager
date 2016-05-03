
"""
Utilities
"""

from globv import symbol_label_parms, uni_size
from symbol import Symbol
import cv2

def create_symbol_with_center_loc(label, loc):
    """
    Create a symbol object with its label and center location

    :param label: name of the symbol
    :type label: string
    :param loc: center location of the symbol
    :type loc: LOC
    :return: sym
    :rtype: Symbol
    """
    if label not in symbol_label_parms.keys():
        return None

    [rows, cols] = symbol_label_parms[label]
    sym = Symbol(label, None)
    sym.set_bbox_with_center(loc, rows, cols)
    return sym


def create_symbol_with_upper_left_corner(label, loc):
    """
    Create a symbol object with its label and upper-left corner

    :param label: name of the symbol
    :type label: string
    :param loc: center location of the symbol
    :type loc: LOC
    :return: sym
    :rtype: Symbol
    """
    if label not in symbol_label_parms.keys():
        return None

    [rows, cols] = symbol_label_parms[label]
    sym = Symbol(label, None)
    sym.set_bbox_with_upper_left_corner(loc, rows, cols)
    return sym


def get_sub_im(im, s):
    """
    Crop subimage for a certain symbol **s** from the whole image **im**

    :param im: whole image
    :type im: cv2.image
    :param s: symbol
    :type s: Symbol
    :return: sub_im
    :rtype: cv2.image
    """

    bbox = s.get_bbox()
    (rows, cols) = im.shape
    y = bbox.get_loc().get_y()
    x = bbox.get_loc().get_x()
    brows = bbox.get_rows()
    bcols = bbox.get_cols()
    if y < 0 or y >= rows or y + brows < 0 or y + brows >= rows:
        return None
    if x < 0 or x >= cols or x + bcols < 0 or x + bcols >= cols:
        return None
    sub_im = im[y:y + brows, x:x + bcols]

    return sub_im


def unify_img_size(img):
    """
    Rescale an image into uniform size

    :param img: input image
    :type img: cv2.image
    :return: image with a uniform size
    :rtype: cv2.image
    """

    return cv2.resize(img, uni_size)
