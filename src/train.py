
# script for symbol training

import cv2
import numpy as np
import random
from globv import *
from symbol import Symbol, LOC, BBox
from sklearn.svm import SVC
from sklearn.externals import joblib
from utils import create_symbol
from feature import hog, pixel_vec


def read_annotations(annotation_file_path):
    """

    :param annotation_file_path: path to annotation file
    :type annotation_file_path: string
    :return: annotations
    :rtype: dict[label: [locations]]
    """


    annotations = {}

    try:
        with open(annotation_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parsed = [x.strip() for x in line.split()]
                label = parsed[0]
                loc = LOC(int(parsed[1]), int(parsed[2]))

                if label not in annotations:
                    annotations[label] = [loc]
                else:
                    annotations[label].append(loc)
    except Exception:
        print "nothing"

    return annotations


def get_sub_im(im, s):
    """
    crop subimage for a certain symbol **s** from the whole image **im**
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
    sub_im = cv2.resize(sub_im, uni_size)

    return sub_im


def is_in_loc_list(loc, ll):
    list = [[l.get_x(), l.get_y()] for l in ll]
    if [loc.get_x(), loc.get_y()] in list:
        return True
    else:
        return False


def prepare_background_data(im, locs):
    (rows, cols) = im.shape
    num = 0
    tot_neg_num = 1000
    data = []
    labels = []

    while num < tot_neg_num:
        y = random.randint(0, rows)
        x = random.randint(0, cols)
        l = LOC( x,y )
        if not is_in_loc_list(l, locs):
            s = create_symbol("background", l)
            sub_im = get_sub_im(im, s)

            if sub_im is not None:
                data.append(get_sub_im(im, s))
                num += 1
                labels.append("background")

    return data, labels


def prepare_data_from_annotation(im, annotations):

    data = []
    labels = []

    try:
        symbols = []
        pos_locs = []
        for label in annotations.keys():
            locs = annotations[label]
            pos_locs = pos_locs + locs
            sl = [create_symbol(label, loc) for loc in locs]
            symbols = symbols + sl
        data_label_pair = [(get_sub_im(im, s), s.get_label()) for s in symbols if s is not None]
        data = [d for (d,l) in data_label_pair if d is not None]
        labels = [l for (d,l) in data_label_pair if d is not None]

        bk_data, bk_labels = prepare_background_data(im, pos_locs)
        labels += bk_labels
        data += bk_data

        print len(labels), len(data)
    except Exception:
        print "prepare annotations"
    return data, labels


def training(img_file_path, annotation_file_path, detector_name):
    im = cv2.imread(img_file_path, 0)
    annotations = read_annotations(annotation_file_path)
    img_data, labels = prepare_data_from_annotation(im, annotations)

    features = []
    if detector_name == "hog":
        features = [hog(im) for im in img_data]
    elif detector_name == "pixel":
        features = [pixel_vec(im) for im in img_data]
    else:
        print("unknown detector.")
        raise Exception

    train_data = np.array(np.float32(features))
    labels = np.array(labels)

    clf = SVC(kernel='linear', C=2.67, decision_function_shape= "ovo", max_iter=5000000, probability=True, verbose=True)
    print train_data.shape
    print labels.shape
    clf.fit(train_data, labels)
    # print len(pos_data), len(neg_data)
    # print clf.predict(train_data)
    joblib.dump(clf, '../models/' + detector_name + '_svm.pkl')