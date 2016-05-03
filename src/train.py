
"""
Training symbol detectors via Linear SVM
"""

import cv2
import numpy as np
import random
#from globv import *
from symbol import LOC
from sklearn.svm import SVC
from sklearn.externals import joblib
from utils import create_symbol_with_center_loc, get_sub_im
from feature import hog, pixel_vec


def read_annotations(annotation_file_path):
    """
    Read annotations given the annotation file path.

    :param annotation_file_path: path to annotation file
    :type annotation_file_path: string
    :return: annotations
    :rtype: dict{label: [locations]}
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


def is_in_loc_list(loc, ll):
    """
    Determine whether a location is in a location list

    :param loc: target location
    :type loc: LOC
    :param ll: location list
    :type ll: [LOC]
    :return: True or False
    :rtype: Boolean
    """
    list = [[l.get_x(), l.get_y()] for l in ll]
    if [loc.get_x(), loc.get_y()] in list:
        return True
    else:
        return False


def prepare_background_data(im, locs):
    """
    Crop subimages to prepare background data

    :param im: whole image
    :type im: cv2.image
    :param locs: locations for the annotated symbols
    :type locs: [LOC]
    :return: [background imgs], [background labels]
    :rtype: [cv2.image], [string]
    """
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
            s = create_symbol_with_center_loc("background", l)
            sub_im = get_sub_im(im, s)

            if sub_im is not None:
                data.append(get_sub_im(im, s))
                num += 1
                labels.append("background")

    return data, labels


def prepare_data_from_annotation(im, annotations):
    """
    Crop subimages for annotated symbols

    :param im: whole image
    :type im: cv2.image
    :param annotations: annotation dict
    :type annotations: dict{label: [LOC]}
    :return: [symbol images], [symbol labels]
    :rtype: [cv2.image], [string]
    """

    data = []
    labels = []

    try:
        symbols = []
        pos_locs = []
        for label in annotations.keys():
            locs = annotations[label]
            pos_locs = pos_locs + locs
            sl = [create_symbol_with_center_loc(label, loc) for loc in locs]
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


def train_svm(img_file_path, annotation_file_path, detector_name):
    """
    Train SVM classifier given symbol annotations and feature extractor.
    SVM parameters: kernel='linear', C=2.67, decision_function_shape= "ovo".
    Trained model will be saved as "../models/?_svm.pkl", ? is the feature extraction method.

    :param img_file_path: path to score image
    :type img_file_path: string
    :param annotation_file_path: path to annotations
    :type annotation_file_path: string
    :param detector_name: "hog" or "pixel" -- two types of features
    :type detector_name: string
    """
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


def train_cnn(img_file_path, annotation_file_path):
    """
    Train CNN multi-class classifier based on symbol annotations.

    :param img_file_path: path to score image
    :type img_file_path: string
    :param annotation_file_path: path to annotations
    :type annotation_file_path: string
    """
    im = cv2.imread(img_file_path, 0)
    annotations = read_annotations(annotation_file_path)
    img_data, labels = prepare_data_from_annotation(im, annotations)

