
# script for symbol training

import cv2
import numpy as np
from glob import *
from symbol import Symbol, LOC, BBox
from sklearn.svm import SVC
from sklearn.externals import joblib
from utils import hog


def read_annotations(annotation_file_path):
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


def get_symbol(label, loc):
    if label not in symbol_label_parms:
        return None
    s = Symbol(label, None)

    if label in symbol_label_parms.keys():
        [rows, cols] = symbol_label_parms[label]
    else:
        rows = default_symbol_rows
        cols = default_symbol_cols
    s.set_bbox(loc, rows, cols)
    return s


def get_sub_im(im, s):
    bbox = s.get_bbox()
    return im[bbox.loc.y:bbox.loc.y + bbox.rows, bbox.loc.x:bbox.loc.x + bbox.cols]


def prepare_data_from_annotation(im, annotations, label):
    pos_data = []
    neg_data = []

    try:
        positions = annotations[label]
        symbols = [get_symbol(label, loc) for loc in positions]
        pos_data = [get_sub_im(im, s) for s in symbols if s is not None]

        for key in annotations.keys():
            if key != label: #and key in symbol_label_parms.keys():
                positions = annotations[key]
                symbols = [get_symbol(label, loc) for loc in positions]
                neg_data = neg_data + [get_sub_im(im, s) for s in symbols if s is not None]
    except Exception:
        print "nothing"
    return pos_data, neg_data


def training(img_file_path, annotation_file_path, detector_name):
    im = cv2.imread(img_file_path, 0)
    annotations = read_annotations(annotation_file_path)
    pos_data, neg_data = prepare_data_from_annotation(im, annotations, "treble_clef")
    img_data = pos_data + neg_data

    for pos in pos_data:
        cv2.imshow("image", pos)
        cv2.waitKey(0)

    if detector_name == "hog":

        # winSize = (64, 64)
        # blockSize = (16, 16)
        # blockStride = (8, 8)
        # cellSize = (8, 8)
        # nbins = 9
        # derivAperture = 1
        # winSigma = 4.
        # histogramNormType = 0
        # L2HysThreshold = 2.0000000000000001e-01
        # gammaCorrection = 0
        # nlevels = 64
        # hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma, histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
        # hog.save("../models/hog.xml")
        # # winStride = (8, 8)
        # # padding = (8, 8)
        # locations = ((0,0),)
        # hog_data = [hog.compute(im,None, None, locations) for im in img_data]
        hog_data = [hog(im) for im in img_data]
        #hog_data.append(np.ones(hog_data[0].size)) #add background

        #print hog_data[0].shape
        #return
        train_data = np.array(np.float32(hog_data))
        responses = np.array([1]*len(pos_data) + [0]*(len(neg_data)+0))

        # svm = cv2.ml.SVM_create()
        # svm.setType(cv2.ml.SVM_C_SVC)
        # svm.setKernel(cv2.ml.SVM_LINEAR)
        # svm.train(train_data, cv2.ml.ROW_SAMPLE, responses)
        # print svm.predict(train_data)
        #svm.save('hog_svm.dat')

        clf = SVC(kernel = 'linear', C = 2.67, max_iter = 5000000, verbose = True)
        clf.fit(train_data, responses)
        print len(pos_data), len(neg_data)
        print clf.predict(train_data)
        joblib.dump(clf, '../models/hog_svm.pkl')