
# script for symbol training


import cv2
import numpy as np
from glob import *
from symbol import Symbol, LOC, BBox

def hog(img):
    print type(img)
    bin_n = 16  # Number of bins
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

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

    [rows, cols] = symbol_label_parms[label]
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
            if key is not label:
                positions = annotations[key]
                symbols = [get_symbol(label, loc) for loc in positions]
                neg_data = neg_data + [get_sub_im(im, s) for s in symbols if s is not None]
    except Exception:
        print "nothing"
    return pos_data, neg_data

def training(img_file_path, annotation_file_path, detector_name):
    im = cv2.imread(img_file_path, 0)
    annotations = read_annotations(annotation_file_path)
    pos_data, neg_data = prepare_data_from_annotation(im, annotations, "solid_note_head")
    img_data = pos_data + neg_data

    if detector_name == "hog":
        svm_params = dict(kernel_type=cv2.ml.SVM_LINEAR,
        svm_type = cv2.ml.SVM_C_SVC,
        C = 2.67, gamma = 5.383 )
        hog_data = [hog(im) for im in img_data]
        train_data = np.float32(hog_data).reshape(-1, 64)
        responses = np.array([1]*len(pos_data) +  [-1]*len(neg_data))
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setGamma(5.383)
        svm.setC(2.67)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.train(train_data, cv2.ml.ROW_SAMPLE, responses)
        svm.save('hog_svm.dat')