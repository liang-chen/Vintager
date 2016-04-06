
# detect various types of music symbols via different detectors
# 1, HOG + Muticlass-SVM
# 2, AutoEncoder

import cv2
import numpy as np
from sklearn.externals import joblib
from glob import *
from utils import hog

class DetectorOption:
    def __init__(self, name):
        self.name = name


class SymbolDetector:
    def __init__(self, option):
        try:
            if option.name == "hog_svm":
                dir = '../models/'
                self.model = joblib.load(dir + option.name + '.pkl')
                #self.extractor = cv2.HOGDescriptor(dir + "hog.xml")
                self.extractor = hog

        except Exception:
            print Exception


    def detect(self, im, label, mode):
        tot_rows, tot_cols = im.shape[:2]
        [rows, cols] = symbol_label_parms[label]
        detected = []
        rows = int(rows)
        cols = int(cols)
        sample_step = 10
        for i in range(0, tot_rows - rows, sample_step):
            for j in range(0, tot_cols - cols, sample_step):
                sub_im = im[i:i+rows, j:j+cols]

                #if self.model.predict(self.extractor.compute(sub_im, None, None, ((0,0),)).reshape(-1,1764)) == 1:
                if self.model.predict(self.extractor(sub_im).reshape(-1,64)) == 1:
                    detected.append((i,j))
                    #print i, j


        for (i,j) in detected:
            cv2.rectangle(im, (i, j), (i + rows, j + cols), (0, 255, 0), 2)
        if mode == "show":
            cv2.imshow("image", im)
            cv2.waitKey(0)
        elif mode == "noshow":
            return
        else:
            raise Exception