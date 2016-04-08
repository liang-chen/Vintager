
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
        sample_step = 2
        for i in range(0, tot_rows - rows, sample_step):
            for j in range(0, tot_cols - cols, sample_step):
                sub_im = im[i:i+rows, j:j+cols]
                #if self.model.predict(self.extractor.compute(sub_im, None, None, ((0,0),)).reshape(-1,1764)) == 1:
                feature = self.extractor(sub_im)
                feature = feature.reshape(-1,feature.size)
                #print i,j
                #print self.model.predict_proba(feature)[0][0], self.model.predict_proba(feature)[0][1]
                if 10*self.model.predict_proba(feature)[0][0] <= self.model.predict_proba(feature)[0][1]:
                    detected.append((i,j))
                    #print i,j

        if mode == "show":
            rgb_im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            for (i, j) in detected:
                cv2.rectangle(rgb_im, (j, i), (j + cols, i + rows), (0, 255, 0), 2)
            cv2.imshow("image", rgb_im)
            cv2.waitKey(0)
        elif mode == "noshow":
            return
        else:
            raise Exception