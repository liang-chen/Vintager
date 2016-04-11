
# detect various types of music symbols via different detectors
# 1, HOG + Muticlass-SVM
# 2, Convolutional Neural Network

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
                sub_im = cv2.resize(sub_im, uni_size)
                #if self.model.predict(self.extractor.compute(sub_im, None, None, ((0,0),)).reshape(-1,1764)) == 1:
                feature = self.extractor(sub_im)
                feature = feature.reshape(-1,feature.size)
                #print i,j
                #print self.model.predict_proba(feature)[0][0], self.model.predict_proba(feature)[0][1]
                #if 10*self.model.predict_proba(feature)[0][0] <= self.model.predict_proba(feature)[0][1]:
                if self.model.predict(feature) == label:
                    detected.append((i,j))
                    #print i,j

        if mode == "show":
            rgb_im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            font = cv2.FONT_HERSHEY_SIMPLEX
            for (i, j) in detected:
                cv2.rectangle(rgb_im, (j, i), (j + cols, i + rows), (0, 255, 0), 2)
                cv2.putText(rgb_im, label, (j - 10, i), font, 1, (0, 255, 0), 2)
            cv2.imshow("image", rgb_im)
            cv2.waitKey(0)
        elif mode == "noshow":
            return
        else:
            raise Exception

    def detect_all(self, im, mode):
        tot_rows, tot_cols = im.shape[:2]
        detected = []
        step_size = 3
        cls = self.model.classes_
        for i in xrange(0, tot_rows, step_size):
            for j in xrange(0, tot_cols, step_size):
                prob = 0
                pred_label = "background"
                print i,j
                for label in symbol_label_parms.keys():
                    [rows, cols] = symbol_label_parms[label]
                    rows = int(rows)
                    cols = int(cols)
                    if i + rows >= tot_rows or j + cols >= tot_cols:
                        continue
                    sub_im = im[i:i + rows, j:j + cols]
                    sub_im = cv2.resize(sub_im, uni_size)
                    feature = self.extractor(sub_im)
                    feature = feature.reshape(-1, feature.size)
                    temp_prob = self.model.predict_proba(feature)[0][np.where(cls == label)]
                    if temp_prob > prob:
                        prob = temp_prob
                        pred_label = label

                if pred_label != "background":
                    detected.append((i, j, pred_label))
                    # print i,j

        if mode == "show":
            rgb_im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            font = cv2.FONT_HERSHEY_SIMPLEX
            for (i, j, label) in detected:
                [rows, cols] = symbol_label_parms[label]
                rows = int(rows)
                cols = int(cols)
                cv2.rectangle(rgb_im, (j, i), (j + cols, i + rows), (0, 0, 255), 2)
                cv2.putText(rgb_im, label, (j - 20, i - 10), font, 0.4, (0, 0, 255), 1)
            cv2.imshow("image", rgb_im)
            cv2.waitKey(0)
        elif mode == "noshow":
            return
        else:
            raise Exception