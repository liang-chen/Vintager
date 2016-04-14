
# detect various types of music symbols via different detectors
# 1, Pixel + Multiclass-SVM
# 2, HOG + Muticlass-SVM
# 3, Convolutional Neural Network

import cv2
import numpy as np
from sklearn.externals import joblib
from glob import *
#from utils import create_symbol
from feature import hog, pixel_vec


class DetectorOption:
    def __init__(self, name):
        self.__name__ = name

    def name(self):
        return self.__name__


class SymbolDetector:
    def __init__(self, option):
        try:
            model_dir = '../models/'
            self.model = joblib.load(model_dir + option.name() + '.pkl')
            if option.name() == "hog_svm":
                self.extractor = hog
            elif option.name() == "pixel_svm":
                self.extractor = pixel_vec

        except Exception:
            print "detector initialization"
            print Exception

    def detect(self, im, label, mode):
        tot_rows, tot_cols = im.shape[:2]
        [rows, cols] = symbol_label_parms[label]
        detected = []
        step_size = 2
        for i in range(0, tot_rows - rows, step_size):
            for j in range(0, tot_cols - cols, step_size):
                sub_im = im[i:i+rows, j:j+cols]
                sub_im = cv2.resize(sub_im, uni_size)
                feature = self.extractor(sub_im)
                feature = feature.reshape(-1,feature.size)
                if self.model.predict(feature) == label:
                    detected.append((i,j))

        if mode == "show":
            rgb_im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            font = cv2.FONT_HERSHEY_SIMPLEX
            for (i, j) in detected:
                cv2.rectangle(rgb_im, (j, i), (j + cols, i + rows), (0, 0, 255), 2)
                cv2.putText(rgb_im, label, (j, i - 10), font, 0.4, (0, 0, 255), 1)
            cv2.imwrite(label+".jpg", rgb_im)
            cv2.imshow("image", rgb_im)
            cv2.waitKey(0)
        elif mode == "noshow":
            return
        else:
            raise Exception

    def detect_all(self, im, mode):
        tot_rows, tot_cols = im.shape[:2]
        detected = []
        step_size = 2
        cls = self.model.classes_
        for i in xrange(0, tot_rows, step_size):
            for j in xrange(0, tot_cols, step_size):
                for label in symbol_label_parms.keys():
                    [rows, cols] = symbol_label_parms[label]
                    if i + rows >= tot_rows or j + cols >= tot_cols:
                        continue
                    sub_im = im[i:i + rows, j:j + cols]
                    sub_im = cv2.resize(sub_im, uni_size)
                    feature = self.extractor(sub_im)
                    feature = feature.reshape(-1, feature.size)
                    if self.model.predict(feature) == label:
                        detected.append((self.model.predict_proba(feature)[0][np.where(cls == label)], i, j, label))
                    # how to get comparable score here???
                    #temp_prob = self.model.predict_proba(feature)[0][np.where(cls == label)]

        #suppressed = [(i,j,label) for (prob,i,j,label) in detected if label is not "background"]
        ##Non-Maxima Suppression
        detected.sort(key=lambda tup: tup[0])
        hashed = np.zeros((tot_rows, tot_cols), dtype=bool)
        suppressed = []
        for (score, i, j, label) in detected:
            if label == "background" or hashed[i][j]:
                continue
            suppressed.append((i,j,label))
            print i,j,score
            hashed[max(0, i - default_symbol_rows / 2): min(i + default_symbol_rows / 2, tot_rows - 1),
            max(0, j - default_symbol_cols / 2): min(j + default_symbol_cols / 2, tot_cols - 1)] = True
            #[rows, cols] = symbol_label_parms[label] #which suppression region should I use? the symbol bbox or the default bbox?

        if mode == "show":
            rgb_im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            font = cv2.FONT_HERSHEY_SIMPLEX

            for (i, j, label) in suppressed:
                [rows, cols] = symbol_label_parms[label]
                rows = int(rows)
                cols = int(cols)
                cv2.rectangle(rgb_im, (j, i), (j + cols, i + rows), (0, 0, 255), 2)
                cv2.putText(rgb_im, label, (j, i - 10), font, 0.4, (0, 0, 255), 1)

            cv2.imwrite('detected.jpg', rgb_im)
            cv2.imshow("image", rgb_im)
            cv2.waitKey(0)
        elif mode == "noshow":
            return
        else:
            raise Exception