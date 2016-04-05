
# detect various types of music symbols via different detectors
# 1, HOG + Muticlass-SVM
# 2, AutoEncoder

import cv2
import numpy as np

class DetectorOption:
    def __init__(self, name):
        self.name = name


class SymbolDetector:
    def __init__(self, option):
        try:
            if option.name == "hog_svm":
                with open(option.name + ".dat", 'rb') as f:
                    svm = f.read()
                f.close()
                print svm

        except Exception:
            print Exception