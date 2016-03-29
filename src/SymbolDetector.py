
# detect various types of music symbols via different detectors
# 1, HOG + muticlass-SVM
# 2, AutoEncoder

import cv2

class SymbolDetector:
    def __init__(self, option):
        print "detector"
