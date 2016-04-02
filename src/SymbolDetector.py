
# detect various types of music symbols via different detectors
# 1, HOG + muticlass-SVM
# 2, AutoEncoder

import cv2

class SymbolDetector:
    def __init__(self, option):
        print "test"
        try:
            im = cv2.imread("/Users/Hipapa/Projects/Git/Vintager/data/train0.jpg")
            cv2.imshow('image', im)
            cv2.waitKey(0)
        except Exception:
            print Exception
