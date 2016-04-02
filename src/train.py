
# script for symbol training

import cv2

def training(img_file_path, annotation_file_path, detector_name):
    im = cv2.imread(img_file_path)
    if detector_name == "hog":
        print "hog"