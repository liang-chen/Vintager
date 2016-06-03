
"""
Demo: Detect symbol via convolutional neural network
"""

from symbolDetector import DetectorOption, SymbolDetector
from pdfReader import PdfReader
import cv2

if "__main__" == __name__:
    data_dir = '../data/'

    ####read pdf (test data)
    pr = PdfReader(data_dir + "test.pdf")

    option = DetectorOption("cnn")
    sd = SymbolDetector(option)
    #cv2.imshow("haha", pr.page(0)[180:280, 20:300])
    #cv2.waitKey(0)
    sd.detect_all(pr.page(0)[180:280, 20:300], "show")