
"""
Demo: Detect symbol via HOG-SVM detector
"""

from symbolDetector import DetectorOption, SymbolDetector
from pdfReader import PdfReader


if "__main__" == __name__:
    data_dir = '../data/'

    ####read pdf (test data)
    pr = PdfReader(data_dir + "test.pdf")

    option = DetectorOption("hog_svm")
    sd = SymbolDetector(option)
    sd.detect_all(pr.page(0)[100:500, 1:500], "show")