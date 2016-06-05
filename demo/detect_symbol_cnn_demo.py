
"""
Demo: Detect symbol via convolutional neural network
"""

from symbolDetector import DetectorOption, SymbolDetector
from pdfReader import PdfReader


if "__main__" == __name__:
    data_dir = '../data/'

    ####read pdf (test data)
    pr = PdfReader(data_dir + "test.pdf")

    option = DetectorOption("cnn")
    sd = SymbolDetector(option)
    sd.detect(pr.page(0)[0:500, 0:500], "treble_clef", "show")