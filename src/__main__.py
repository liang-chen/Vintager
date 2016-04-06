
# main function, a quick testing

from pdfReader import PdfReader
from symbolDetector import SymbolDetector, DetectorOption
from train import training

if "__main__" == __name__:
    pr = PdfReader("/Users/Hipapa/Projects/Git/Vintager/data/test.pdf")
    try:
        pr.read()
    except Exception:
        print Exception

    training("/Users/Hipapa/Projects/Git/Vintager/data/train0.jpg", "/Users/Hipapa/Projects/Git/Vintager/annotations/train.ant", "hog")
    #option = DetectorOption("hog_svm")
    #sd = SymbolDetector(option)
