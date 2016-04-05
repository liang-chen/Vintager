
# main function, a quick testing

from pdfReader import PdfReader
from symbolDetector import SymbolDetector
from train import training

if "__main__" == __name__:
    pr = PdfReader("/Users/Hipapa/Projects/Git/Vintager/data/train.pdf")
    try:
        pr.read()
    except Exception:
        print Exception

    training("/Users/Hipapa/Projects/Git/Vintager/data/train0.jpg", "/Users/Hipapa/Projects/Git/Vintager/annotations/train.ant", "hog")
    #    sd = SymbolDetector(None)
