
# main function, a quick testing

from pdfReader import PdfReader
from symbolDetector import SymbolDetector, DetectorOption
from train import training

if "__main__" == __name__:

    data_dir = '../data/'
    anno_dir = '../annotations/'

    training(data_dir + "train0.jpg", anno_dir + "train.ant", "hog")
    pr = PdfReader(data_dir + "test.pdf")
    try:
        pr.read()
    except Exception:
        print Exception


    option = DetectorOption("hog_svm")
    sd = SymbolDetector(option)
    #sd.detect(pr.images[0], "solid_note_head", "show")