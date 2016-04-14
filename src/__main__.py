
# main function, a quick testing

from pdfReader import PdfReader
from symbolDetector import SymbolDetector, DetectorOption
from symbolAnnotator import SymbolAnnotator
from train import training, read_annotations
from glob import symbol_label_parms

if "__main__" == __name__:

    data_dir = '../data/'
    anno_dir = '../annotations/'

    ####training
    #training(data_dir + "train0.jpg", anno_dir + "train.ant", "hog")

    ####display annotations
    # pr = PdfReader(data_dir + "train.pdf")
    # try:
    #     pr.read()
    # except Exception:
    #     print Exception
    # annotations = read_annotations(anno_dir + "train.ant")
    # sa = SymbolAnnotator(pr.images[0], annotations)
    # sa.display()
    #
    # exit(0)

    ####read pdf (test data)
    pr = PdfReader(data_dir + "test.pdf")
    try:
        pr.read()
    except Exception:
        print Exception

    ####detect symbols on test data
    option = DetectorOption("hog_svm")
    sd = SymbolDetector(option)
    sd.detect_all(pr.images[0][1:500,1:500],"show")
    #sd.detect(pr.images[0][1:500,1:500], "treble_clef", "show")
    #for label in symbol_label_parms.keys():
     #   sd.detect(pr.images[0], label, "show")