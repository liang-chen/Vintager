
from pdfReader import PdfReader
from symbolDetector import SymbolDetector, DetectorOption
from symbolAnnotator import SymbolAnnotator
from train import train_svm, read_annotations
from globv import symbol_label_parms
import cv2
from cnn import train_cnn_var, detect_cnn

data_dir = '../data/'
anno_dir = '../annotations/'

train_cnn_var()

im = cv2.imread("../data/bass_clef/1.jpg", 0)
print detect_cnn(im)

exit(0)

####training
#train_svm(data_dir + "train0.jpg", anno_dir + "train.ant", "hog")

###display annotations
test_img = cv2.imread(data_dir + "train0.jpg", 0)
annotations = read_annotations(anno_dir + "train.ant")
sa = SymbolAnnotator(test_img, annotations)
sa.display()
sa.crop_and_save()
#
exit(0)

####read pdf (test data)
pr = PdfReader(data_dir + "test.pdf")
try:
    pr.read()
except Exception:
    print Exception

####detect symbols on test data
option = DetectorOption("hog_svm")
sd = SymbolDetector(option)
#sd.detect_all(pr.images[0][1:300, 1:300], "show")
sd.detect(pr.images[0][1:500,1:500], "flat", "show")
# for label in symbol_label_parms.keys():
# sd.detect(pr.images[0], label, "show"