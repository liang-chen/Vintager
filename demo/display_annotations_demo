
"""
Demo: display annotations
"""

from symbolAnnotator import SymbolAnnotator
from train import read_annotations
import cv2

data_dir = '../data/'
anno_dir = '../annotations/'

test_img = cv2.imread(data_dir + "train.jpg", 0)
annotations = read_annotations(anno_dir + "train.bop")
sa = SymbolAnnotator(test_img, annotations)
sa.display()

#crop symbols and save to disk (optional)
#sa.crop_and_save()