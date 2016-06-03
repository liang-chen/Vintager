
"""
Generate training set
"""

from symbolAnnotator import SymbolAnnotator
from train import read_annotations
import cv2

data_dir = '../data/'
anno_dir = '../annotations/'
with open(data_dir+"train.txt", "r") as file:
    for line in file:
        sub = line.strip()
        filename = data_dir + sub
        train_img = cv2.imread(filename, 0)
        filename_split = sub.split('/')
        filestub = filename_split[1][:-4]
        annotations = read_annotations(anno_dir + filestub + ".bop")
        sa = SymbolAnnotator(train_img, annotations)
        sa.crop_and_save()