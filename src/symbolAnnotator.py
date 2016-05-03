
"""
Symbol Annotator
Display annotations from annotation file
"""

import cv2
import os
import glob
from utils import create_symbol_with_center_loc, get_sub_im, unify_img_size


class SymbolAnnotator:
    """
    Symbol Annotator Class
    """
    def __init__(self,im,annotations):
        """
        Initialize a symbol annotator

        :param im: input image
        :type im: cv2.image
        :param annotations: annotations read by train.read_annotations
        :type annotations: dict containing symbol annotations
        """
        self._symbols = []
        self.load_annotations(annotations)
        self._image = im

    def load_annotations(self, annotations):
        """
        Load symbols from annotations

        :param annotations: annotations read by train.read_annotations
        :type annotations: dict containing symbol annotations
        """
        for label in annotations.keys():
            locs = annotations[label]
            sl = [create_symbol_with_center_loc(label, loc) for loc in locs]
            self._symbols += [s for s in sl if s is not None]

    def display(self):
        """
        Display annotations on the image
        """
        img = cv2.cvtColor(self._image, cv2.COLOR_GRAY2RGB)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for s in self._symbols:
            label = s.get_label()
            bbox = s.get_bbox()
            rows = bbox.get_rows()
            cols = bbox.get_cols()
            i = bbox.get_loc().get_y()
            j = bbox.get_loc().get_x()
            cv2.rectangle(img, (j, i), (j + cols, i + rows), (0, 0, 255), 2)
            cv2.putText(img, label, (j, i - 10), font, 0.4, (0, 0, 255), 1)

        cv2.imwrite("annotations.jpg", img)
        cv2.imshow("annotations", img)
        cv2.waitKey(0)

    def crop_and_save(self):
        """
        Crop annotated symbols and save as jpg files

        :return:
        :rtype:
        """

        data_label_pair = [(get_sub_im(self._image, symbol), symbol.get_label()) for symbol in
                           self._symbols if symbol is not None]

        for (sub_img, label) in data_label_pair:
            if sub_img is None:
                continue
            dir = "../data/" + label + "/"
            if not os.path.exists(dir):
                os.makedirs(dir)

            resized_img = unify_img_size(sub_img)
            cnt = len([f for f in os.listdir(dir) if f.endswith('.jpg') and os.path.isfile(os.path.join(dir, f))])
            cv2.imwrite(dir + str(cnt + 1) + ".jpg", resized_img)