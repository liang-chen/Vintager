
"""
Symbol Annotator
Display annotations from annotation file
"""

import cv2
import os
import random
from symbol import LOC, BBox
from utils import create_symbol_with_center_loc, create_symbol_with_bounding_box, get_sub_im
from globv import symbol_label_parms, symbol_label_list

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

    @staticmethod
    def is_in_loc_list(loc, loc_list):
        list = [[l.get_x(), l.get_y()] for l in loc_list]
        if [loc.get_x(), loc.get_y()] in list:
            return True
        else:
            return False

    def prepare_background_data(self, num):
        """
        Collect background data which doesn't overlap with annotations
        """
        (rows, cols) = self._image.shape
        i = 0
        data = []
        labels = []
        locs = [s.get_center() for s in self._symbols]
        while i < num:
            y = random.randint(0, rows)
            x = random.randint(0, cols)
            l = LOC(x, y)

            if not SymbolAnnotator.is_in_loc_list(l, locs):
                #randomize the size of background for better training
                [rows, cols] = symbol_label_parms[random.choice(symbol_label_list)]
                l_upper_left = LOC(l.get_x() - cols/2, l.get_y() - rows/2)
                bbox = BBox(l_upper_left, rows, cols);
                s = create_symbol_with_bounding_box("background", bbox)
                sub_im = get_sub_im(self._image, s)

                if sub_im is not None:
                    data.append(get_sub_im(self._image, s))
                    i += 1
                    labels.append("background")

        return data, labels

    def crop_and_save(self):
        """
        Crop annotated symbols and save them as jpg files
        """
        data_label_pair = [(get_sub_im(self._image, symbol), symbol.get_label()) for symbol in
                           self._symbols if symbol is not None]

        bk_data, bk_label = self.prepare_background_data(100)
        data_label_pair += [(d,l) for d,l in zip(bk_data, bk_label)]
        for (sub_img, label) in data_label_pair:
            if sub_img is None:
                continue
            dir = "../data/" + label + "/"
            if not os.path.exists(dir):
                os.makedirs(dir)

            cnt = len([f for f in os.listdir(dir) if f.endswith('.jpg') and os.path.isfile(os.path.join(dir, f))])
            cv2.imwrite(dir + str(cnt + 1) + ".jpg", sub_img)