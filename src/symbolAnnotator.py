
"""
Symbol Annotator
Display annotations from annotation file
"""

import cv2
from utils import create_symbol_with_center_loc


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
        self.__symbols__ = []
        self.load_annotations(annotations)
        self.__image__ = im

    def load_annotations(self, annotations):
        """
        Load symbols from annotations

        :param annotations: annotations read by train.read_annotations
        :type annotations: dict containing symbol annotations
        """
        for label in annotations.keys():
            locs = annotations[label]
            sl = [create_symbol_with_center_loc(label, loc) for loc in locs]
            self.__symbols__ += [s for s in sl if s is not None]

    def display(self):
        """
        Display annotations on the image
        """
        img = cv2.cvtColor(self.__image__, cv2.COLOR_GRAY2RGB)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for s in self.__symbols__:
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