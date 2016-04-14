
# annotate symbols with boundingboxes

import cv2
from utils import create_symbol


class SymbolAnnotator:
    def __init__(self,im,annotations):
        self.__symbols__ = []
        self.load_symbols(annotations)
        self.__image__ = im

    def load_symbols(self, annotations):
        for label in annotations.keys():
            locs = annotations[label]
            sl = [create_symbol(label, loc) for loc in locs]
            self.__symbols__ += [s for s in sl if s is not None]

    def display(self):
        img = cv2.cvtColor(self.__image__, cv2.COLOR_GRAY2RGB)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for s in self.__symbols__:
            label = s.get_label()
            bbox = s.get_bbox()
            rows = bbox.rows
            cols = bbox.cols
            i = bbox.loc.y
            j = bbox.loc.x
            cv2.rectangle(img, (j, i), (j + cols, i + rows), (0, 0, 255), 2)
            cv2.putText(img, label, (j, i - 10), font, 0.4, (0, 0, 255), 1)

        cv2.imwrite("annotations.jpg", img)
        cv2.imshow("annotations", img)
        cv2.waitKey(0)
