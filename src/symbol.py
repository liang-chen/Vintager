
# symbol class


class LOC:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)


class BBox:
    def __init__(self, loc, rows, cols):
        self.loc = loc
        self.rows = int(rows)
        self.cols = int(cols)


class Symbol:
    def __init__(self, label, bbox):
        self.__label__ = label
        self.__bbox__ = bbox

    def get_bbox(self):
        return self.__bbox__

    def get_label(self):
        return self.__label__

    def get_center(self):
        loc = self.__bbox__.loc
        loc.x += int(self.__bbox__.cols*0.5)
        loc.y += int(self.__bbox__.rows*0.5)

    def set_bbox(self, center, rows, cols):
        loc = LOC(center.x - cols*0.5, center.y - rows*0.5)
        if self.__bbox__ is None:
            self.__bbox__ = BBox(loc, rows, cols)
        else:
            self.__bbox__.loc = loc
            self.__bbox__.rows = int(rows)
            self.__bbox__.cols = int(cols)

    def set_label(self, label):
        self.__label__ = label
