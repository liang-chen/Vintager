
# symbol class


class LOC:
    def __init__(self, x, y):
        self.__x__ = int(x)
        self.__y__ = int(y)

    def get_x(self):
        return self.__x__

    def get_y(self):
        return self.__y__

    def set_x(self, x):
        self.__x__ = int(x)

    def set_y(self, y):
        self.__y__ = int(y)

    def copy(self):
        loc = LOC(self.__x__, self.__y__)
        return loc


class BBox:
    def __init__(self, loc, rows, cols):
        self.__loc__ = loc
        self.__rows__ = int(rows)
        self.__cols__ = int(cols)

    def get_loc(self):
        return self.__loc__

    def get_rows(self):
        return self.__rows__

    def get_cols(self):
        return self.__cols__

    def set_loc(self, loc):
        if not isinstance(loc, LOC):
            raise TypeError("%s attribute must be set to an instance of %s" % ("loc", "LOC"))
        self.__loc__ = loc

    def set_rows(self, rows):
        self.__rows__ = int(rows)

    def set_cols(self, cols):
        self.__cols__ = int(cols)


class Symbol:
    def __init__(self, label, bbox):
        self.__label__ = label
        self.__bbox__ = bbox

    def get_bbox(self):
        return self.__bbox__

    def get_label(self):
        return self.__label__

    def get_center(self):
        loc = self.__bbox__.loc.copy()
        loc.set_x(loc.get_x() + self.__bbox__.cols*0.5)
        loc.set_y(loc.get_y() + self.__bbox__.rows * 0.5)
        return loc

    def set_bbox(self, center, rows, cols):
        loc = LOC(center.get_x() - cols*0.5, center.get_y() - rows*0.5)
        if self.__bbox__ is None:
            self.__bbox__ = BBox(loc, rows, cols)
        else:
            self.__bbox__.set_loc(loc)
            self.__bbox__.set_rows(rows)
            self.__bbox__.set_cols(cols)

    def set_label(self, label):
        self.__label__ = label
