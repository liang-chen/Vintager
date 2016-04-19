

class LOC:
    """
    LOC class
    defines a location on a 2D plane
    """
    def __init__(self, x, y):
        """

        :param x: x-coordinate
        :type x: int
        :param y: y-coordinate
        :type y: int
        """
        self.__x__ = int(x)
        self.__y__ = int(y)

    def get_x(self):
        """

        :return: x-coordinate
        :rtype: int
        """
        return self.__x__

    def get_y(self):
        """

        :return: y-coordinate
        :rtype: int
        """
        return self.__y__

    def set_x(self, x):
        """
        set x-coordinate

        :param x: new x-coordinate
        :type x: int
        """
        self.__x__ = int(x)

    def set_y(self, y):
        """
        set y-coordinate

        :param y: new y-coordinate
        :type y: int
        """
        self.__y__ = int(y)

    def copy(self):
        """
        create a new LOC instance from the existing one

        :return: loc
        :rtype: LOC
        """
        loc = LOC(self.__x__, self.__y__)
        return loc


class BBox:
    """
    BBox Class
    creates bounding box for a certain symbol
    """
    def __init__(self, loc, rows, cols):
        """

        :param loc: upper-left corner
        :type loc: LOC
        :param rows: number of rows
        :type rows: int
        :param cols: number of columns
        :type cols: int
        """
        self.__loc__ = loc
        self.__rows__ = int(rows)
        self.__cols__ = int(cols)

    def get_loc(self):
        """

        :return: the upper left corner of this bounding box
        :rtype: LOC
        """
        return self.__loc__

    def get_rows(self):
        """

        :return: number of rows
        :rtype: int
        """
        return self.__rows__

    def get_cols(self):
        """

        :return: number of columns
        :rtype: int
        """
        return self.__cols__

    def set_loc(self, loc):
        """
        set upper left corner

        :param loc: new location
        :type loc: LOC
        """
        if not isinstance(loc, LOC):
            raise TypeError("%s attribute must be set to an instance of %s" % ("loc", "LOC"))
        self.__loc__ = loc

    def set_rows(self, rows):
        """
        set rows

        :param rows: new number of rows
        :type rows: int
        """
        self.__rows__ = int(rows)

    def set_cols(self, cols):
        """
        set columns

        :param cols: new number of columns
        :type cols: int
        """
        self.__cols__ = int(cols)


class Symbol:
    """
    Symbol Class
    creates a symbol object
    """
    def __init__(self, label, bbox):
        """

        :param label: name of the symbol
        :type label: string
        :param bbox: bounding box of the symbol
        :type bbox: BBox
        """
        self.__label__ = label
        self.__bbox__ = bbox

    def get_bbox(self):
        """
        get the bounding box parameters

        :return: bounding box
        :rtype: BBox
        """
        return self.__bbox__

    def get_label(self):
        """
        get the name of the symbol

        :return: label
        :rtype: string
        """
        return self.__label__

    def get_center(self):
        """
        get the center of the symbol

        :return: center location
        :rtype: LOC
        """
        loc = self.__bbox__.loc.copy()
        loc.set_x(loc.get_x() + self.__bbox__.cols*0.5)
        loc.set_y(loc.get_y() + self.__bbox__.rows * 0.5)
        return loc

    def set_bbox(self, center, rows, cols):
        """
        set the bounding box for this symbol

        :param center: center location of the symbol
        :type center: LOC
        :param rows: number of rows
        :type rows: int
        :param cols: number of columns
        :type cols: int
        """
        loc = LOC(center.get_x() - cols*0.5, center.get_y() - rows*0.5)
        if self.__bbox__ is None:
            self.__bbox__ = BBox(loc, rows, cols)
        else:
            self.__bbox__.set_loc(loc)
            self.__bbox__.set_rows(rows)
            self.__bbox__.set_cols(cols)

    def set_label(self, label):
        """
        set the name of this symbol

        :param label: label
        :type label: string
        """
        self.__label__ = label
