

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
        self._x = int(x)
        self._y = int(y)

    def get_x(self):
        """

        :return: x-coordinate
        :rtype: int
        """
        return self._x

    def get_y(self):
        """

        :return: y-coordinate
        :rtype: int
        """
        return self._y

    def set_x(self, x):
        """
        set x-coordinate

        :param x: new x-coordinate
        :type x: int
        """
        self._x = int(x)

    def set_y(self, y):
        """
        set y-coordinate

        :param y: new y-coordinate
        :type y: int
        """
        self._y = int(y)

    def copy(self):
        """
        create a new LOC instance from the existing one

        :return: loc
        :rtype: LOC
        """
        loc = LOC(self._x, self._y)
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
        self._loc = loc
        self._rows = int(rows)
        self._cols = int(cols)

    def get_loc(self):
        """

        :return: the upper left corner of this bounding box
        :rtype: LOC
        """
        return self._loc

    def get_rows(self):
        """

        :return: number of rows
        :rtype: int
        """
        return self._rows

    def get_cols(self):
        """

        :return: number of columns
        :rtype: int
        """
        return self._cols

    def set_loc(self, loc):
        """
        set upper left corner

        :param loc: new location
        :type loc: LOC
        """
        if not isinstance(loc, LOC):
            raise TypeError("%s attribute must be set to an instance of %s" % ("loc", "LOC"))
        self._loc = loc

    def set_rows(self, rows):
        """
        set rows

        :param rows: new number of rows
        :type rows: int
        """
        self._rows = int(rows)

    def set_cols(self, cols):
        """
        set columns

        :param cols: new number of columns
        :type cols: int
        """
        self._cols = int(cols)


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
        self._label = label
        self._bbox = bbox

    def get_bbox(self):
        """
        get the bounding box parameters

        :return: bounding box
        :rtype: BBox
        """
        return self._bbox

    def get_label(self):
        """
        get the name of the symbol

        :return: label
        :rtype: string
        """
        return self._label

    def get_center(self):
        """
        get the center of the symbol

        :return: center location
        :rtype: LOC
        """
        loc = self._bbox.get_loc().copy()
        loc.set_x(loc.get_x() + self._bbox.get_cols()*0.5)
        loc.set_y(loc.get_y() + self._bbox.get_rows() * 0.5)
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
        if self._bbox is None:
            self._bbox = BBox(loc, rows, cols)
        else:
            self._bbox.set_loc(loc)
            self._bbox.set_rows(rows)
            self._bbox.set_cols(cols)

    def set_label(self, label):
        """
        set the name of this symbol

        :param label: label
        :type label: string
        """
        self._label = label
