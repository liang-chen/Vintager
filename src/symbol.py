
# symbol class

class LOC:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class BBox:
    def __init__(self, loc, rows, cols):
        self.loc = loc
        self.rows = rows
        self.cols = cols

class Symbol:
    def __init__(self, label, bbox):
        self.__label__ = label
        self.__bbox__ = bbox
    def getBBox(self):
        return self.__bbox__
    def getLabel(self):
        return self.__label__
    def getCenter(self):
        loc = self.__bbox__.loc
        loc.x += self.__bbox__.cols*0.5
        loc.y += self.__bbox__.rows*0.5
    def setBBox(self, center, rows, cols):
        loc = LOC(center.x - cols*0.5, center.y - rows*0.5);
        self.__bbox__.loc = loc
        self.__bbox__.rows = rows
        self.__bbox__.cols = cols
    def setLabel(self, label):
        self.__label__ = label