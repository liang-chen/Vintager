
"""
PDF Reader
read multi-page PDF and convert each page into gray-level image
save each page as a cv2.image object
"""

import cv2


class PdfReader:
    """
    PdfReader Class
    read score in pdf format and convert pages into cv2.image
    """
    def __init__(self, path):
        """
        Initialize PdfReader object with PDF file path

        :param path: path to PDF file
        :type path: string
        """
        self._path = path
        self._images = []
        self._pages = 0
        self.read()

    def read(self):
        """
        Read PDF and perform PDF to image conversion, save each page as cv2.image into self.images list
        """
        # with Image(filename = self.path, resolution=180) as img:
        #    self.image = img
        #    print "image width = ", img.width
        #    print "image height = ", img.height
        #    display(img)

        pdf = file(self._path, "rb").read()
        startmark = "\xff\xd8"
        startfix = 0
        endmark = "\xff\xd9"
        endfix = 2
        i = 0

        njpg = 0
        while True:
            istream = pdf.find("stream", i)
            if istream < 0:
                break
            istart = pdf.find(startmark, istream, istream + 20)
            if istart < 0:
                i = istream + 20
                continue
            iend = pdf.find("endstream", istart)
            if iend < 0:
                raise Exception("Didn't find end of stream!")
            iend = pdf.find(endmark, iend - 20)
            if iend < 0:
                raise Exception("Didn't find end of JPG!")
            istart += startfix
            iend += endfix
            print "JPG %d from %d to %d" % (njpg, istart, iend)
            jpg = pdf[istart:iend]
            jpgfile = file(self._path[:-4]+"%d.jpg" % njpg, "wb")

            jpgfile.write(jpg)
            jpgfile.close()
            self._images.append(cv2.imread(self._path[:-4] + "%d.jpg" % njpg, 0))

            njpg += 1
            self._pages += 1
            i = iend

    def nPages(self):
        return self._pages

    def page(self, i):
        if i >= self._pages:
            return None
        return self._images[i]