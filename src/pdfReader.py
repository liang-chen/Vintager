
# convert pdf to pixel image
# dependent on the installation of ImageMagick and wand binding in Python

#from wand.image import Image
#from wand.display import display
import sys

class pdfReader:
    def __init__(self, path):
        self.path = path
        self.image = None

    def read(self):
        #with Image(filename = self.path, resolution=180) as img:
        #    self.image = img
        #    print "image width = ", img.width
        #    print "image height = ", img.height
        #    display(img)

        pdf = file(self.path, "rb").read()
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
            #print "JPG %d from %d to %d" % (njpg, istart, iend)
            jpg = pdf[istart:iend]
            jpgfile = file(self.path[:-4]+"%d.jpg" % njpg, "wb")
            jpgfile.write(jpg)
            jpgfile.close()

            njpg += 1
            i = iend