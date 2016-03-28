
from wand.image import Image

class pdfReader:
    def __init__(self, path):
        self.path = path
        self.image = None

    def read(self):
        with Image(filename = self.path) as img:
            print "image width = ", img.width
            print "image height = ", img.height