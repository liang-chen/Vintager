
# main function, a quick testing

from pdfReader import PdfReader
from symbolDetector import SymbolDetector

if "__main__" == __name__:
    pr = PdfReader("/Users/Hipapa/Projects/Git/Vintager/data/train.pdf")
    try:
        pr.read()
    except Exception:
        print Exception

    sd = SymbolDetector(None)
