from pdfReader import PdfReader


if "__main__" == __name__:
    pr = PdfReader("/Users/Hipapa/Projects/Git/Vintager/data/train.pdf")
    try:
        pr.read()
    except Exception:
        print Exception
