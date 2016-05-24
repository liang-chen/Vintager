
"""
Demo: Train symbol detector via HOG + linear-SVM
"""


from train import train_svm


if "__main__" == __name__:
    data_dir = '../data/'
    anno_dir = '../annotations/'
    train_svm(data_dir + "train.jpg", anno_dir + "train.bop", "hog")