
import cv2
import numpy as np
from glob import uni_feature_len

def hog(img):
    bin_n = 16  # Number of bins
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing bin values in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    hist = hist/1000.0
    #hist.resize(uni_feature_len)
    #hist = np.squeeze(img.reshape(-1, img.size)/256.0) # test
    #print hist

    if uni_feature_len > len(hist):
        hist = np.pad(hist, (0, uni_feature_len - len(hist)), mode = 'constant', constant_values=0)
    return hist[:uni_feature_len]