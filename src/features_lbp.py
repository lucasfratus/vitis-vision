import numpy as np
from skimage.feature import local_binary_pattern

def extract_lbp_features(gray_img, P=16, R=2):
    lbp = local_binary_pattern(gray_img, P, R, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, P+3), range=(0, P+2))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist.tolist()
