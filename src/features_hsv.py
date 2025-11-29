import cv2
import numpy as np

def extract_hsv_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    
    features = []

    # médias e desvios
    for channel in (H, S, V):
        features.append(np.mean(channel))
        features.append(np.std(channel))

    # histogramas (16 bins cada)
    for channel in (H, S, V):
        hist = cv2.calcHist([channel], [0], None, [16], [0, 256])
        hist = hist.flatten()
        features.extend(hist)

    return features
