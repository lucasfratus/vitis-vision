import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(gray_img):
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    features = []

    for d in distances:
        for a in angles:
            glcm = graycomatrix(gray_img, 
                                distances=[d],
                                angles=[a],
                                symmetric=True,
                                normed=True)

            features.append(graycoprops(glcm, 'contrast')[0, 0])
            features.append(graycoprops(glcm, 'homogeneity')[0, 0])
            features.append(graycoprops(glcm, 'energy')[0, 0])
            features.append(graycoprops(glcm, 'correlation')[0, 0])

    return features
