import cv2

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Erro ao carregar a imagem: {path}")
    return img

def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def resize(img, size=(256, 256)):
    return cv2.resize(img, size)
