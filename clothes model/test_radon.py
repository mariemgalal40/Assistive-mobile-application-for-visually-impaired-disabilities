import cv2
import numpy as np
from skimage.transform import radon, iradon

def radon_fun(imgs):
    image = cv2.imread(imgs,1)
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = np.int8(image)
    image = np.double(image)

    sobelx = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobely = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)

    gradient_image = np.hypot(sobelx, sobely)
    img = np.int8(gradient_image)

    # create radon
    theta = np.linspace(0, 180, 180)
    vars = np.zeros(theta.shape)
    for i in range(len(theta)):
        # len -> 128 that equal no of rows , shape-> (128,1)
        # each angle crate 128 value
        sinogram = radon(img,theta = [i])
        sig = sinogram.flatten()
        vars[i] = np.var(sig)

    # normalization of vars
    var_min, var_max = min(vars), max(vars)
    for i, val in enumerate(vars):
        vars[i] = (val-var_min) / (var_max-var_min)
    # print(vars)
    return vars