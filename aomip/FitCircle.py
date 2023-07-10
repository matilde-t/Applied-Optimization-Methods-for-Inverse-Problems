# import numpy as np
# from PIL import Image, ImageDraw


# def box(img, border):
#     img[0:border, :] = 0
#     img[-border:, :] = 0
#     img[:, 0:border] = 0
#     img[:, -border:] = 0
#     return img.flatten()


import cv2
import numpy as np

def fitCircle(img, radius):
    # read image
    hh, ww = img.shape[:2]
    hh2 = hh // 2
    ww2 = ww // 2

    # define circles
    yc = hh2
    xc = ww2

    # draw filled circle in white on black background as mask
    mask = np.zeros_like(img)
    mask = cv2.circle(mask, (xc,yc), radius, 1, -1)

    # apply mask to image
    img[mask==0] = 0

    return img.flatten()

