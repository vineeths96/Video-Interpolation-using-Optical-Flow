import glob
import cv2
import numpy as np
import regex as re
from .lk_pyramid import lucas_kanade_pyramid
from .interpolations import warp_flow
from .parameters import *


def sphere_interpolation(N=5):
    """
    Sphere dataset interpolation of Frame N+1 from Frame N and Frame N+2
    :param N: Block size N x N
    :return: None
    """

    images = glob.glob("./input/sphere/*.ppm")
    images.sort(key=lambda f: int(re.sub("\D", "", f)))

    for ind in range(0, len(images) - 2, 2):
        firstImage = cv2.imread(images[ind], flags=cv2.IMREAD_GRAYSCALE)
        secondImage = cv2.imread(images[ind + 2], flags=cv2.IMREAD_GRAYSCALE)

        firstImage = np.array(firstImage, dtype=np.float32)
        secondImage = np.array(secondImage, dtype=np.float32)

        forward_flow, If = lucas_kanade_pyramid(firstImage, secondImage, N, ITERATION, LEVEL)
        backward_flow, Ib = lucas_kanade_pyramid(secondImage, firstImage, N, ITERATION, LEVEL)

        warp_flow(firstImage, secondImage, forward_flow, If, backward_flow, Ib, ind, "sphere")
