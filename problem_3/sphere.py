import glob
import cv2
import numpy as np
import regex as re
from .horn_schunk import horn_schunk
from .interpolations import warp_flow
from .parameters import *


def sphere_interpolation():
    images = glob.glob('./input/sphere/*.ppm')
    images.sort(key=lambda f: int(re.sub('\D', '', f)))

    for ind in range(0, len(images) - 2, 2):
        firstImage = cv2.imread(images[ind], flags=cv2.IMREAD_GRAYSCALE)
        secondImage = cv2.imread(images[ind + 2], flags=cv2.IMREAD_GRAYSCALE)

        firstImage = np.array(firstImage, dtype=np.float32)
        secondImage = np.array(secondImage, dtype=np.float32)

        forward_flow, If = horn_schunk(firstImage, secondImage, LAMBADA, ITERS, ind, 'sphere')
        backward_flow, Ib = horn_schunk(secondImage, firstImage, LAMBADA, ITERS, ind, 'sphere')

        warp_flow(firstImage, secondImage, forward_flow, If, backward_flow, Ib, ind, 'sphere')