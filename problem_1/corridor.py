import glob
import cv2
import numpy as np
from .lucas_kanade import lucas_kanade
from .interpolations import warp_flow


def corridor_interpolation(N=4):
    images = glob.glob('./input/corridor/*.pgm')
    images.sort(key=lambda f: int(re.sub('\D', '', f)))

    for ind in range(0, len(images) - 1, 2):
        firstImage = cv2.imread(images[ind], flags=cv2.IMREAD_UNCHANGED)
        secondImage = cv2.imread(images[ind + 2], flags=cv2.IMREAD_UNCHANGED)

        firstImage = np.array(firstImage, dtype=np.float32)
        secondImage = np.array(secondImage, dtype=np.float32)

        forward_flow = lucas_kanade(firstImage, secondImage, N, ind, 'corridor')
        backward_flow = lucas_kanade(secondImage, firstImage, N, ind, 'corridor')

        warp_flow(firstImage, secondImage, forward_flow, backward_flow, ind, 'corridor')
