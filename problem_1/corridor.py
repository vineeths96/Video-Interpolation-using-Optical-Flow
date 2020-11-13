import glob
import cv2

from .lucas_kanade import lucas_kanade


def corridor_interpolation():
    images = glob.glob('./input/corridor/*.pgm')
    images.sort()

    for ind in range(len(images) - 1):
        firstImage = cv2.imread(images[ind], flags=cv2.IMREAD_UNCHANGED)
        secondImage = cv2.imread(images[ind + 1], flags=cv2.IMREAD_UNCHANGED)

        lucas_kanade(firstImage, secondImage, 11, ind, 'corridor')
