import glob
import cv2

from .lucas_kanade import lucas_kanade


def sphere_interpolation():
    images = glob.glob('./input/sphere/*.ppm')
    images.sort()

    for ind in range(len(images) - 1):
        firstImage = cv2.imread(images[ind], flags=cv2.IMREAD_GRAYSCALE)
        secondImage = cv2.imread(images[ind + 1], flags=cv2.IMREAD_GRAYSCALE)

        lucas_kanade(firstImage, secondImage, 11, ind, 'sphere')
