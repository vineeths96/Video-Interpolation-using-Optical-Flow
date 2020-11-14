import glob
import cv2
import numpy as np
from .lucas_kanade import lucas_kanade
from .interpolations import warp_flow


def corridor_interpolation(N=11):
    images = glob.glob('./input/corridor/*.pgm')
    images.sort()

    forward_flows = []
    backward_flows = []

    for ind in range(0, len(images) - 1, 2):
        firstImage = cv2.imread(images[ind], flags=cv2.IMREAD_UNCHANGED)
        secondImage = cv2.imread(images[ind + 2], flags=cv2.IMREAD_UNCHANGED)

        forward_flow = lucas_kanade(firstImage, secondImage, N, ind, 'corridor')
        forward_flows.append(forward_flow)

        warp_flow(np.array(firstImage, dtype=np.float32), forward_flow, ind, 'corridor')
        continue

        backward_flow_u, backward_flow_v = lucas_kanade(secondImage, firstImage, N, ind, 'corridor')
        backward_flows_u.append(backward_flow_u)
        backward_flows_v.append(backward_flows_v)




