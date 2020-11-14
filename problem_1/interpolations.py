import cv2
import numpy as np


def warp_flow(firstImage, flow, image_ind, dataset):
    height, width = flow.shape[:2]
    R2 = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))

    pixel_map = np.array(R2 + flow, dtype=np.float32)#.astype(np.float32)
    res = cv2.remap(firstImage, pixel_map[:, :, 0], pixel_map[:, :, 1], cv2.INTER_LINEAR)

    cv2.imwrite(f'./results/problem_1/interpolated_frames/{dataset}/flow_map_{image_ind+1}.png', res)

    return res
