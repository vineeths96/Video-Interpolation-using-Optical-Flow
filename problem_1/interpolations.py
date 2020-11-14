import cv2
import numpy as np


def warp_flow(firstImage, secondImage, forward_flow, backward_flow, image_ind, dataset):
    height, width = forward_flow.shape[:2]
    forward_flow = forward_flow / 2
    R2 = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))

    pixel_map = np.array(R2 + forward_flow, dtype=np.float32)
    forward_prediction = cv2.remap(firstImage, pixel_map[:, :, 0], pixel_map[:, :, 1], cv2.INTER_CUBIC)
    cv2.imwrite(f'./results/problem_1/interpolated_frames/{dataset}/forward_interpolated_{image_ind + 1}.png',
                forward_prediction)

    height, width = backward_flow.shape[:2]
    backward_flow = backward_flow / 2
    R2 = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))

    pixel_map = np.array(R2 + backward_flow, dtype=np.float32)
    backward_prediction = cv2.remap(secondImage, pixel_map[:, :, 0], pixel_map[:, :, 1], cv2.INTER_CUBIC)
    cv2.imwrite(f'./results/problem_1/interpolated_frames/{dataset}/backward_interpolated_{image_ind + 1}.png',
                backward_prediction)

    interpolated_frame = cv2.addWeighted(forward_prediction, 0.5, backward_prediction, 0.5, 0)
    cv2.imwrite(f'./results/problem_1/interpolated_frames/{dataset}/interpolated_{image_ind + 1}.png',
                interpolated_frame)
