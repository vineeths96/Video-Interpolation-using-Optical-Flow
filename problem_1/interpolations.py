import cv2
import numpy as np


def warp_flow(firstImage, secondImage, forward_flow, If, backward_flow, Ib, image_ind, dataset):
    height, width = forward_flow.shape[:2]
    forward_prediction = np.zeros([height, width])
    forward_flow = forward_flow
    Ix, Iy, It = If

    for i in range(height):
        for j in range(width):
            forward_prediction[i, j] = firstImage[i, j] / 255 + forward_flow[i, j, 0] * Ix[i, j] + forward_flow[
                i, j, 1] * Iy[i, j] + It[i, j]

    forward_prediction = forward_prediction * 255
    cv2.imwrite(f'../results/problem_1/interpolated_frames/{dataset}/forward_prediction_{image_ind + 1}.png',
                forward_prediction)

    height, width = backward_flow.shape[:2]
    backward_prediction = np.zeros([height, width])
    backward_flow = backward_flow
    Ix, Iy, It = Ib

    for i in range(height):
        for j in range(width):
            backward_prediction[i, j] = secondImage[i, j] / 255 + backward_flow[i, j, 0] * Ix[i, j] + backward_flow[
                i, j, 1] * Iy[i, j] + It[i, j]

    backward_prediction = backward_prediction * 255
    cv2.imwrite(f'../results/problem_1/interpolated_frames/{dataset}/backward_prediction_{image_ind + 1}.png',
                backward_prediction)

    interpolated_frame = cv2.addWeighted(forward_prediction, 0.5, backward_prediction, 0.5, 0)
    interpolated_frame = interpolated_frame * 255
    cv2.imwrite(f'../results/problem_1/interpolated_frames/{dataset}/interpolated_{image_ind + 1}.png',
                interpolated_frame)

    # trial
    R2 = np.dstack(np.meshgrid(np.arange(width), np.arange(height))[::-1])

