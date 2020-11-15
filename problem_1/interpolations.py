import cv2
import numpy as np
import scipy.interpolate


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
    cv2.imwrite(f'./results/problem_1/interpolated_frames/{dataset}/forward_prediction_{image_ind + 1}.png',
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
    cv2.imwrite(f'./results/problem_1/interpolated_frames/{dataset}/backward_prediction_{image_ind + 1}.png',
                backward_prediction)

    interpolated_frame = cv2.addWeighted(forward_prediction, 0.5, backward_prediction, 0.5, 0)
    interpolated_frame = interpolated_frame * 255
    cv2.imwrite(f'./results/problem_1/interpolated_frames/{dataset}/interpolated_{image_ind + 1}.png',
                interpolated_frame)

    # trial
    ut = np.full(forward_flow.shape, np.nan)
    occ_detect = True
    t = 0.5
    if occ_detect:
        similarity = np.full([height, width], np.inf)

    xx = np.broadcast_to(np.arange(width), (height, width))
    yy = np.broadcast_to(np.arange(height)[:, None], (height, width))

    xt = np.round(xx + t * forward_flow[:, :, 0])
    yt = np.round(yy + t * forward_flow[:, :, 1])

    for j in range(height):
        for i in range(width):
            j1 = int(yt[j, i])
            i1 = int(xt[j, i])

            if i1 >= 0 and i1 < width and j1 >= 0 and j1 < height:
                if occ_detect:
                    e = np.square(secondImage[j1, i1] - firstImage[j, i])
                    s = np.sum(e)

                    if s < similarity[j1, i1]:
                        ut[j1, i1, :] = forward_flow[j, i, :]
                        similarity[j1, i1] = s
                else:
                    ut[j1, i1, :] = forward_flow[j, i, :]

    uti = outside_in_fill(ut)

    img0_for_x = xx - t * uti[:, :, 0]
    img0_for_y = yy - t * uti[:, :, 1]

    xt0 = np.clip(img0_for_x, 0, height - 1)
    yt0 = np.clip(img0_for_y, 0, width - 1)

    img1_for_x = xx + (1 - t) * uti[:, :, 0]
    img1_for_y = yy + (1 - t) * uti[:, :, 1]

    xt1 = np.clip(img1_for_x, 0, height - 1)
    yt1 = np.clip(img1_for_y, 0, width - 1)

    It = np.zeros(firstImage.shape)
    ip1 = scipy.interpolate.RectBivariateSpline(np.arange(width), np.arange(height), firstImage.T)
    ip2 = scipy.interpolate.RectBivariateSpline(np.arange(width), np.arange(height), secondImage.T)

    for i in range(It.shape[0]):
        for j in range(It.shape[1]):
            It[i, j] = (1 - t) * ip1(xt0[i, j], yt0[i, j]) + t * ip2(xt1[i, j], yt1[i, j])

    cv2.imwrite(f'./results/problem_1/interpolated_frames/{dataset}/interpolated_{image_ind + 1}.png',
                It)


def outside_in_fill(image):
    rows, cols = image.shape[:2]

    cstart = 0
    cend = cols
    rstart = 0
    rend = rows
    lastValid = np.full([2], np.nan)

    while cstart < cend or rstart < rend:
        for c in range(cstart, cend):
            if not np.isnan(image[rstart, c, 0]):
                lastValid = image[rstart, c, :]
            else:
                image[rstart, c, :] = lastValid

        for r in range(rstart, rend):
            if not np.isnan(image[r, cend - 1, 0]):
                lastValid = image[r, cend - 1, :]
            else:
                image[r, cend - 1, :] = lastValid

        for c in reversed(range(cstart, cend)):
            if not np.isnan(image[rend - 1, c, 0]):
                lastValid = image[rend - 1, c, :]
            else:
                image[rend - 1, c, :] = lastValid

        for r in reversed(range(rstart, rend)):
            if not np.isnan(image[r, cstart, 0]):
                lastValid = image[r, cstart, :]
            else:
                image[r, cstart, :] = lastValid

        if cstart < cend:
            cstart = cstart + 1
            cend = cend - 1

        if rstart < rend:
            rstart = rstart + 1
            rend = rend - 1
    output = image

    return output
