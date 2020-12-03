import cv2
import numpy as np
import scipy.interpolate


def warp_flow(firstImage, secondImage, forward_flow, If, backward_flow, Ib, image_ind, dataset):
    """
    Intermediate frame interpolation based on algorithm proposed in MiddleBury paper
    https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf
    :param firstImage: First image (Frame N)
    :param secondImage: Second image (Frame N+2)
    :param forward_flow: Optical flow from Frame N to Frame N+2
    :param If: Forward gradients [Ix, Iy, It]
    :param backward_flow: Optical flow from Frame N+2 to Frame N
    :param Ib: Backward gradients [Ix, Iy, It]
    :param image_ind: Current image index
    :param dataset: Dataset Name
    :return: None
    """

    height, width = firstImage.shape

    uf, vf = forward_flow
    Ix, Iy, It = If

    # Image is scaled because flow is calculated with scaled images
    forward_prediction = firstImage / 255 + uf * Ix + vf * Iy + It
    forward_prediction = forward_prediction * 255
    cv2.imwrite(f'./results/horn_schunck/interpolated_frames/{dataset}/forward_prediction_{image_ind + 1}.png',
                forward_prediction)

    ub, vb = backward_flow
    Ix, Iy, It = Ib

    # Image is scaled because flow is calculated with scaled images
    backward_prediction = secondImage / 255 + ub * Ix + vb * Iy + It
    backward_prediction = backward_prediction * 255
    cv2.imwrite(f'./results/horn_schunck/interpolated_frames/{dataset}/backward_prediction_{image_ind + 1}.png',
                backward_prediction)

    # interpolated_frame = cv2.addWeighted(forward_prediction, 0.5, backward_prediction, 0.5, 0)
    # cv2.imwrite(f'./results/horn_schunck/interpolated_frames/{dataset}/interpolated_{image_ind + 1}.png',
    #             interpolated_frame)

    # Intermediate frame t=0.5
    t = 0.5
    ut = np.full([uf.shape[0], uf.shape[1], 2], np.nan)

    # Occlusion detection
    occ_detect = True
    if occ_detect:
        similarity = np.full([height, width], np.inf)

    # Predict intermediate optical flow
    xx = np.broadcast_to(np.arange(width), (height, width))
    yy = np.broadcast_to(np.arange(height)[:, None], (height, width))

    xt = np.round(xx + t * uf)
    yt = np.round(yy + t * vf)

    for i in range(height):
        for j in range(width):
            i_ind_image = int(yt[i, j])
            j_ind_image = int(xt[i, j])

            if i_ind_image >= 0 and i_ind_image < height and j_ind_image >= 0 and j_ind_image < width:
                if occ_detect:
                    e = np.square(secondImage[i_ind_image, j_ind_image] - firstImage[i, j])
                    s = np.sum(e)

                    if s < similarity[i_ind_image, j_ind_image]:
                        ut[i_ind_image, j_ind_image, 0] = uf[i, j]
                        ut[i_ind_image, j_ind_image, 1] = vf[i, j]
                        similarity[i_ind_image, j_ind_image] = s
                else:
                    ut[i_ind_image, j_ind_image, 0] = uf[i, j]
                    ut[i_ind_image, j_ind_image, 1] = vf[i, j]

    uti = outside_in_fill(ut)

    # Occlusion masks
    occlusion_first = np.zeros_like(firstImage)
    occlusion_second = np.zeros_like(secondImage)

    occlusion_x1 = np.round(xx + uf).astype(np.int)
    occlusion_y1 = np.round(yy + vf).astype(np.int)

    occlusion_x1 = np.clip(occlusion_x1, 0, height - 1)
    occlusion_y1 = np.clip(occlusion_y1, 0, width - 1)

    for i in range(occlusion_first.shape[0]):
        for j in range(occlusion_first.shape[1]):
            if np.abs(uf[i, j] + ub[occlusion_x1[i, j], occlusion_y1[i, j]]) > 0.5:
                occlusion_first[i, j] = 1

            if np.abs(vf[i, j] + vb[occlusion_x1[i, j], occlusion_y1[i, j]]) > 0.5:
                occlusion_first[i, j] = 1

    occlusion_x1 = np.round(xx + ub).astype(np.int)
    occlusion_y1 = np.round(yy + vb).astype(np.int)

    occlusion_x1 = np.clip(occlusion_x1, 0, height - 1)
    occlusion_y1 = np.clip(occlusion_y1, 0, width - 1)

    for i in range(occlusion_second.shape[0]):
        for j in range(occlusion_second.shape[1]):
            if np.abs(ub[i, j] + uf[occlusion_x1[i, j], occlusion_y1[i, j]]) > 0.5:
                occlusion_second[i, j] = 1

            if np.abs(vb[i, j] + vf[occlusion_x1[i, j], occlusion_y1[i, j]]) > 0.5:
                occlusion_second[i, j] = 1

    # Intermediate image indices
    img0_for_x = xx - t * uti[:, :, 0]
    img0_for_y = yy - t * uti[:, :, 1]

    xt0 = np.clip(img0_for_x, 0, height - 1)
    yt0 = np.clip(img0_for_y, 0, width - 1)

    img1_for_x = xx + (1 - t) * uti[:, :, 0]
    img1_for_y = yy + (1 - t) * uti[:, :, 1]

    xt1 = np.clip(img1_for_x, 0, height - 1)
    yt1 = np.clip(img1_for_y, 0, width - 1)

    # Interpolate the images according to occlusion masks
    It = np.zeros(firstImage.shape)
    image1_interp = scipy.interpolate.RectBivariateSpline(np.arange(width), np.arange(height), firstImage.T)
    image2_interp = scipy.interpolate.RectBivariateSpline(np.arange(width), np.arange(height), secondImage.T)

    for i in range(It.shape[0]):
        for j in range(It.shape[1]):
            if not (occlusion_first[i, j] or occlusion_second[i, j]) or (
                    occlusion_first[i, j] and occlusion_second[i, j]):
                It[i, j] = t * image1_interp(xt0[i, j], yt0[i, j]) + (1 - t) * image2_interp(xt1[i, j], yt1[i, j])
            elif occlusion_first[i, j]:
                It[i, j] = image2_interp(xt1[i, j], yt1[i, j])
            elif occlusion_second[i, j]:
                It[i, j] = image1_interp(xt0[i, j], yt0[i, j])

    It = It.astype(np.int)
    cv2.imwrite(f'./results/horn_schunck/interpolated_frames/{dataset}/interpolated_{image_ind + 1}.png', It)


def outside_in_fill(image):
    """
    Outside in fill mentioned in paper
    :param image: Image matrix to be filled
    :return: output
    """

    rows, cols = image.shape[:2]

    col_start = 0
    col_end = cols
    row_start = 0
    row_end = rows
    lastValid = np.full([2], np.nan)

    while col_start < col_end or row_start < row_end:
        for c in range(col_start, col_end):
            if not np.isnan(image[row_start, c, 0]):
                lastValid = image[row_start, c, :]
            else:
                image[row_start, c, :] = lastValid

        for r in range(row_start, row_end):
            if not np.isnan(image[r, col_end - 1, 0]):
                lastValid = image[r, col_end - 1, :]
            else:
                image[r, col_end - 1, :] = lastValid

        for c in reversed(range(col_start, col_end)):
            if not np.isnan(image[row_end - 1, c, 0]):
                lastValid = image[row_end - 1, c, :]
            else:
                image[row_end - 1, c, :] = lastValid

        for r in reversed(range(row_start, row_end)):
            if not np.isnan(image[r, col_start, 0]):
                lastValid = image[r, col_start, :]
            else:
                image[r, col_start, :] = lastValid

        if col_start < col_end:
            col_start = col_start + 1
            col_end = col_end - 1

        if row_start < row_end:
            row_start = row_start + 1
            row_end = row_end - 1

    output = image

    return output
