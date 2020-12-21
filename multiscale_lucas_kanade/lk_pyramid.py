import numpy as np
import scipy.ndimage
from .lucas_kanade import lucas_kanade
from .utils import downSample, upSample


def lucas_kanade_iterative(firstImage, secondImage, flow, N):
    """
    Lucas Kanade Iterative Optical flow estimation between firstImage and secondImage
    :param firstImage: First image
    :param secondImage: Second Image
    :param flow: Current estimate of optical flow
    :param N: Block size N x N
    :return: Refined optical flow
    """

    image_shape = firstImage.shape

    coarse_u = np.round(flow[0])
    coarse_v = np.round(flow[1])
    half_window_size = N // 2

    u = np.zeros(image_shape)
    v = np.zeros(image_shape)

    # Find Lucas Kanade OF for a block N x N with least squares solution
    for i in range(half_window_size, image_shape[0] - half_window_size):
        for j in range(half_window_size, image_shape[1] - half_window_size):
            firstImageSlice = firstImage[
                i - half_window_size : i + half_window_size + 1, j - half_window_size : j + half_window_size + 1
            ]

            # Find indices to warp second image
            lr = (i - half_window_size) + coarse_v[i, j]
            hr = (i + half_window_size) + coarse_v[i, j]
            lc = (j - half_window_size) + coarse_u[i, j]
            hc = (j + half_window_size) + coarse_u[i, j]

            # Find edge locations and choose possible window
            if lr < 0:
                lr = 0
                hr = N - 1

            if lc < 0:
                lc = 0
                hc = N - 1

            if hr > (len(firstImage[:, 0]) - 1):
                lr = len(firstImage[:, 0]) - N
                hr = len(firstImage[:, 0]) - 1

            if hc > (len(firstImage[0, :]) - 1):
                lc = len(firstImage[0, :]) - N
                hc = len(firstImage[0, :]) - 1

            if np.isnan(lr):
                lr = i - half_window_size
                hr = i + half_window_size

            if np.isnan(lc):
                lc = j - half_window_size
                hc = j + half_window_size

            secondImageSlice = secondImage[int(lr) : int(hr + 1), int(lc) : int(hc + 1)]

            # Refine optical flow
            uSlice, vSlice = lucas_kanade(firstImageSlice, secondImageSlice, N)
            u[i, j] = uSlice[half_window_size, half_window_size] + coarse_u[i, j]
            v[i, j] = vSlice[half_window_size, half_window_size] + coarse_v[i, j]

    return u, v


def lucas_kanade_pyramid(firstImage, secondImage, N, iteration, num_levels):
    """
    Lucas Kanade Pyramid Optical flow estimation between firstImage and secondImage
    :param firstImage: First image
    :param secondImage: Second Image
    :param N: Block size N x N
    :param iteration: Number of iterations for iterative LK
    :param num_levels: Number of levels of pyramid
    :return: Optical flow, Gradients
    """

    firstImage_reference = firstImage.copy()
    secondImage_reference = secondImage.copy()

    u_levels = []
    v_levels = []
    image_levels = []

    firstImage = np.array(firstImage)
    secondImage = np.array(secondImage)

    # Create pyramids by downsampling
    firstImagePyramid = np.empty((firstImage.shape[0], firstImage.shape[1], num_levels))
    secondImagePyramid = np.empty((secondImage.shape[0], secondImage.shape[1], num_levels))
    firstImagePyramid[:, :, 0] = firstImage
    secondImagePyramid[:, :, 0] = secondImage

    for level in range(1, num_levels):
        firstImage = downSample(firstImage)
        secondImage = downSample(secondImage)
        firstImagePyramid[0 : firstImage.shape[0], 0 : firstImage.shape[1], level] = firstImage
        secondImagePyramid[0 : secondImage.shape[0], 0 : secondImage.shape[1], level] = secondImage

    # Find optical flow at level0 and refine it with iterative Lucas Kande method
    level0 = num_levels - 1
    level0_scale = 2 ** level0
    firstImage_level0 = firstImagePyramid[
        0 : (len(firstImagePyramid[:, 0]) // level0_scale), 0 : (len(firstImagePyramid[0, :]) // level0_scale), level0
    ]
    secondImage_level0 = secondImagePyramid[
        0 : (len(secondImagePyramid[:, 0]) // level0_scale),
        0 : (len(secondImagePyramid[0, :]) // level0_scale),
        level0,
    ]
    (u, v) = lucas_kanade(firstImage_reference, secondImage_reference, N)

    for i in range(0, iteration):
        (u, v) = lucas_kanade_iterative(firstImage_level0, secondImage_level0, [u, v], N)

    u_levels.append(u.copy())
    v_levels.append(v.copy())
    image_levels.append(firstImage_level0.copy())

    # Find optical flow at all levels of pyramid
    for k in range(1, num_levels):
        upsampled_u = upSample(u)
        upsampled_v = upSample(v)
        levelk = num_levels - k - 1
        levelk_scale = 2 ** levelk
        firstImageIntermediate = firstImagePyramid[
            0 : (len(firstImagePyramid[:, 0]) // levelk_scale),
            0 : (len(firstImagePyramid[0, :]) // levelk_scale),
            levelk,
        ]
        secondImageIntermediate = secondImagePyramid[
            0 : (len(secondImagePyramid[:, 0]) // levelk_scale),
            0 : (len(secondImagePyramid[0, :]) // levelk_scale),
            levelk,
        ]
        (u, v) = lucas_kanade_iterative(firstImageIntermediate, secondImageIntermediate, [upsampled_u, upsampled_v], N)

        u_levels.append(u.copy())
        v_levels.append(v.copy())
        image_levels.append(firstImageIntermediate.copy())

    firstImage = firstImage_reference / 255
    secondImage = secondImage_reference / 255

    # Kernels for finding gradients Ix, Iy, It
    kernel_x = np.array([[-1, 1]])
    kernel_y = np.array([[-1], [1]])
    kernel_t = np.array([[1]])

    # kernel_x = np.array([[-1., 1.], [-1., 1.]]) / 4
    # kernel_y = np.array([[-1., -1.], [1., 1.]]) / 4
    # kernel_t = np.array([[1., 1.], [1., 1.]]) / 4

    Ix = scipy.ndimage.convolve(input=firstImage, weights=kernel_x, mode="nearest")
    Iy = scipy.ndimage.convolve(input=firstImage, weights=kernel_y, mode="nearest")
    It = scipy.ndimage.convolve(input=secondImage, weights=kernel_t, mode="nearest") + scipy.ndimage.convolve(
        input=firstImage, weights=-kernel_t, mode="nearest"
    )

    flow = [u_levels[-1], v_levels[-1]]
    I = [Ix, Iy, It]

    return flow, I
