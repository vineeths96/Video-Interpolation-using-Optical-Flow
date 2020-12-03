import numpy as np
import scipy.signal
import scipy.ndimage
import matplotlib.pyplot as plt


def lucas_kanade(firstImage, secondImage, N, image_ind=None, dataset=None, tau=1e-3):
    """
    Lucas Kanade Optical flow estimation between firstImage and secondImage
    :param firstImage: First image
    :param secondImage: Second Image
    :param N: Block size N x N
    :param image_ind: Current image index
    :param dataset: Dataset name
    :param tau: Threshold parameter
    :return: Optical flow, Gradients
    """

    firstImage = firstImage / 255
    secondImage = secondImage / 255
    image_shape = firstImage.shape
    half_window_size = N // 2

    # Kernels for finding gradients Ix, Iy, It
    kernel_x = np.array([[-1, 1]])
    kernel_y = np.array([[-1], [1]])
    kernel_t = np.array([[1]])

    # kernel_x = np.array([[-1., 1.], [-1., 1.]]) / 4
    # kernel_y = np.array([[-1., -1.], [1., 1.]]) / 4
    # kernel_t = np.array([[1., 1.], [1., 1.]]) / 4

    Ix = scipy.ndimage.convolve(input=firstImage, weights=kernel_x, mode='nearest')
    Iy = scipy.ndimage.convolve(input=firstImage, weights=kernel_y, mode='nearest')
    It = scipy.ndimage.convolve(input=secondImage, weights=kernel_t, mode='nearest') + scipy.ndimage.convolve(
        input=firstImage, weights=-kernel_t, mode='nearest')

    u = np.zeros(image_shape)
    v = np.zeros(image_shape)

    # Find Lucas Kanade OF for a block N x N with least squares solution
    for row_ind in range(half_window_size, image_shape[0] - half_window_size):
        for col_ind in range(half_window_size, image_shape[1] - half_window_size):
            Ix_windowed = Ix[row_ind - half_window_size: row_ind + half_window_size + 1,
                          col_ind - half_window_size: col_ind + half_window_size + 1].flatten()
            Iy_windowed = Iy[row_ind - half_window_size: row_ind + half_window_size + 1,
                          col_ind - half_window_size: col_ind + half_window_size + 1].flatten()
            It_windowed = It[row_ind - half_window_size: row_ind + half_window_size + 1,
                          col_ind - half_window_size: col_ind + half_window_size + 1].flatten()

            A = np.asarray([Ix_windowed, Iy_windowed]).reshape(-1, 2)
            b = np.asarray(It_windowed)

            A_transpose_A = np.transpose(A) @ A

            A_transpose_A_eig_vals, _ = np.linalg.eig(A_transpose_A)
            A_transpose_A_min_eig_val = np.min(A_transpose_A_eig_vals)

            # Noise thresholding
            if A_transpose_A_min_eig_val < tau:
                continue

            A_transpose_A_PINV = np.linalg.pinv(A_transpose_A)
            w = A_transpose_A_PINV @ np.transpose(A) @ b

            u[row_ind, col_ind], v[row_ind, col_ind] = w + (1, 1)

    # Plot, visualize and save the optical flow
    # flow_map = compute_flow_map(u, v, 8)
    # plt.imshow(firstImage * 255, cmap='gray')
    # plt.imshow(flow_map, cmap=None)

    # firstImage_denormalized = (firstImage * 255).astype(np.float32)
    # added_image = cv2.addWeighted(firstImage_denormalized, 0.5, flow_map, 1, 0, dtype=cv2.CV_32F)
    # cv2.imwrite(f'./results/multiscale_lucas_kanade/optical_flow/{dataset}/flow_map_{image_ind}.png', added_image)
    # vis_optic_flow_arrows(firstImage, [u, v],
    #                       f'./results/multiscale_lucas_kanade/optical_flow/{dataset}/flow_map_{image_ind}.png')
    plt.show()

    flow = [u, v]

    return flow


def compute_flow_map(u, v, gran=8):
    """
    Plot optical flow map
    """

    flow_map = np.zeros(u.shape)

    for y in range(flow_map.shape[0]):
        for x in range(flow_map.shape[1]):

            if y % gran == 0 and x % gran == 0:
                dx = 2 * int(u[y, x])
                dy = 2 * int(v[y, x])

                if dx > 0 or dy > 0:
                    cv2.arrowedLine(flow_map, (x, y), (x + dx, y + dy), 255, 1)

    return flow_map


def vis_optic_flow_arrows(img, flow, filename, show=True):
    """
    Visualize optical flow on image
    """

    x = np.arange(0, img.shape[1], 1)
    y = np.arange(0, img.shape[0], 1)
    x, y = np.meshgrid(x, y)
    u, v = flow

    plt.figure()
    fig = plt.imshow(img, cmap='gray', interpolation='bicubic')

    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    step = img.shape[0] // 50

    plt.quiver(x[::step, ::step], y[::step, ::step], u[::step, ::step], v[::step, ::step], color='r', pivot='middle',
               headwidth=2, headlength=3)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    if show:
        plt.show()
