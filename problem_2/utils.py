import cv2
import numpy as np
from .parameters import *


def downSample(matrix):
    """
    Downsample matrix/image with Gaussian smoothing
    :param matrix: Matrix/Image
    :return: Downsampled image
    """

    gaussian_matrix = cv2.GaussianBlur(matrix, KERNEL, SIGMA, SIGMA)
    downsampled_matrix = gaussian_matrix[::2, ::2]

    return downsampled_matrix


def upSample(matrix):
    """
    Upsample matrix/image with Gaussian smoothing
    :param matrix: Matrix/Image
    :return: Upsampled image
    """

    matrix_shape = matrix.shape
    matrix = 2 * matrix

    row_upsampled_matrix = np.zeros((matrix_shape[0], 2 * matrix_shape[1]))
    row_upsampled_matrix[:, ::2] = matrix

    matrix_shape = np.shape(row_upsampled_matrix)
    row_col_upsampled_matrix = np.zeros((2 * matrix_shape[0], matrix_shape[1]))
    row_col_upsampled_matrix[::2, :] = row_upsampled_matrix

    upsampled_matrix = cv2.GaussianBlur(row_col_upsampled_matrix, KERNEL, SIGMA, SIGMA)

    return upsampled_matrix
