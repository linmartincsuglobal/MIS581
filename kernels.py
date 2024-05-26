
import numpy as np
import scipy.misc as misc


def gaussian_2d(shape=(3, 3), sigma=0.5):
    """
    Generates a 2d gaussian filter
    :param shape:
    :param sigma:
    :return:
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def circular(xcenter, ycenter, width, radius):
    """
    Generates a circular kernel
    :param xcenter: The center x position for the kernel
    :param ycenter: The center y position for the kernel
    :param width: The width of the kernel
    :return:
    """
    y, x = np.ogrid[-ycenter:width - ycenter,
                    -xcenter:width - xcenter]
    mask = x * x + y * y <= (radius * radius)
    return mask


def annulus(kern_size):
    """
    Computes an annular kernel
    :param kern_size: The kernel size in pixels
    :return:
    """
    k1 = circular(kern_size / 2, kern_size / 2, kern_size, kern_size / 2)
    k2 = circular(kern_size / 2, kern_size / 2, kern_size, kern_size / 2 - 1)
    return k1 - k2


def motion_blur(length, angle):
    kernel = np.zeros((length + 2, length + 2))
    kernel[(length + 2) // 2, 1:-1] = 255
    kernel = misc.imrotate(kernel, angle)
    return kernel
