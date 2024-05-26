
import numpy as np
import scipy.optimize as optimize

import kernels as rd_kernels


def gauss_func(x, a, b, c):
    """
    Evaluates a basic gaussian function
    :param x: The x values at which to evaluate the function
    :param a: The amplitude of the gaussian
    :param b: The offset of the gaussian
    :param c: The width of the guassian (fwhm)
    :return:
    """
    return a * np.exp(-(x - b) ** 2.0 / (2 * c ** 2))


def gauss_func2(x, sigma, mu):
    """
    Evaluates a basic gaussian function
    :param x: The x values at which to evaluate the function
    :param sigma: The standard deviation of the gaussian
    :param mu: The mean of the gaussian
    :return:
    """
    return 1.0 / (sigma * np.sqrt(2 * np.pi)) * \
           np.exp(-1 * (x - mu) ** 2 / (2 * sigma ** 2))


def image_noise_and_mean(image, image_range=[0, 1000], reject_high=False):
    """
    Computes the sigma and mean of an image chip based. S1 version...
    on the histogram of the pixel intensities
    :param image: The 2D image chip
    :param reject_high: Whether or not to remove outliers prior to computation
    :return: noise, mean
    """
    image2 = image.flatten().copy()
    if reject_high:
        idx = np.where((image2 < (np.median(image2) + np.std(image2) * 3)))[0]
    else:
        idx = np.arange(len(image2))
    image_mean = image2[idx].mean()
    image_std = image2[idx].std()
    h, e = np.histogram(image2[idx],
                        bins=int(np.sqrt(len(idx))),
                        density=True, range=image_range)
    try:
        popt, pcov = optimize.curve_fit(gauss_func2, e[1:], h, p0=[image_std,
                                                                  image_mean])
    except Exception as e:
        print('Fit did not go well. Returning numpy std and mean.')
        return image2[idx].std(), image2[idx].mean()
    return popt[0], popt[1]


def get_image_stats(image, reject_high=True):
    """
    Computes the sigma and mean of an image chip based
    on the histogram of the pixel intensities
    :param image: The 2D image chip
    :param reject_high: Whether or not to remove outliers prior to computation
    :return: noise, mean
    """
    image2 = image.flatten().copy()
    if reject_high:
        idx = np.where((image2 < (np.median(image2) + np.std(image2) * 3)))[0]
    else:
        idx = np.arange(len(image2))
    image_mean = image2[idx].mean()
    image_std = image2[idx].std()
    h, e = np.histogram(image2[idx],
                        bins=int(np.sqrt(len(idx))),
                        density=True)
    try:
        popt, pcov = optimize.curve_fit(gauss_func, e[1:], h, p0=[image_std,
                                                                  image_mean])
    except Exception as e:
        print('Fit did not go well. Returning numpy std and mean.')
        return image2[idx].std(), image2[idx].mean()
    # In the future can add iterative rejection of extremely
    # high and low value pixels to close in on a better solution
    sigma = popt[2] / 2.355
    return sigma, popt[1] + image_mean


def pearson(x):
    """
    Computes a vectorized version of the pearson correlation coefficient for
    when you're comparing multiple signals to one another
    :param x: Signals variable (signal_length, num_signals)
    :return:
    """
    x2 = np.repeat(x[..., None], x.shape[1], axis=-1)
    x2_mean = x2.mean(0)
    num1 = np.sum((x2 - x2_mean) * (x2.transpose((0, 2, 1)) - x2_mean.T),
                  axis=0)
    denom1 = np.sqrt(np.sum((x2 - x2_mean) ** 2, axis=0))
    denom2 = np.sqrt(np.sum((x2.transpose((0, 2, 1)) - x2_mean.T) ** 2, axis=0))
    r_vals = num1 / (denom1 * denom2)
    return r_vals


def imaginary_distance(x):
    """
    Computes a vectorized version of the pearson correlation coefficient for
    when you're comparing multiple signals to one another
    :param x: Signals variable (signal_length, num_signals)
    :return:
    """
    x2 = np.repeat(x[..., None], x.shape[1], axis=-1)
    x2_mean = x2.mean(0)
    num1 = np.sum((x2 - x2_mean) * np.conj(x2.transpose((0, 2, 1)) - x2_mean.T),
                  axis=0)
    denom1 = np.sqrt(np.sum((x2 - x2_mean) ** 2, axis=0))
    denom2 = np.sqrt(np.sum((x2.transpose((0, 2, 1)) - x2_mean.T) ** 2, axis=0))
    r_vals = num1 / (denom1 * denom2)
    return r_vals


def circular_correlation(circ_vec):
    """
    Performs circular correlation of the circular vectors
    Based on Jammalamadaka and SenGupta (2001)
    :param circ_vec: An N x 2 vector of angles to correlate
    :return:
    """
    mean_a = np.arctan2(np.sum(np.sin(circ_vec[:, 0])),
                        np.sum(np.cos(circ_vec[:, 0])))
    mean_b = np.arctan2(np.sum(np.sin(circ_vec[:, 1])),
                        np.sum(np.cos(circ_vec[:, 1])))
    val1 = circ_vec[:, 0] - mean_a
    val2 = circ_vec[:, 1] - mean_b
    num1 = np.sum(np.sin(val1) * np.sin(val2))
    denom = np.sqrt(np.sum(np.sin(val1) ** 2) * np.sum(np.sin(val2) ** 2))
    r_vals = num1 / denom
    return r_vals
