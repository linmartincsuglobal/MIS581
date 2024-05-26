
import os

try:
    import cv2
except ImportError as e:
    print('Could not load OpenCV. Some processing will be limited.')
import numpy as np
# import osgeo.gdal as gdal
import scipy.misc as misc
import scipy.ndimage as ndimage
import scipy.optimize as optimize
import scipy.spatial.distance as distance
import skimage.filters as filters

from PIL import Image, ImageDraw, ImageFont

import stats as rd_stats


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def compute_gain_offset(image1, image2, image_mask=None, remove_outliers=True):
    if image_mask is None:
        mask = (image1 != 0) & (image2 != 0)
    else:
        mask = (image1 != 0) & (image2 != 0) & image_mask
    cs = np.polyfit(image2[mask], image1[mask], 1)
    if remove_outliers:
        diff = (image1[mask] - (image2[mask] * cs[0] + cs[1])) / image1[mask]
        mask2 = np.abs(diff) < 0.5
        cs = np.polyfit(image2[mask][mask2], image1[mask][mask2], 1)
    return cs


def create_superoverlay_rgb(image, out_dir, tile_size=256):
    sz = image.shape
    divs = sz[0] / tile_size
    nlevels = int(np.log2(divs) + 1)
    print('Number of levels: ', nlevels)
    for level in range(nlevels):
        print(level)
        skip = 2 ** level
        ts = tile_size * skip
        nxs = sz[1] / ts
        nys = sz[0] / ts
        for iy in range(nys):
            for ix in range(nxs):
                xs = ix * ts
                ys = iy * ts
                xe = xs + ts
                ye = ys + ts
                chip = image[ys:ye:skip, xs:xe:skip, :]
                misc.imsave(os.path.join(out_dir, str(( nlevels - 1) - level) +
                                         '-' + str(iy) + '-' + str(ix) +
                                         '.png'),
                            chip)


def percent_stretch(img, lower_percent=1, higher_percent=99):
    a = 0
    b = 255
    c = np.percentile(img, lower_percent)
    d = np.percentile(img, higher_percent)
    if (d - c) == 0:
        return img
    img2 = img.copy().astype('float32')
    img2 = a + (img2 - c) * (b - a) / (d - c)
    img2[img2 < a] = a
    img2[img2 > b] = b
    return img2


def stretch_8bit(bands, lower_percent=1, higher_percent=99, mask=None):
    if mask is None:
        mask = np.ones_like(bands[:, :, 0])
    out = np.zeros_like(bands)
    for i in range(3):
        a = 0
        b = 255
        c = np.percentile(bands[:, :, i][mask], lower_percent)
        d = np.percentile(bands[:, :, i][mask], higher_percent)
        if d == c:
            return bands
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    return out.astype(np.uint8)


# Adjusts colors along a given
def PCA_color_adjust(data, center=0, stddev=0.1):
    """
    Shifts each pixel value of an image by [p1,p2,...pn][a1v1,a2v2...anvn]^T
    where n is the number of color bands, [p1,p2,...pn] are the eigenvectors
    of the covariance matrix or pixel values, l1...ln the eigenvalues and
    a1...an random normal variables
    :param data: The input image, can be a two or three dimensional nd-array or
            a string. If data is a stringit is assumed to be a file path to
            the desired image. Reading in the image will only be successful
            if it is a format recognized by scipy.misc.imread()
    :param center: The center of the normal distribution for a1...an, default 0
    :param stddev: The standard deviation for a1...an, default is 0.1

    """
    # if image is a string assume it is a file path
    if isinstance(data, str):
        data = misc.imread(data)

    raw = data.copy()
    image = data.copy()
    im_size = image.shape
    image = np.divide(image.astype(float), np.max(image).astype(float))

    # get number of bands, reshape and find mean of each band
    if image.ndim == 2:
        bands = 1
        image = np.reshape(image, (im_size[0] * im_size[1], bands))
        m = np.mean(image)
    else:
        bands = im_size[2]
        image = np.reshape(image, (im_size[0] * im_size[1], bands))
        m = np.mean(image, 0)

        # normalize
    image = image - np.tile(m, (im_size[0] * im_size[1], 1))

    # find eigenvalues and eigenvectors of the dot product
    w, v = np.linalg.eig(np.dot(np.transpose(image), image))

    # find amount of shift
    dw = np.sqrt(w)
    amount = np.random.normal(center, stddev,
                              (bands, 1)) * np.reshape(dw, (bands, 1))

    # derive shift by going amount in the direction of the eigenvectors
    shift = np.dot(v, amount)

    # add shift to raw data
    image = raw + np.transpose(shift)

    return image


def pixels_between_points(points):
    pixels = []
    dx = 1.0
    for ix in range(len(points) - 1):
        pt = points[ix].copy()
        rpt = np.round(pt)
        pt2 = points[ix + 1]
        rise = float(pt2[1] - pt[1])
        run = float(pt2[0] - pt[0])
        d = np.sqrt(rise ** 2 + run ** 2)
        curr_pixels = []
        count = 0
        while (rpt[0] != pt2[0]) & (rpt[1] != pt2[1]):
            pt1 = pt
            pt1[0] += (run / d)
            pt1[1] += (rise / d)
            pt = pt1.copy()
            rpt = np.round(pt1)
            curr_pixels.append(rpt)
            count += 1
        pixels.extend(curr_pixels)
    return pixels


def bytescale(data, cmin=None, cmax=None):
    if cmin is None:
        cmin = data.min()

    if cmax is None:
        cmax = data.max()

    if cmax == cmin:
        return np.zeros_like(data, dtype='uint8')

    data2 = (data.astype('float32').copy() - cmin) / (cmax - cmin)
    data2[data2 < 0] = 0
    data2[data2 > 1] = 1

    return (data2 * 255).astype('uint8')


def sharpen(img, sigma_blur, alpha):
    """

    :param img:
    :param sigma_blur:
    :param alpha:
    :return:
    """
    sharp_img = img.copy().astype('float32')
    for ix in range(3):
        tmp = img[:, :, ix].astype('float32') / 255.0
        blur_img = filters.gaussian(tmp, sigma=sigma_blur)
        sharp_img[:, :, ix] = tmp + alpha * (tmp - blur_img)
    return sharp_img


def draw_rectangles(img, bboxes, width=1, fill=None, outline='#ffffff'):
    """
    Annotates an image with rectangles
    :param img: A numpy array with all color bands if color is desired,
    include alpha if you want to have a transparent fill color
    :param bboxes: The rectangle coords. List of [[x0,y0,x1,y1],...]
    :param width: The width in pixels of the outline.
    :param fill: The color of the fill. Gray, RGB, RGBA depending on
    application. None indicates no fill.
    :param outline: The color of the outline. Gray, RGB, RGBA depending on
    application. None indicates no outline.
    :return:
    """
    if isinstance(img, str):
        image = Image.open(img)
    elif isinstance(img, Image.Image):
        image = img.copy()
    else:
        image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle(bbox, fill=fill, outline=outline, width=width)
    return image


def draw_lines(img, xys, width=1, fill='#ff0000'):
    """
    Annotates an image with lines
    :param img: A numpy array with all color bands if color is desired,
    include alpha if you want to have a transparent fill color
    :param xys: The coordinates for the lines. List of lists [[x0,y0,x1,y1,...],...]
    :param width: The width in pixels of the outline.
    :param fill: The color of the fill. Gray, RGB, RGBA depending on
    application. None indicates no fill.
    :return:
    """
    if isinstance(img, str):
        image = Image.open(img)
    elif isinstance(img, Image.Image):
        image = img.copy()
    else:
        image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    for xy in xys:
        draw.line(xy, fill=fill, width=width)
    return image


def annotate(img, text, xy, font_name='arial.ttf', font_size=24,
             color='#ffffff', x_center=False):
    """
    Annotates an image with text
    :param img: A numpy array with all color bands if color is desired,
    include alpha if you want to have a transparent fill color
    :param text: The string that is desired to display
    :param xy: The XY position of the text in the image space. X will be
    overridden to the center point if x_center is set to True.
    :param font_name: The name of the font to use
    :param font_size: The size of the font
    :param color: The color of the font. Gray, RGB, RGBA depending on
    application.
    :param x_center: Set to true to center on X, otherwise it will put it at
    position X defined by the XY parameter.
    :return:
    """
    if isinstance(img, str):
        image = Image.open(img)
    elif isinstance(img, Image.Image):
        image = img.copy()
    else:
        image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_name, font_size, encoding='unic')
    w, h = draw.textsize(text, font=font)
    if x_center:
        xy[0] = (image.size[0] - w) // 2
    draw.text(xy, text, fill=color, font=font)
    return image


def draw_polygon(img, coords, outline=(0, 255, 255, 255), fill=(0, 0, 0, 0)):
    """
    Draws a polygon given by the coordinates
    :param img: A numpy array with all color bands if color is desired,
    include alpha if you want to have a transparent fill color
    :param coords: The coordinates of the polygon in pixel space. The last
    coordinate should equal the first coordinate.
    :param outline: The outline color. Gray, RGB, or RGBA depending on
    application.
    :param fill: The fill color. Gray, RGB, or RGBA depending on
    application.
    :return:
    """
    if isinstance(img, str):
        image = Image.open(img)
    else:
        image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    if isinstance(coords, list):
        for coord in coords:
            draw.polygon(coord.flatten().tolist(), fill=fill, outline=outline)
    else:
        draw.polygon(coords, fill=fill, outline=outline)
    return image


def alignment_matrix(img1, img2, warp_mode=0):

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations,
                termination_eps)

    (cc, warp_matrix) = cv2.findTransformECC(img1, img2, warp_matrix,
                                             warp_mode, criteria, None, 5)

    return cc, warp_matrix


def color_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def get_edges(img_gray):
    img_mask = img_gray > 0
    img_mask = ndimage.binary_erosion(img_mask, iterations=8)
    img_gray = filters.sobel(img_gray)
    img_gray[~img_mask] = 0
    img_gray = img_gray.astype('float32')
    return img_gray


def warp_image(img, warp_matrix):
    sz = img.shape
    warp_flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP

    if len(sz) == 3:
        img_aligned = np.zeros_like(img)
        for ix in range(sz[2]):
            img_aligned[:, :, ix] = cv2.warpAffine(img[:, :, ix],
                                                   warp_matrix, (sz[1], sz[0]),
                                                   flags=warp_flags)
    else:
        img_aligned = cv2.warpAffine(img, warp_matrix, (sz[1], sz[0]),
                                     flags=warp_flags)

    return img_aligned


def align_images(img1, img2, warp_mode=0, return_cc=False):
    # Find size of image1
    sz = img1.shape

    if len(sz) == 3:
        img1_gray = color_to_gray(img1)
        img2_gray = color_to_gray(img2)
    else:
        img1_gray = img1.copy()
        img2_gray = img2.copy()

    # img1_gray = get_edges(img1_gray)
    # img2_gray = get_edges(img2_gray)

    cc, warp_matrix = alignment_matrix(img1_gray, img2_gray,
                                       warp_mode=warp_mode)

    warp_flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP

    if len(sz) == 3:
        img2_aligned = np.zeros_like(img2)
        for ix in range(sz[2]):
            img2_aligned[:, :, ix] = cv2.warpAffine(img2[:, :, ix],
                                                    warp_matrix, (sz[1], sz[0]),
                                                    flags=warp_flags)
    else:
        img2_aligned = cv2.warpAffine(img2, warp_matrix, (sz[1], sz[0]),
                                      flags=warp_flags)

    if return_cc:
        return img2_aligned, cc

    return img2_aligned


def image_sharpness_metric(image, method='laplace'):
    if len(image.shape) > 2:
        gray_image = color_to_gray(image)
    else:
        gray_image = image.copy()

    mask = gray_image > 0
    mask = ndimage.binary_erosion(mask, iterations=3)

    nels = float(np.product(gray_image.shape))

    if method == 'laplace':
        gray_edges = filters.laplace(gray_image)
        h, e = np.histogram(gray_edges[mask], bins=100, range=[-0.5, 0.5])

        # Get estimates for all of these values
        fwhm_val = np.std(h / nels) * 2.355
        offset_val = e[np.argmax(h)]
        amp_val = np.max(h)

        p0 = [amp_val / nels, offset_val, fwhm_val]

        popt, pcov = optimize.curve_fit(rd_stats.gauss_func, e[1:], h / nels,
                                        p0=p0)
        metric = popt[-1]
    elif method == 'gradient':
        rolly = np.diff(gray_image, 1, 0)
        rollx = np.diff(gray_image, 1, 1)
        gray_edges = rollx[:-1, :] ** 2 + rolly[:, :-1] ** 2
        metric = np.mean(gray_edges[mask[:-1, :-1]])
    else:
        metric = None

    return metric


def chip_gray(image, center, width, height, enforce_size=False):
    x_buf = width // 2
    y_buf = height // 2
    x_start = center[0] - x_buf
    x_end = center[0] + x_buf + 1
    y_start = center[1] - y_buf
    y_end = center[1] + y_buf + 1
    y_pad = [0] * 2
    x_pad = [0] * 2
    if y_start < 0:
        y_pad[0] = np.abs(0 - y_start)
        y_start = 0
    if y_end > image.shape[0]:
        y_pad[1] = y_end - image.shape[0]
        y_end = image.shape[0]
    if x_start < 0:
        x_pad[0] = np.abs(0 - x_start)
        x_start = 0
    if x_end > image.shape[1]:
        x_pad[1] = x_end - image.shape[1]
        x_end = image.shape[1]
    image_chip = image[y_start:y_end, x_start:x_end, ...]
    if enforce_size:
        if len(image_chip.shape) == 3:
            image_chip = np.pad(image_chip, (y_pad, x_pad, (0, 0)), mode='constant',
                                constant_values=0)
        else:
            image_chip = np.pad(image_chip, (y_pad, x_pad), mode='constant',
                                constant_values=0)
    return image_chip


def chip_gray_bounds(image, x_start, y_start, x_end, y_end, enforce_size=False):
    y_pad = [0] * 2
    x_pad = [0] * 2
    if y_start < 0:
        y_pad[0] = np.abs(y_start)
        y_start = 0
    if y_end > image.shape[0]:
        y_pad[1] = y_end - image.shape[0]
        y_end = image.shape[0]
    if x_start < 0:
        x_pad[0] = np.abs(x_start)
        x_start = 0
    if x_end > image.shape[1]:
        x_pad[1] = x_end - image.shape[1]
        x_end = image.shape[1]
    image_chip = image[y_start:y_end, x_start:x_end, ...]
    if enforce_size:
        image_chip = np.pad(image_chip, (y_pad, x_pad), mode='constant',
                            constant_values=0)
    return image_chip


def get_focus_metrics(image, x_grid=3, y_grid=3, chip_size=256,
                      method='laplace'):
    """
    Gets focus metric values for a grayscale image
    :param image: The grayscale image to process
    :param x_grid: The number of regions to process across the x direction
    :param y_grid: the number of regions to process across the y direction
    :return:
    """
    x_grid_size = image.shape[1] // x_grid
    y_grid_size = image.shape[0] // y_grid
    x_buf = (x_grid_size - chip_size) // 2
    y_buf = (y_grid_size - chip_size) // 2
    y_grid_metrics = []
    for iy in range(y_grid):
        x_grid_metrics = []
        for ix in range(x_grid):
            x_start = ix * x_grid_size + x_buf
            y_start = iy * y_grid_size + y_buf
            chip = image[y_start:y_start + 256, x_start:x_start + 256]
            sharp_metric = image_sharpness_metric(chip, method=method)
            x_grid_metrics.append(np.abs(sharp_metric))
        y_grid_metrics.append(x_grid_metrics)
    return np.array(y_grid_metrics)


def dog(image, noise_filter=0.7, blur_filter=2.0):
    """
    Difference of Gaussians background removal
    :param image:
    :param noise_filter:
    :param blur_filter:
    :return:
    """
    img1 = filters.gaussian(image, noise_filter)
    img2 = filters.gaussian(image, blur_filter)
    dog_image = img1 - img2
    dog_image /= dog_image.max()
    dog_image *= image.max()
    return dog_image


def add_noise(image, noise_type='gaussian', sigma=1.0, amplitude=1.0):
    """
    Adds noise to the image in the form of gaussian or uniform
    :param image: The image to be augmented
    :param noise_type: The type of noise (gaussian or uniform)
    :param sigma: The standard deviation for gaussian noise, does nothing for
    uniform noise
    :param amplitude: The amplitude of the resulting noise
    :return:
    """
    new_image = image.copy()
    if type == 'gaussian':
        noise_img = np.random.normal(scale=sigma, size=new_image.shape)
    elif type == 'uniform':
        noise_img = np.random.uniform(0, 1.0, new_image.shape)
    else:
        raise ValueError('Unrecognized noise function: %s.' % noise_type)
    noise_img += noise_img.min()
    noise_img /= noise_img.max()
    noise_img *= amplitude
    new_image += noise_img
    return new_image


def mirror(image, vertical=False, horizontal=False):
    """
    Flips the image along the vertical or horizontal axes
    :param image: The image to be augmented
    :param vertical: Whether to flip the image vertically
    :param horizontal: Whether to flip the image horizontally
    :return:
    """
    new_image = image.copy()
    if vertical:
        new_image = np.flipud(new_image)
    if horizontal:
        new_image = np.fliplr(new_image)
    return new_image


def intensity_gain_offset(image, min_offset, max_offset, min_gain, max_gain,
                          gain_value=None, offset_value=None):
    """
    Applies an intensity gain and offset to the image, if the gain and offset
    values are not supplied explicitly then random ones are generated
    :param image: The image to be augmented
    :param min_offset: The minimum offset value when a random value is desired
    :param max_offset: The maximum offset value when a random value is desired
    :param min_gain: The minimum gain value when a random value is desired
    :param max_gain: The maximum gain value when a random value is desired
    :param gain_value: The explicit gain value to use
    :param offset_value: The explicit offset value to use
    :return:
    """
    new_image = image.astype('float32').copy()
    if gain_value is None:
        gain_value = np.random.uniform(min_gain, max_gain)
    if offset_value is None:
        offset_value = np.random.uniform(min_offset, max_offset)
    new_image = new_image * gain_value + offset_value
    return new_image


def negate_image(image):
    """
    Multiplies the image by -1, amazosaurs
    :param image: The image to be augmented
    :return:
    """
    return image * -1


def dropout(image, percent=0.1):
    """
    Randomly masks out (sets to zero) pixels in the image based on a percent
    :param image: The image to be augmented
    :param percent: The percent of pixels to be set to zero
    :return:
    """
    new_image = image.copy()
    drop_mask = np.random.random(image.shape) <= percent
    new_image[drop_mask] = 0
    return new_image


def augment(image, params):
    """
    Augments an image based on a parameters dictionary. Parameters dictionary
    can be generated by using the augment_params method.
    :param image: The image to be augmented
    :param params: A dictionary of desired augmentations and their parameters
    :return:
    """
    augmentation_funcs = {
        'noise': add_noise,
        'mirror': mirror,
        'intensity_scale': intensity_gain_offset,
        'smooth': filters.gaussian,
        'invert': negate_image,
        'dropout': dropout
    }
    new_image = image.copy()
    for augmentation in params:
        if augmentation not in augmentation_funcs:
            raise ValueError('Augmentation function "%s" not supported.' %
                             augmentation)
        new_image = augmentation_funcs[augmentation](**params[augmentation])
    return new_image


def augment_params():
    """
    A seeder method for generating a parameters list for the augment method
    :return:
    """
    params = {
        'noise':
            {
                'type': 'gaussian',
                'sigma': 1.0,
                'amplitude': 0.01
            },
        'mirror':
            {
                'horizontal': True,
                'vertical': True
            },
        'intensity_scale':
            {
                'min_scale': 0.9,
                'max_scale': 1.1
            },
        'smooth':
            {
                'sigma': 1.0
            },
        'negate_image':
            {
                # Defaults to not inverting, must set to -1.0 to invert
                'value': 1.0
            },
        'dropout':
            {
                'percent': 0.1
            }
    }
    return params


# def label_distance(image):
#     driver = gdal.GetDriverByName('MEM')
#     tmp_ds = driver.Create('', image.shape[1], image.shape[0], 1, gdal.GDT_Byte)
#     tmp_ds.GetRasterBand(1).WriteArray(image)
#     out_ds = driver.Create('', image.shape[1], image.shape[0], 1,
#                            gdal.GDT_Float32)
#     gdal.ComputeProximity(tmp_ds.GetRasterBand(1), out_ds.GetRasterBand(1))
#     dist_img = out_ds.GetRasterBand(1).ReadAsArray()
#     out_ds = None
#     tmp_ds = None
#     return dist_img


def combine_images(image1, image2, weight1, weight2):
    total_image = (image1 * weight1 + image2 * weight2) / (weight1 + weight2)
    return total_image


def dem_descent(image, start_pt, max_count=10):
    """
    Descends a dem based on a starting point as if you were water
    :param image: The image of the digital elevation map
    :param start_pt: The starting point in the image (x, y)
    :param max_count: The number of steps (pixels) to go through the dem before stopping
    :return:
    """
    next_lower = True
    next_pt = np.array(start_pt)
    points = list()
    points.append(start_pt)
    count = 0
    while next_lower:
        img_chip = image[next_pt[1] - 1:next_pt[1] + 2,
                         next_pt[0] - 1:next_pt[0] + 2].copy()
        min_xy = np.unravel_index(np.argmin(img_chip), img_chip.shape)[::-1]
        min_xy += np.array(next_pt) - 1
        d = distance.euclidean(next_pt, min_xy)
        if d > 0:
            points.append(min_xy)
            next_pt = min_xy
        else:
            next_lower = False
        count += 1
        if count > max_count:
            break
    return np.array(points)
