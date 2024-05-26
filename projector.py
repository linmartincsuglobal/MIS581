
import ctypes
import logging
import os
import warnings

try:
    import cv2
except ImportError as e:
    print(e)
import numpy as np
import skimage.io as io
import skimage.transform as transform
import utm

from numpy.ctypeslib import ndpointer

import images as rd_images
import maps_api as rd_bing
import geotiff as geotiff
import progress as progress

logger = logging.getLogger(__name__)


def lla_grid_to_mercator_grid(image, lats_frame, lons_frame, grid_spacing=20, transform_type='piecewise-affine',
                              degrees_per_pixel=0.01, degrees_per_pixel2=0, order=1):
    """
    Converts an image that has LLA grids to a Mercator projection
    :param image: The image array to be warped
    :param lats_frame: The latitude frame (same size as image)
    :param lons_frame: The longitude frame (same size as image)
    :param grid_spacing: The number of samples across the image to determine
    the transformation matrix
    :param transform_type: Type of transform warpping to use for projection
    :param degrees_per_pixel: The number of degrees per pixel in the output image
    :param degrees_per_pixel2: Options number of degrees per pixel for longitude direction
    :return: The warped Mercator image
    """
    img_shape = image.shape
    xs, ys = np.meshgrid(np.linspace(0, img_shape[1] - 1,
                                     num=grid_spacing).astype('int'),
                         np.linspace(0, img_shape[0] - 1,
                                     num=grid_spacing).astype('int'))
    lats = lats_frame[ys.flatten(), xs.flatten()]
    lons = lons_frame[ys.flatten(), xs.flatten()]

    min_lon = lons_frame.min()
    max_lat = lats_frame.max()

    min_lon = lons_frame.min()
    max_lat = lats_frame.max()

    if degrees_per_pixel2 == 0:
        out_xs = (lons - min_lon) / degrees_per_pixel
        out_ys = (max_lat - lats) / degrees_per_pixel
    else:
        out_xs = (lons - min_lon) / degrees_per_pixel2
        out_ys = (max_lat - lats) / degrees_per_pixel
    # Should do some kind of intelligent decimation of the pixels to be used
    # to determine the transform here. We don't need densely clustered points
    # in the middle and sparse points at the edges, we'd actually like the
    # opposite.

    src = np.hstack((xs.flatten()[:, np.newaxis], ys.flatten()[:, np.newaxis]))
    dst = np.hstack((out_xs[:, np.newaxis], out_ys[:, np.newaxis]))

    allowed_transforms = ['euclidean', 'similarity', 'affine', 'piecewise-affine', 'projective']
    if not transform in allowed_transforms:
        transform_type = 'piecewise-affine'

    tform = transform.estimate_transform(transform_type, src, dst)

    image_max = image.max()

    new_image = transform.warp(image.astype('float32') / image_max,
                               tform.inverse,
                               output_shape=(out_ys.max().astype('int'),
                                             out_xs.max().astype('int')),
                               order=order, mode='constant')

    return new_image * image_max, (min_lon, max_lat)


def project_tiff_to_bing(tiff_file, zoom_level, x_tile_range=None,
                         y_tile_range=None, sampling=5, x_range=None,
                         y_range=None):
    # Only use the center of the sensor focal plane for tiling
    tiff = geotiff.Geotiff(tiff_file)

    image = tiff.read_band(1)

    if x_range is None:
        x_range = [0, image.shape[1]]

    if y_range is None:
        y_range = [0, image.shape[0]]

    x_vals = np.linspace(x_range[0], x_range[1] - 1, num=sampling)
    y_vals = np.linspace(y_range[0], y_range[1] - 1, num=sampling)

    src_vals = []
    for x in x_vals:
        for y in y_vals:
            src_vals.append([x, y])
    src_vals = np.array(src_vals)

    llas = tiff.pixel_to_lla(src_vals)

    src_vals[:, 0] -= x_range[0]
    src_vals[:, 1] -= y_range[0]

    # Get the convert LLAs to world xy positions
    pixel_xs, pixel_ys = rd_bing.lat_long_to_pixel_xy(llas[:, 0], llas[:, 1],
                                                      zoom_level)

    dst_vals = np.hstack((pixel_xs[:, None], pixel_ys[:, None]))

    min_pixels = [np.min(pixel_xs), np.min(pixel_ys)]
    max_pixels = [np.max(pixel_xs), np.max(pixel_ys)]

    if x_tile_range is None:
        min_tile_x = np.int32(np.floor(min_pixels[0] / 256) * 256)
        max_tile_x = np.int32(np.ceil(max_pixels[0] / 256) * 256)
    else:
        min_tile_x = x_tile_range[0]
        max_tile_x = x_tile_range[1]

    if y_tile_range is None:
        min_tile_y = np.int32(np.floor(min_pixels[1] / 256) * 256)
        max_tile_y = np.int32(np.ceil(max_pixels[1] / 256) * 256)
    else:
        min_tile_y = y_tile_range[0]
        max_tile_y = y_tile_range[1]

    out_x_size = (max_tile_x - min_tile_x)
    out_y_size = (max_tile_y - min_tile_y)

    dst_vals[:, 0] -= min_tile_x
    dst_vals[:, 1] -= min_tile_y

    logger.info('Estimating transform.')
    tform = transform.estimate_transform('piecewise-affine', src_vals, dst_vals)
    img_new = image[y_range[0]:y_range[1],
                    x_range[0]:x_range[1]].astype('float32')
    img_max = np.abs(img_new).max()

    logger.info('Performing warp.')
    img_new = transform.warp(img_new / img_max, tform.inverse,
                             output_shape=(out_y_size, out_x_size),
                             mode='constant', cval=0.0, order=3)
    img_new = (img_new * img_max).astype(image.dtype)
    return img_new, np.array([min_tile_x, max_tile_x]),\
        np.array([min_tile_y, max_tile_y])


def create_bing_tiles(output_dir, tiles_map, new_img, tile_x_range,
                      tile_y_range, zoom_level=14, img_range=None,
                      write_images=True, is_cloud_image=False,
                      image_type='jpg'):

    min_tile_x = tile_x_range[0]
    min_tile_y = tile_y_range[0]

    num_x_tiles = (tile_x_range[1] - tile_x_range[0]) // 256
    num_y_tiles = (tile_y_range[1] - tile_y_range[0]) // 256

    # Loop through potential tiles
    for iy in range(num_y_tiles):
        for ix in range(num_x_tiles):
            x_tile = (ix * 256 + min_tile_x) // 256
            y_tile = (iy * 256 + min_tile_y) // 256
            tile_str = '%d_%d' % (x_tile, y_tile)
            quad_key = rd_bing.tile_xy_to_quad_key(x_tile, y_tile, zoom_level)
            y_start = iy * 256
            x_start = ix * 256
            chip = new_img[y_start:y_start + 256, x_start:x_start + 256, ...]

            if tile_str not in tiles_map['tiles']:
                tiles_map['tiles'][tile_str] = dict()

            if is_cloud_image:
                num_non_zero = np.sum(chip > 0)
                if num_non_zero == 0:
                    tiles_map['tiles'][tile_str][
                        'cloud_coverage'] = None
                    continue
                num_cloud_pixels = np.sum(chip == 255)
                cloud_coverage = float(num_cloud_pixels) / float(num_non_zero)
                cloud_coverage = int(cloud_coverage * 100)
                tiles_map['tiles'][tile_str]['cloud_coverage'] = cloud_coverage
                continue

            num_non_zero = np.sum(chip > 0)
            num_pixels = np.product(chip.shape)
            tiles_map['tiles'][tile_str]['coverage'] = float(num_non_zero)\
                                                       / float(num_pixels)

            # If we just want the updated tile map information then don't
            # write out the image file
            if not write_images:
                continue

            if num_non_zero == 0:
                continue

            if img_range is not None:
                tiles_map['tiles'][tile_str]['gain'] = \
                    float((img_range[1] - img_range[0]) / 256.0)
                tiles_map['tiles'][tile_str]['offset'] = float(img_range[0])
            else:
                tiles_map['tiles'][tile_str]['gain'] = 1
                tiles_map['tiles'][tile_str]['offset'] = 0
            tiles_map['tiles'][tile_str]['min'] = float(chip.min())
            tiles_map['tiles'][tile_str]['max'] = float(chip.max())
            tiles_map['tiles'][tile_str]['mean'] = float(chip.mean())
            tiles_map['tiles'][tile_str]['stddev'] = float(chip.std())
            tiles_map['tiles'][tile_str]['quad_key'] = quad_key

            # Create the output filepath
            base_name = os.path.join(str(zoom_level), str(x_tile),
                                     str(y_tile) + '.' + image_type)
            image_file = os.path.join(output_dir, base_name)

            # Need to make the folders if they don't exist
            if not os.path.exists(os.path.dirname(image_file)):
                os.makedirs(os.path.dirname(image_file))

            if img_range is not None:
                chip = rd_images.bytescale(chip, img_range[0], img_range[1])

            if image_type == 'png':
                if len(chip.shape) < 3:
                    chip = np.dstack((chip, chip, chip))
                chip_mask = chip.max(-1) > 0
                chip = np.dstack((chip, chip_mask.astype('uint8') * 255))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if image_type == 'jpg':
                    io.imsave(image_file, chip, quality=90)
                else:
                    io.imsave(image_file, chip)

    return tiles_map


def create_bing_tiles_s1(output_dir, tiles_map, new_img, tile_x_range,
                         tile_y_range, zoom_level=14, img_range=None,
                         write_images=True, stretch_root=1.0, image_type='jpg'):

    min_tile_x = tile_x_range[0]
    min_tile_y = tile_y_range[0]

    num_x_tiles = (tile_x_range[1] - tile_x_range[0]) // 256
    num_y_tiles = (tile_y_range[1] - tile_y_range[0]) // 256

    # Loop through potential tiles
    for iy in range(num_y_tiles):
        for ix in range(num_x_tiles):
            x_tile = (ix * 256 + min_tile_x) // 256
            y_tile = (iy * 256 + min_tile_y) // 256
            tile_str = '%d_%d' % (x_tile, y_tile)
            quad_key = rd_bing.tile_xy_to_quad_key(x_tile, y_tile, zoom_level)
            y_start = iy * 256
            x_start = ix * 256
            chip = new_img[y_start:y_start + 256, x_start:x_start + 256, ...]

            if tile_str not in tiles_map['tiles']:
                tiles_map['tiles'][tile_str] = dict()

            tile_exists = 'coverage' in tiles_map['tiles'][tile_str]

            num_non_zero = np.sum(chip > 0)

            coverage = float(num_non_zero / (256 ** 2))

            if num_non_zero == 0 and not tile_exists:
                tiles_map['tiles'][tile_str]['cloud_coverage'] = None
                tiles_map['tiles'][tile_str]['coverage'] = coverage
                continue

            if tile_exists:
                if coverage <= tiles_map['tiles'][tile_str]['coverage']:
                    continue

            tiles_map['tiles'][tile_str]['coverage'] = coverage

            if chip.max() == 0:
                continue

            if img_range is not None:
                tiles_map['tiles'][tile_str]['gain'] = \
                    float((img_range[1] - img_range[0]) / 256.0)
                tiles_map['tiles'][tile_str]['offset'] = float(img_range[0])
            else:
                tiles_map['tiles'][tile_str]['gain'] = 1
                tiles_map['tiles'][tile_str]['offset'] = 0
            tiles_map['tiles'][tile_str]['min'] = float(chip.min())
            tiles_map['tiles'][tile_str]['max'] = float(chip.max())
            tiles_map['tiles'][tile_str]['mean'] = float(chip.mean())
            tiles_map['tiles'][tile_str]['stddev'] = float(chip.std())
            tiles_map['tiles'][tile_str]['quad_key'] = quad_key

            # Create the output filepath
            base_name = os.path.join(str(zoom_level), str(x_tile),
                                     str(y_tile) + '.' + image_type)
            image_file = os.path.join(output_dir, base_name)

            # If we just want the updated tile map information then don't
            # write out the image file
            if not write_images:
                continue

            # Need to make the folders if they don't exist
            if not os.path.exists(os.path.dirname(image_file)):
                os.makedirs(os.path.dirname(image_file))

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if img_range is not None:
                    root_val = 1.0 / stretch_root
                    chip = rd_images.bytescale(chip ** root_val, img_range[0],
                                               img_range[1])
                    io.imsave(image_file, chip, quality=90)
                else:
                    io.imsave(image_file, chip, quality=90)

    return tiles_map


def grid_resample_mercator(data, lons, lats, dpp, dll_path, lla_bounds=None,
                           mask=None, weights=True):
    if mask is None:
        mask = np.ones_like(data, dtype='bool')

    if lla_bounds is None:
        lla_bounds = [lons[mask].min(),
                      lats[mask].min(),
                      lons[mask].max(),
                      lats[mask].max()]
    data_mask = (lats >= lla_bounds[1]) & (lats <= lla_bounds[3]) & \
                (lons >= lla_bounds[0]) & (lons <= lla_bounds[2]) & mask
    ys, xs = np.where(data_mask)
    pixel_array = np.zeros((len(xs), 3), dtype='float32')
    pixel_array[:, 0] = (lons[ys, xs] - lla_bounds[0]) / dpp
    pixel_array[:, 1] = (lla_bounds[3] - lats[ys, xs]) / dpp
    pixel_array[:, 2] = data[ys, xs]

    num_cols = np.round((lla_bounds[2] - lla_bounds[0] + dpp) / dpp).astype(
        'int') + 1
    num_rows = np.round((lla_bounds[3] - lla_bounds[1] + dpp) / dpp).astype(
        'int') + 1

    weights_img = np.zeros((num_rows + 2, num_cols + 2), dtype='float32')
    new_img = np.zeros((num_rows + 2, num_cols + 2), dtype='float32')

    kernel = np.zeros((5, 5), dtype='float32')
    kernel[2, 2] = 1

    nd = ctypes.CDLL(dll_path)
    grid_resample = nd.gridResample
    grid_resample.restype = int
    grid_resample.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ctypes.c_int32,
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS")]

    success = grid_resample(pixel_array, len(pixel_array), kernel,
                            kernel.copy(), np.array([5, 5]), new_img,
                            weights_img, np.array(new_img.shape))

    new_img[weights_img != 0] /= weights_img[weights_img != 0]
    new_img[new_img == 0] = 0

    if weights:
        return new_img, weights_img
    else:
        return new_img


def grid_resample_mercator_1d(data, lons, lats, dpp, dll_path, lla_bounds=None,
                              mask=None, weights=True):
    if mask is None:
        mask = np.ones_like(data, dtype='bool')

    if lla_bounds is None:
        lla_bounds = [lons[mask].min(),
                      lats[mask].min(),
                      lons[mask].max(),
                      lats[mask].max()]
    data_mask = (lats >= lla_bounds[1]) & (lats <= lla_bounds[3]) & \
                (lons >= lla_bounds[0]) & (lons <= lla_bounds[2]) & mask
    xs = np.where(data_mask)[0]
    pixel_array = np.zeros((len(xs), 3), dtype='float32')
    pixel_array[:, 0] = (lons[xs] - lla_bounds[0]) / dpp
    pixel_array[:, 1] = (lla_bounds[3] - lats[xs]) / dpp
    pixel_array[:, 2] = data[xs]

    num_cols = np.round((lla_bounds[2] - lla_bounds[0] + dpp) / dpp).astype(
        'int') + 1
    num_rows = np.round((lla_bounds[3] - lla_bounds[1] + dpp) / dpp).astype(
        'int') + 1

    weights_img = np.zeros((num_rows + 2, num_cols + 2), dtype='float32')
    new_img = np.zeros((num_rows + 2, num_cols + 2), dtype='float32')

    kernel = np.zeros((5, 5), dtype='float32')
    kernel[2, 2] = 1

    nd = ctypes.CDLL(dll_path)
    grid_resample = nd.gridResample
    grid_resample.restype = int
    grid_resample.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ctypes.c_int32,
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS")]

    success = grid_resample(pixel_array, len(pixel_array), kernel,
                            kernel.copy(), np.array([5, 5]), new_img,
                            weights_img, np.array(new_img.shape))

    new_img[weights_img != 0] /= weights_img[weights_img != 0]
    new_img[new_img == 0] = 0

    if weights:
        return new_img, weights_img
    else:
        return new_img


def grid_resample_utm_python_1d(data, lons, lats, mpp, lla_bounds=None,
                                mask=None, kernel=None):
    """
    Resamples a 1D input data stream to be bilinearly interpolated to a 2D image
    :param data:
    :param lons:
    :param lats:
    :param zoom_level:
    :param lla_bounds:
    :param x_tile_range:
    :param y_tile_range:
    :param mask:
    :param kernel:
    :return:
    """
    if mask is None:
        mask = np.ones_like(data, dtype='bool')

    if lla_bounds is None:
        lla_bounds = [lons[mask].min(),
                      lats[mask].min(),
                      lons[mask].max(),
                      lats[mask].max()]
    data_mask = (lats >= lla_bounds[1]) & (lats <= lla_bounds[3]) & \
                (lons >= lla_bounds[0]) & (lons <= lla_bounds[2]) & mask
    xs = np.where(data_mask)[0]

    lon_mean = lons[xs].mean()
    lat_mean = lats[xs].mean()

    zone_number = utm.latlon_to_zone_number(lat_mean, lon_mean)

    utm_vals_min = utm.from_latlon(lla_bounds[1], lla_bounds[0], force_zone_number=zone_number)
    utm_vals_max = utm.from_latlon(lla_bounds[3], lla_bounds[2], force_zone_number=zone_number)
    utm_bounds = [utm_vals_min[0], utm_vals_min[1], utm_vals_max[0], utm_vals_max[1]]

    pixel_array = np.zeros((len(xs), 3), dtype='float32')
    for ix in range(len(xs)):
        utm_vals = utm.from_latlon(lats[xs[ix]], lons[xs[ix]], force_zone_number=zone_number)
        pixel_array[ix, 0] = (utm_vals[0] - utm_bounds[0]) / mpp
        pixel_array[ix, 1] = (utm_bounds[3] - utm_vals[1]) / mpp
    pixel_array[:, 2] = data[xs]

    if kernel is None:
        kernel = np.zeros((5, 5), dtype='float32')
        kernel[2, 2] = 1

    half_kern = kernel.shape[0] // 2

    num_cols = np.round((utm_bounds[2] - utm_bounds[0] + mpp) / mpp).astype('int') + 1
    num_rows = np.round((utm_bounds[3] - utm_bounds[1] + mpp) / mpp).astype('int') + 1

    weights_img = np.zeros((num_rows + (half_kern * 2),
                            num_cols + (half_kern * 2)), dtype='float32')
    new_img = np.zeros((num_rows + (half_kern * 2),
                        num_cols + (half_kern * 2)), dtype='float32')

    for ix, (x, y, val) in enumerate(pixel_array):
        progress.n_of_m(ix + 1, len(pixel_array))
        x = x + half_kern
        y = y + half_kern
        x2 = np.floor(x).astype('int')
        y2 = np.floor(y).astype('int')
        xdiff = x - x2
        ydiff = y - y2
        tform = transform.AffineTransform(translation=(ydiff, xdiff))
        kernel2 = transform.warp(kernel, tform.inverse)
        kernel2 /= np.sum(kernel2)
        try:
            weights_img[y2 - half_kern:y2 + half_kern + 1, x2 - half_kern:x2 + half_kern + 1] += kernel2
        except ValueError:
            print(y2, x2, weights_img.shape)
        new_img[y2 - half_kern:y2 + half_kern + 1, x2 - half_kern:x2 + half_kern + 1] += (kernel2 * val)
    new_img = new_img[half_kern:-half_kern, half_kern:-half_kern]
    weights_img = weights_img[half_kern:-half_kern, half_kern:-half_kern]

    new_img[weights_img > 0] /= weights_img[weights_img > 0]
    new_img[new_img < 0] = 0
    return new_img, weights_img


def grid_resample_mercator_python_1d(data, lons, lats, dpp, lla_bounds=None,
                                     mask=None, kernel=None):
    """
    Resamples a 1D input data stream to be bilinearly interpolated to a 2D image
    :param data:
    :param lons:
    :param lats:
    :param zoom_level:
    :param lla_bounds:
    :param x_tile_range:
    :param y_tile_range:
    :param mask:
    :param kernel:
    :return:
    """
    if mask is None:
        mask = np.ones_like(data, dtype='bool')

    if lla_bounds is None:
        lla_bounds = [lons[mask].min(),
                      lats[mask].min(),
                      lons[mask].max(),
                      lats[mask].max()]
    data_mask = (lats >= lla_bounds[1]) & (lats <= lla_bounds[3]) & \
                (lons >= lla_bounds[0]) & (lons <= lla_bounds[2]) & mask
    xs = np.where(data_mask)[0]
    pixel_array = np.zeros((len(xs), 3), dtype='float32')
    pixel_array[:, 0] = (lons[xs] - lla_bounds[0]) / dpp
    pixel_array[:, 1] = (lla_bounds[3] - lats[xs]) / dpp
    pixel_array[:, 2] = data[xs]

    if kernel is None:
        kernel = np.zeros((5, 5), dtype='float32')
        kernel[2, 2] = 1

    half_kern = kernel.shape[0] // 2

    num_cols = np.round((lla_bounds[2] - lla_bounds[0] + dpp) / dpp).astype(
        'int') + 1
    num_rows = np.round((lla_bounds[3] - lla_bounds[1] + dpp) / dpp).astype(
        'int') + 1

    weights_img = np.zeros((num_rows + (half_kern * 2),
                            num_cols + (half_kern * 2)), dtype='float32')
    new_img = np.zeros((num_rows + (half_kern * 2),
                        num_cols + (half_kern * 2)), dtype='float32')

    for ix, (x, y, val) in enumerate(pixel_array):
        progress.n_of_m(ix + 1, len(pixel_array))
        x = x + half_kern
        y = y + half_kern
        x2 = np.floor(x).astype('int')
        y2 = np.floor(y).astype('int')
        xdiff = x - x2
        ydiff = y - y2
        tform = transform.AffineTransform(translation=(ydiff, xdiff))
        kernel2 = transform.warp(kernel, tform.inverse)
        kernel2 /= np.sum(kernel2)
        try:
            weights_img[y2 - half_kern:y2 + half_kern + 1, x2 - half_kern:x2 + half_kern + 1] += kernel2
        except ValueError:
            print(y2, x2, weights_img.shape)
        new_img[y2 - half_kern:y2 + half_kern + 1, x2 - half_kern:x2 + half_kern + 1] += (kernel2 * val)
    new_img = new_img[half_kern:-half_kern, half_kern:-half_kern]
    weights_img = weights_img[half_kern:-half_kern, half_kern:-half_kern]

    new_img[weights_img > 0] /= weights_img[weights_img > 0]
    new_img[new_img < 0] = 0
    return new_img, weights_img


def grid_resample_bing_1d(data, lons, lats, zoom_level, dll_path, lla_bounds=None,
                          x_tile_range=None, y_tile_range=None, mask=None,
                          kernel=None):
    if mask is None:
        mask = np.ones_like(data, dtype='bool')

    if np.sum(mask) == 0:
        return None, None, None

    if lla_bounds is None:
        lla_bounds = [lons[mask].min(),
                      lats[mask].min(),
                      lons[mask].max(),
                      lats[mask].max()]
    data_mask = (lats >= lla_bounds[1]) & (lats <= lla_bounds[3]) & \
                (lons >= lla_bounds[0]) & (lons <= lla_bounds[2]) & mask
    inds = np.where(data_mask)

    if len(inds) == 0:
        return None, None, None

    # Get the convert LLAs to world xy positions
    pixel_xs, pixel_ys = rd_bing.lat_long_to_pixel_xy(lons[inds],
                                                      lats[inds],
                                                      zoom_level,
                                                      do_round=False)

    dst_vals = np.hstack((pixel_xs[:, None], pixel_ys[:, None],
                          data[inds][:, None])).astype('float32')

    min_pixels = [np.min(pixel_xs), np.min(pixel_ys)]
    max_pixels = [np.max(pixel_xs), np.max(pixel_ys)]

    if x_tile_range is None:
        min_tile_x = np.int32(np.floor(min_pixels[0] / 256) * 256)
        max_tile_x = np.int32(np.ceil(max_pixels[0] / 256) * 256)
    else:
        min_tile_x = x_tile_range[0]
        max_tile_x = x_tile_range[1]

    if y_tile_range is None:
        min_tile_y = np.int32(np.floor(min_pixels[1] / 256) * 256)
        max_tile_y = np.int32(np.ceil(max_pixels[1] / 256) * 256)
    else:
        min_tile_y = y_tile_range[0]
        max_tile_y = y_tile_range[1]

    num_cols = (max_tile_x - min_tile_x)
    num_rows = (max_tile_y - min_tile_y)

    dst_vals[:, 0] -= min_tile_x
    dst_vals[:, 1] -= min_tile_y

    half_kern = kernel.shape[0] // 2

    weights_img = np.zeros((num_rows + (half_kern * 2),
                            num_cols + (half_kern * 2)), dtype='float32')
    new_img = np.zeros((num_rows + (half_kern * 2),
                        num_cols + (half_kern * 2)), dtype='float32')

    if kernel is None:
        kernel = np.zeros((5, 5), dtype='float32')
        kernel[2, 2] = 1
    # Just in case another datatype was passed in. Force to Float32.
    kernel = kernel.astype('float32')

    nd = ctypes.CDLL(dll_path)
    grid_resample = nd.gridResample
    grid_resample.restype = int
    grid_resample.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ctypes.c_int32,
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS")]

    success = grid_resample(dst_vals, len(dst_vals), kernel,
                            kernel.copy(), np.array(kernel.shape).astype('int32'), new_img,
                            weights_img, np.array(new_img.shape).astype('int32'))

    new_img = new_img[half_kern:-half_kern, half_kern:-half_kern]
    weights_img = weights_img[half_kern:-half_kern, half_kern:-half_kern]
    new_img[weights_img != 0] /= weights_img[weights_img != 0]
    new_img[new_img < 0] = 0

    return new_img, np.array([min_tile_x, max_tile_x]),\
        np.array([min_tile_y, max_tile_y])


def grid_resample_bing(data, lons, lats, zoom_level, dll_path, lla_bounds=None,
                       x_tile_range=None, y_tile_range=None, mask=None,
                       kernel=None):
    if mask is None:
        mask = np.ones_like(data, dtype='bool')

    if np.sum(mask) == 0:
        return None, None, None

    if lla_bounds is None:
        lla_bounds = [lons[mask].min(),
                      lats[mask].min(),
                      lons[mask].max(),
                      lats[mask].max()]
    data_mask = (lats >= lla_bounds[1]) & (lats <= lla_bounds[3]) & \
                (lons >= lla_bounds[0]) & (lons <= lla_bounds[2]) & mask
    ys, xs = np.where(data_mask)

    if len(ys) == 0:
        return None, None, None

    # Get the convert LLAs to world xy positions
    pixel_xs, pixel_ys = rd_bing.lat_long_to_pixel_xy(lons[ys, xs],
                                                      lats[ys, xs],
                                                      zoom_level,
                                                      do_round=False)

    dst_vals = np.hstack((pixel_xs[:, None], pixel_ys[:, None],
                          data[ys, xs][:, None])).astype('float32')

    min_pixels = [np.min(pixel_xs), np.min(pixel_ys)]
    max_pixels = [np.max(pixel_xs), np.max(pixel_ys)]

    if x_tile_range is None:
        min_tile_x = np.int32(np.floor(min_pixels[0] / 256) * 256)
        max_tile_x = np.int32(np.ceil(max_pixels[0] / 256) * 256)
    else:
        min_tile_x = x_tile_range[0]
        max_tile_x = x_tile_range[1]

    if y_tile_range is None:
        min_tile_y = np.int32(np.floor(min_pixels[1] / 256) * 256)
        max_tile_y = np.int32(np.ceil(max_pixels[1] / 256) * 256)
    else:
        min_tile_y = y_tile_range[0]
        max_tile_y = y_tile_range[1]

    num_cols = (max_tile_x - min_tile_x)
    num_rows = (max_tile_y - min_tile_y)

    dst_vals[:, 0] -= min_tile_x
    dst_vals[:, 1] -= min_tile_y

    half_kern = kernel.shape[0] // 2

    weights_img = np.zeros((num_rows + (half_kern * 2),
                            num_cols + (half_kern * 2)), dtype='float32')
    new_img = np.zeros((num_rows + (half_kern * 2),
                        num_cols + (half_kern * 2)), dtype='float32')

    if kernel is None:
        kernel = np.zeros((5, 5), dtype='float32')
        kernel[2, 2] = 1
    # Just in case another datatype was passed in. Force to Float32.
    kernel = kernel.astype('float32')

    nd = ctypes.CDLL(dll_path)
    grid_resample = nd.gridResample
    grid_resample.restype = int
    grid_resample.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ctypes.c_int32,
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS")]

    success = grid_resample(dst_vals, len(dst_vals), kernel,
                            kernel.copy(), np.array(kernel.shape).astype('int32'), new_img,
                            weights_img, np.array(new_img.shape).astype('int32'))

    new_img = new_img[half_kern:-half_kern, half_kern:-half_kern]
    weights_img = weights_img[half_kern:-half_kern, half_kern:-half_kern]
    new_img[weights_img != 0] /= weights_img[weights_img != 0]
    new_img[new_img < 0] = 0

    return new_img, np.array([min_tile_x, max_tile_x]),\
        np.array([min_tile_y, max_tile_y])


def grid_resample_bing_python(data, lons, lats, zoom_level, lla_bounds=None,
                              x_tile_range=None, y_tile_range=None, mask=None,
                              kernel=None):

    if mask is None:
        mask = np.ones_like(data, dtype='bool')

    if np.sum(mask) == 0:
        return None, None, None

    if lla_bounds is None:
        lla_bounds = [lons[mask].min(),
                      lats[mask].min(),
                      lons[mask].max(),
                      lats[mask].max()]
    data_mask = (lats >= lla_bounds[1]) & (lats <= lla_bounds[3]) & \
                (lons >= lla_bounds[0]) & (lons <= lla_bounds[2]) & mask
    ys, xs = np.where(data_mask)

    if len(ys) == 0:
        return None, None, None

    # Get the convert LLAs to world xy positions
    pixel_xs, pixel_ys = rd_bing.lat_long_to_pixel_xy(lons[ys, xs],
                                                      lats[ys, xs],
                                                      zoom_level,
                                                      do_round=False)

    dst_vals = np.hstack((pixel_xs[:, None], pixel_ys[:, None],
                          data[ys, xs][:, None])).astype('float32')

    min_pixels = [np.min(pixel_xs), np.min(pixel_ys)]
    max_pixels = [np.max(pixel_xs), np.max(pixel_ys)]

    if x_tile_range is None:
        min_tile_x = np.int32(np.floor(min_pixels[0] / 256) * 256)
        max_tile_x = np.int32(np.ceil(max_pixels[0] / 256) * 256)
    else:
        min_tile_x = x_tile_range[0]
        max_tile_x = x_tile_range[1]

    if y_tile_range is None:
        min_tile_y = np.int32(np.floor(min_pixels[1] / 256) * 256)
        max_tile_y = np.int32(np.ceil(max_pixels[1] / 256) * 256)
    else:
        min_tile_y = y_tile_range[0]
        max_tile_y = y_tile_range[1]

    num_cols = (max_tile_x - min_tile_x)
    num_rows = (max_tile_y - min_tile_y)

    dst_vals[:, 0] -= min_tile_x
    dst_vals[:, 1] -= min_tile_y

    half_kern = kernel.shape[0] // 2

    weights_img = np.zeros((num_rows + (half_kern * 2),
                            num_cols + (half_kern * 2)), dtype='float32')
    new_img = np.zeros((num_rows + (half_kern * 2),
                        num_cols + (half_kern * 2)), dtype='float32')

    if kernel is None:
        kernel = np.zeros((5, 5), dtype='float32')
        kernel[2, 2] = 1

    for x, y, val in dst_vals:
        x = x + half_kern
        y = y + half_kern
        x2 = np.floor(x).astype('int')
        y2 = np.floor(y).astype('int')
        xdiff = x - x2
        ydiff = y - y2
        tform = transform.AffineTransform(translation=(ydiff, xdiff))
        kernel2 = transform.warp(kernel, tform.inverse)
        kernel2 /= np.sum(kernel2)
        try:
            weights_img[y2 - 2:y2 + 3, x2 - 2:x2 + 3] += kernel2
        except ValueError:
            print(y2, x2, weights_img.shape)
        new_img[y2 - 2:y2 + 3, x2 - 2:x2 + 3] += (kernel2 * val)
    new_img = new_img[half_kern:-half_kern, half_kern:-half_kern]
    weights_img = weights_img[half_kern:-half_kern, half_kern:-half_kern]

    new_img[weights_img > 0] /= weights_img[weights_img > 0]
    new_img[new_img < 0] = 0
    return new_img, np.array([min_tile_x, max_tile_x]),\
        np.array([min_tile_y, max_tile_y])


def grid_resample_bing_python_1d(data, lons, lats, zoom_level, lla_bounds=None,
                                 x_tile_range=None, y_tile_range=None, mask=None,
                                 kernel=None):
    """
    Resamples a 1D input data stream to be bilinearly interpolated to a 2D image
    :param data:
    :param lons:
    :param lats:
    :param zoom_level:
    :param lla_bounds:
    :param x_tile_range:
    :param y_tile_range:
    :param mask:
    :param kernel:
    :return:
    """
    if mask is None:
        mask = np.ones_like(data, dtype='bool')

    if np.sum(mask) == 0:
        return None, None, None

    if lla_bounds is None:
        lla_bounds = [lons[mask].min(),
                      lats[mask].min(),
                      lons[mask].max(),
                      lats[mask].max()]
    data_mask = (lats >= lla_bounds[1]) & (lats <= lla_bounds[3]) & \
                (lons >= lla_bounds[0]) & (lons <= lla_bounds[2]) & mask
    inds = np.where(data_mask)

    if len(inds) == 0:
        return None, None, None

    # Get the convert LLAs to world xy positions
    pixel_xs, pixel_ys = rd_bing.lat_long_to_pixel_xy(lons[inds],
                                                      lats[inds],
                                                      zoom_level,
                                                      do_round=False)

    dst_vals = np.hstack((pixel_xs[:, None], pixel_ys[:, None],
                          data[inds][:, None])).astype('float32')

    min_pixels = [np.min(pixel_xs), np.min(pixel_ys)]
    max_pixels = [np.max(pixel_xs), np.max(pixel_ys)]

    if x_tile_range is None:
        min_tile_x = np.int32(np.floor(min_pixels[0] / 256) * 256)
        max_tile_x = np.int32(np.ceil(max_pixels[0] / 256) * 256)
    else:
        min_tile_x = x_tile_range[0]
        max_tile_x = x_tile_range[1]

    if y_tile_range is None:
        min_tile_y = np.int32(np.floor(min_pixels[1] / 256) * 256)
        max_tile_y = np.int32(np.ceil(max_pixels[1] / 256) * 256)
    else:
        min_tile_y = y_tile_range[0]
        max_tile_y = y_tile_range[1]

    num_cols = (max_tile_x - min_tile_x)
    num_rows = (max_tile_y - min_tile_y)

    dst_vals[:, 0] -= min_tile_x
    dst_vals[:, 1] -= min_tile_y

    half_kern = kernel.shape[0] // 2

    weights_img = np.zeros((num_rows + (half_kern * 2),
                            num_cols + (half_kern * 2)), dtype='float32')
    new_img = np.zeros((num_rows + (half_kern * 2),
                        num_cols + (half_kern * 2)), dtype='float32')

    if kernel is None:
        kernel = np.zeros((5, 5), dtype='float32')
        kernel[2, 2] = 1

    for x, y, val in dst_vals:
        x = x + half_kern
        y = y + half_kern
        x2 = np.floor(x).astype('int')
        y2 = np.floor(y).astype('int')
        xdiff = x - x2
        ydiff = y - y2
        tform = transform.AffineTransform(translation=(ydiff, xdiff))
        kernel2 = transform.warp(kernel, tform.inverse)
        kernel2 /= np.sum(kernel2)
        try:
            weights_img[y2 - 2:y2 + 3, x2 - 2:x2 + 3] += kernel2
        except ValueError:
            print(y2, x2, weights_img.shape)
        new_img[y2 - 2:y2 + 3, x2 - 2:x2 + 3] += (kernel2 * val)
    new_img = new_img[half_kern:-half_kern, half_kern:-half_kern]
    weights_img = weights_img[half_kern:-half_kern, half_kern:-half_kern]

    new_img[weights_img > 0] /= weights_img[weights_img > 0]
    new_img[new_img < 0] = 0
    return new_img, np.array([min_tile_x, max_tile_x]),\
        np.array([min_tile_y, max_tile_y])


def grid_resample_bing_1d(data, lons, lats, zoom_level, dll_path, lla_bounds=None,
                          x_tile_range=None, y_tile_range=None, mask=None,
                          kernel=None):
    """
    Resamples a 1D input data stream to be bilinearly interpolated to a 2D image
    :param data: The data as a 1D set of values
    :param lons: A 1D array of longitudes
    :param lats: A 1D array of latitudes
    :param zoom_level: The zoom level of the bing map to be produced
    :param dll_path: The path to the grid resample DLL
    :param lla_bounds: The LLA bounds to limit the map creation to
    :param x_tile_range: The x_tile_range to remove x bias
    :param y_tile_range: The y_tile_range to remove y bias
    :param mask: A mask showing where good data resides
    :param kernel: A kernel for interpolating signals across pixels
    :return:
    """
    if mask is None:
        mask = np.ones_like(data, dtype='bool')

    if np.sum(mask) == 0:
        return None, None, None

    if lla_bounds is None:
        lla_bounds = [lons[mask].min(),
                      lats[mask].min(),
                      lons[mask].max(),
                      lats[mask].max()]
    data_mask = (lats >= lla_bounds[1]) & (lats <= lla_bounds[3]) & \
                (lons >= lla_bounds[0]) & (lons <= lla_bounds[2]) & mask
    inds = np.where(data_mask)

    if len(inds) == 0:
        return None, None, None

    # Get the convert LLAs to world xy positions
    pixel_xs, pixel_ys = rd_bing.lat_long_to_pixel_xy(lons[inds],
                                                      lats[inds],
                                                      zoom_level,
                                                      do_round=False)

    dst_vals = np.hstack((pixel_xs[:, None], pixel_ys[:, None],
                          data[inds][:, None])).astype('float32')

    min_pixels = [np.min(pixel_xs), np.min(pixel_ys)]
    max_pixels = [np.max(pixel_xs), np.max(pixel_ys)]

    if x_tile_range is None:
        min_tile_x = np.int32(np.floor(min_pixels[0] / 256) * 256)
        max_tile_x = np.int32(np.ceil(max_pixels[0] / 256) * 256)
    else:
        min_tile_x = x_tile_range[0]
        max_tile_x = x_tile_range[1]

    if y_tile_range is None:
        min_tile_y = np.int32(np.floor(min_pixels[1] / 256) * 256)
        max_tile_y = np.int32(np.ceil(max_pixels[1] / 256) * 256)
    else:
        min_tile_y = y_tile_range[0]
        max_tile_y = y_tile_range[1]

    num_cols = (max_tile_x - min_tile_x)
    num_rows = (max_tile_y - min_tile_y)

    dst_vals[:, 0] -= min_tile_x
    dst_vals[:, 1] -= min_tile_y

    half_kern = kernel.shape[0] // 2

    weights_img = np.zeros((num_rows + (half_kern * 2),
                            num_cols + (half_kern * 2)), dtype='float32')
    new_img = np.zeros((num_rows + (half_kern * 2),
                        num_cols + (half_kern * 2)), dtype='float32')

    if kernel is None:
        kernel = np.zeros((5, 5), dtype='float32')
        kernel[2, 2] = 1

    kernel = kernel.astype('float32')

    nd = ctypes.CDLL(dll_path)
    grid_resample = nd.gridResample
    grid_resample.restype = int
    grid_resample.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ctypes.c_int32,
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS")]

    success = grid_resample(dst_vals, len(dst_vals), kernel,
                            kernel.copy(), np.array(kernel.shape).astype('int32'), new_img,
                            weights_img, np.array(new_img.shape).astype('int32'))

    new_img = new_img[half_kern:-half_kern, half_kern:-half_kern]
    weights_img = weights_img[half_kern:-half_kern, half_kern:-half_kern]
    new_img[weights_img != 0] /= weights_img[weights_img != 0]
    new_img[new_img < 0] = 0

    return new_img, np.array([min_tile_x, max_tile_x]),\
        np.array([min_tile_y, max_tile_y])
