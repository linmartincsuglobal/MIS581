
import requests
import string

from io import BytesIO

import numpy as np

from PIL import Image

import math as rd_math

MIN_LAT = -85.05112878
MAX_LAT = 85.05112878
MIN_LONG = -180.0
MAX_LONG = 180.0

API_KEY = 'Agvan6s5emScIfWKIxtjgTxVSoCYN3tBsBJjtd7E3wm2HuVgJuZiXQNcwj8JxZyH'


def ground_resolution(llas, zoom_level):
    """
    Calculates a ground resolution based on the LLA
    :param llas: [lon, lat, alt]
    :param zoom_level: The zoom level of the tiles
    :return:
    """
    grs = (np.cos(llas[:, 1] * np.pi / 180.0) * 2.0 * np.pi * 6378137.0)\
        / (256 * 2 ** zoom_level)
    return grs


def meta_at_lat_long(lat, long, xsize, ysize, zoom_level, api_key=None):
    if api_key is None:
        api_key = API_KEY
    base_url = string.Template(
        '''https://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/${lat},${long}/${zoom_level}?format=png&key=${api_key}&mapSize=${xsize},${ysize}&mapMetadata=${map_metadata}''')
    query_url = base_url.substitute({'lat': lat, 'long': long, 'xsize': xsize,
                                     'ysize': ysize, 'api_key': api_key,
                                     'zoom_level': zoom_level,
                                     'map_metadata': 1})
    response = requests.get(query_url)
    if response.ok:
        meta = response.json()
    else:
        print('Query failed to return metadata.')
        print(response.content)
        return None
    return meta


def image_at_lat_long(lat, long, xsize, ysize, zoom_level, api_key=None):
    if api_key is None:
        api_key = API_KEY
    base_url = string.Template(
        '''https://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/${lat},${long}/${zoom_level}?format=png&key=${api_key}&mapSize=${xsize},${ysize}&mapMetadata=${map_metadata}''')
    query_url = base_url.substitute({'lat': lat, 'long': long, 'xsize': xsize,
                                     'ysize': ysize, 'api_key': api_key,
                                     'zoom_level': zoom_level,
                                     'map_metadata': 0})
    print(query_url)
    response = requests.get(query_url)
    if response.ok:
        image = Image.open(BytesIO(response.content))
        image = np.asarray(image)
    else:
        print('Query failed to return image.')
        print(response.content)
        return None

    return image


def tile_xy_to_quad_key(tile_x, tile_y, zoom_level):
    """
    Creates a quadrant key based on the tile XY of interest
    :param tile_x: The tile x position
    :param tile_y: The tile y position
    :param zoom_level: The zoom level of the data
    :return: quadrant key string used to retrieve desired tile
    """
    quad_key = ''
    for ix in range(1, zoom_level + 1)[::-1]:
        digit = 0
        mask = np.left_shift(1, (ix - 1))
        if (tile_x & mask) != 0:
            digit += 1
        if (tile_y & mask) != 0:
            digit += 2
        quad_key += str(digit)
    return quad_key


def lat_long_to_pixel_xy(long, lat, zoom_level, do_round=True):
    """
    Returns the tile position for bing maps from a long/lat position
    :param long: The longitude of interest
    :param lat: The latitude of interest
    :param zoom_level: The zoom level of the data
    :return: tuple of x,y coordinates related to the long/lat position
    """
    lat = rd_math.clip(lat, MIN_LAT, MAX_LAT)
    long = rd_math.clip(long, MIN_LONG, MAX_LONG)

    x = (long + 180) / 360
    sin_latitude = np.sin(lat * np.pi / 180.0)
    y = 0.5 - np.log((1 + sin_latitude) / (1 - sin_latitude)) / (4 * np.pi)
    map_size = np.left_shift(256, zoom_level)
    pixel_x = rd_math.clip(x * map_size + 0.5, 0, map_size - 1)
    pixel_y = rd_math.clip(y * map_size + 0.5, 0, map_size - 1)
    if do_round:
        pixel_x = np.int32(pixel_x)
        pixel_y = np.int32(pixel_y)
    return pixel_x, pixel_y


def pixel_xy_to_tile_xy(pixel_x, pixel_y):
    return pixel_x // 256, pixel_y // 256


def tile_xy_to_pixel_xy(tile_x, tile_y):
    return tile_x * 256, tile_y * 256


def pixel_xy_to_lat_long(pixel_x, pixel_y, zoom_level):
    map_size = np.left_shift(256, zoom_level)
    x = (rd_math.clip(pixel_x, 0, map_size - 1) / float(map_size)) - 0.5
    y = 0.5 - (rd_math.clip(pixel_y, 0, map_size - 1) / float(map_size))

    lat = 90 - 360 * np.arctan(np.exp(-y * 2 * np.pi)) / np.pi
    long = 360 * x
    return long, lat


def generate_image_from_bounding_box(lla_bounds, zoom_level):
    """
    Generates an image from tile data. Not a good idea to run this terribly
    often since it can hit their tile servers hard.
    :param lla_bounds: The [min_lon, min_lat, max_lon, max_lat] bounding box
    :param zoom_level: The zoom level for the data of interest
    :return: An RGB image stitched from all of the tiles
    """
    # The lower left tile
    x_tiles, y_tiles = get_image_boundaries(lla_bounds, zoom_level)

    min_x_tile = x_tiles.min()
    min_y_tile = y_tiles.min()

    x_tiles -= min_x_tile
    y_tiles -= min_y_tile

    x_size = len(x_tiles) * 256
    y_size = len(y_tiles) * 256

    if (x_size > 8192) or (y_size > 8192):
        print('Image is too large. Returning.')
        print(y_size, x_size)
        return

    image = np.zeros((y_size, x_size, 3), dtype='uint8')

    for x_tile in x_tiles:
        for y_tile in y_tiles:
            x_pos = x_tile
            y_pos = y_tile
            x_start = x_pos * 256
            y_start = y_pos * 256
            x_end = x_start + 256
            y_end = y_start + 256
            quad_key = tile_xy_to_quad_key(x_tile + min_x_tile, y_tile +
                                           min_y_tile, zoom_level)
            url = 'https://ecn.t3.tiles.virtualearth.net/tiles/a' + quad_key \
                  + '.jpeg?g=5793&mkt='
            response = requests.get(url)
            if response.ok:
                img = Image.open(BytesIO(response.content))
                img = np.asarray(img)
                image[y_start:y_end, x_start:x_end, :] = img
    return image


def get_tiles_for_bounds(lla_bounds, zoom_level):
    """
    Computes the tiles included within the lla boundary
    :param lla_bounds: The [min_lon, min_lat, max_lon, max_lat] bounding box
    :param zoom_level: The zoom level for the data of interest
    :return: A tuple of x tiles and y tiles
    """
    # The lower left tile
    ll = lat_long_to_pixel_xy(lla_bounds[0], lla_bounds[1], zoom_level)
    ul = lat_long_to_pixel_xy(lla_bounds[0], lla_bounds[3], zoom_level)
    ur = lat_long_to_pixel_xy(lla_bounds[2], lla_bounds[3], zoom_level)
    lr = lat_long_to_pixel_xy(lla_bounds[2], lla_bounds[1], zoom_level)

    tile_bounds = np.vstack((ll, ul, ur, lr))

    min_x, min_y = tile_bounds.min(0)
    max_x, max_y = tile_bounds.max(0)

    min_x_tile = min_x // 256
    min_y_tile = min_y // 256
    max_x_tile = max_x // 256
    max_y_tile = max_y // 256

    x_tiles = np.arange(min_x_tile, max_x_tile + 1)
    y_tiles = np.arange(min_y_tile, max_y_tile + 1)

    return x_tiles, y_tiles


def get_tiles_bounds_from_lla_bounds(lla_bounds, zoom_level):
    x_tiles, y_tiles = get_tiles_for_bounds(lla_bounds, zoom_level)
    min_x_tile = x_tiles.min()
    min_y_tile = y_tiles.min()
    max_x_tile = x_tiles.max() + 1
    max_y_tile = y_tiles.max() + 1

    bing_minx, bing_miny = tile_xy_to_pixel_xy(min_x_tile, min_y_tile)
    bing_maxx, bing_maxy = tile_xy_to_pixel_xy(max_x_tile, max_y_tile)

    bing_min_long, bing_max_lat = pixel_xy_to_lat_long(bing_minx, bing_miny,
                                                       zoom_level)
    bing_max_long, bing_min_lat = pixel_xy_to_lat_long(bing_maxx, bing_maxy,
                                                       zoom_level)


