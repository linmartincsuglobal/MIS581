
import numpy as np
import osgeo.gdal as gdal
import osgeo.osr as osr
import pyproj as pyproj

K0 = 0.9996

E = 0.00669438
E2 = E * E
E3 = E2 * E
E_P2 = E / (1.0 - E)

SQRT_E = np.sqrt(1 - E)
_E = (1 - SQRT_E) / (1 + SQRT_E)
_E2 = _E * _E
_E3 = _E2 * _E
_E4 = _E3 * _E
_E5 = _E4 * _E

M1 = (1 - E / 4 - 3 * E2 / 64 - 5 * E3 / 256)
M2 = (3 * E / 8 + 3 * E2 / 32 + 45 * E3 / 1024)
M3 = (15 * E2 / 256 + 45 * E3 / 1024)
M4 = (35 * E3 / 3072)

P2 = (3. / 2 * _E - 27. / 32 * _E3 + 269. / 512 * _E5)
P3 = (21. / 16 * _E2 - 55. / 32 * _E4)
P4 = (151. / 96 * _E3 - 417. / 128 * _E5)
P5 = (1097. / 512 * _E4)

R = 6378137

ZONE_LETTERS = "CDEFGHJKLMNPQRSTUVWXX"

EARTH_A = 6378137.0
EARTH_F = 1.0 / 298.2572235630
EARTH_B = EARTH_A * (1 - EARTH_F)


def lla_to_ecef(llas):
    """
    Turns lon/lat/alt (degrees) array into Earth Centered Earth Fixed
    coordinates
    :param llas: An array or list of lon/lat/alts
    :return: An array of ECEFs matching the length of the LLA array
    """
    ecef_proj = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla_proj = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    ecefs = []
    for lla in llas:
        ecef = pyproj.transform(lla_proj, ecef_proj, lla[0],
                                lla[1], lla[2], radians=False)
        ecefs.append(np.array(ecef))
    return ecefs


def ecef_to_lla(ecefs):
    """
    Turns ECEF array into lon/lat/alt degrees coordinates
    :param ecefs: An array or list of ECEF coordinates
    :return: An array of LLAs (degrees) matching the length of the ECEF array
    """
    ecef_proj = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla_proj = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    llas = []
    for ecef in ecefs:
        lla = pyproj.transform(ecef_proj, lla_proj, ecef[0],
                               ecef[1], ecef[2], radians=False)
        llas.append(np.array([lla[0], lla[1], lla[2]]))
    return llas


def lla_to_pixel(tiff_file, llas):
    """
    Converts a lla to a pixel location based on the input tiff's geolocation
    :param tiff_file: The landsat file
    :param llas: The llas [lon, lat, alts] desired to be converted
    :return:
    """
    if isinstance(tiff_file, gdal.Dataset):
        ds = tiff_file
    else:
        ds = gdal.Open(tiff_file)
    wgs84_wkt = """
        GEOGCS["WGS 84",
            DATUM["WGS_1984",
                SPHEROID["WGS 84",6378137,298.257223563,
                    AUTHORITY["EPSG","7030"]],
                AUTHORITY["EPSG","6326"]],
            PRIMEM["Greenwich",0,
                AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.01745329251994328,
                AUTHORITY["EPSG","9122"]],
            AUTHORITY["EPSG","4326"]]"""
    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)
    band_cs = osr.SpatialReference()
    band_cs.ImportFromWkt(ds.GetProjectionRef())
    tx = osr.CoordinateTransformation(new_cs, band_cs)
    # Get the Geotransform vector
    gt = ds.GetGeoTransform()
    # Work out the boundaries of the new dataset in the target projection
    tiff_pos = np.array(tx.TransformPoints(llas))
    pixels = np.zeros_like(tiff_pos)
    pixels[:, 0] = (tiff_pos[:, 0] - gt[0]) / gt[1]
    pixels[:, 1] = (tiff_pos[:, 1] - gt[3]) / gt[5]
    return pixels


def ecef2enu(x, y, z, lat0, lon0):
    lat0 = np.deg2rad(lat0)
    lon0 = np.deg2rad(lon0)
    t = np.cos(lon0) * x + np.sin(lon0) * y
    east = -np.sin(lon0) * x + np.cos(lon0) * y
    up = np.cos(lat0) * t + np.sin(lat0) * z
    north = -np.sin(lat0) * t + np.cos(lat0) * z
    return east, north, up


def lla2enu(lat, lon, alt, lat0, lon0, alt0):
    x1, y1, z1 = lla2ecef(lat, lon, alt)
    x2, y2, z2 = lla2ecef(lat0, lon0, alt0)
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    return ecef2enu(dx, dy, dz, lat0, lon0)


def lla2ecef(lat, lon, alt):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    # radius of curvature of the prime vertical section
    earth_norm = get_radius_normal(lat)
    # Compute cartesian (geocentric) coordinates given  (curvilinear) geodetic
    # coordinates.
    x = (earth_norm + alt) * np.cos(lat) * np.cos(lon)
    y = (earth_norm + alt) * np.cos(lat) * np.sin(lon)
    z = (earth_norm * (EARTH_B / EARTH_A) ** 2 + alt) * np.sin(lat)
    return x, y, z


def get_radius_normal(lats):
    num = EARTH_A ** 2
    den = np.sqrt(EARTH_A ** 2 * np.cos(lats) ** 2 +
                  EARTH_B ** 2 * np.sin(lats) ** 2)
    return num / den


def lla_to_utm(lats, lons, force_zone_number=None):
    if (lats.min() <= -80.0) or (lats.max() >= 84.0):
        raise ValueError('latitude out of range (must be between 80 deg S and '
                         '84 deg N)')
    if (lons.min() < -180.0) or (lons.max() >= 180.0):
        raise ValueError('longitude out of range (must be between 180 deg W '
                         'and 180 deg E)')

    lat_rad = np.deg2rad(lats)
    lat_sin = np.sin(lat_rad)
    lat_cos = np.cos(lat_rad)

    lat_tan = lat_sin / lat_cos
    lat_tan2 = lat_tan * lat_tan
    lat_tan4 = lat_tan2 * lat_tan2

    mean_lat = lats.mean()
    mean_lon = lons.mean()

    if force_zone_number is None:
        zone_number = latlon_to_zone_number(mean_lat, mean_lon)
    else:
        zone_number = force_zone_number

    zone_letter = latitude_to_zone_letter(mean_lat)

    lon_rad = np.deg2rad(lons)
    central_lon = zone_number_to_central_longitude(zone_number)
    central_lon_rad = np.deg2rad(central_lon)

    n = R / np.sqrt(1 - E * lat_sin**2)
    c = E_P2 * lat_cos**2

    a = lat_cos * (lon_rad - central_lon_rad)
    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    a5 = a4 * a
    a6 = a5 * a

    m = R * (M1 * lat_rad -
             M2 * np.sin(2 * lat_rad) +
             M3 * np.sin(4 * lat_rad) -
             M4 * np.sin(6 * lat_rad))

    easting = K0 * n * (a +
                        a3 / 6 * (1 - lat_tan2 + c) +
                        a5 / 120 * (5 - 18 * lat_tan2 + lat_tan4 + 72 * c - 58 * E_P2)) + 500000

    northing = K0 * (m + n * lat_tan * (a2 / 2 +
                                        a4 / 24 * (5 - lat_tan2 + 9 * c + 4 * c**2) +
                                        a6 / 720 * (61 - 58 * lat_tan2 + lat_tan4 + 600 * c - 330 * E_P2)))

    northing[lats < 0] += 10000000

    return easting, northing, zone_number, zone_letter


def latlon_to_zone_number(latitude, longitude):
    if 56 <= latitude < 64 and 3 <= longitude < 12:
        return 32

    if 72 <= latitude <= 84 and longitude >= 0:
        if longitude <= 9:
            return 31
        elif longitude <= 21:
            return 33
        elif longitude <= 33:
            return 35
        elif longitude <= 42:
            return 37

    return int((longitude + 180) / 6) + 1


def latitude_to_zone_letter(latitude):
    if -80 <= latitude <= 84:
        return ZONE_LETTERS[int(latitude + 80) >> 3]
    else:
        return None


def zone_number_to_central_longitude(zone_number):
    return (zone_number - 1) * 6 - 180 + 3
