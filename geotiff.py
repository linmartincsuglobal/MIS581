
import json
import logging
import tempfile

import numpy as np
import osgeo.gdal as gdal
import osgeo.osr as osr
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, LineString

import geodetic_conversion as geo_conv

logger = logging.getLogger(__name__)


class Geotiff:
    def __init__(self, file_path):
        if isinstance(file_path, gdal.Dataset):
            self.ds = file_path
        else:
            self.ds = gdal.Open(file_path)
            if self.ds is None:
                raise IOError("GDAL could not read the file correctly.")

    @property
    def num_bands(self):
        return self.ds.RasterCount

    @property
    def metadata(self):
        return self.ds.GetMetadata_Dict()

    @property
    def projection(self):
        return self.ds.GetProjectionRef()

    @property
    def geotransform(self):
        return self.ds.GetGeoTransform()

    @property
    def cornercoords(self):
        aa = self.get_xsize(1) - 0.5
        bb = self.get_ysize(1) - 0.5
        coords = np.array(
            [[0.5, 0.5], [aa, 0.5], [aa, bb], [0.5, bb], [0.5, 0.5]])
        llas = self.pixel_to_lla(coords)

        return llas

    def get_raster_metadata(self, band):
        return self.ds.GetRasterBand(band).GetMetadata()

    def get_xsize(self, band):
        return self.ds.GetRasterBand(band).XSize

    def get_ysize(self, band):
        return self.ds.GetRasterBand(band).YSize

    def get_size(self, band):
        return [self.ds.GetRasterBand(band).YSize,
                self.ds.GetRasterBand(band).XSize]

    def get_raster(self, band):
        return self.ds.GetRasterBand(band)

    def get_nodatavalue(self, band):
        return self.ds.GetRasterBand(band).GetNoDataValue()

    def read_band(self, band, subset=None):
        if not self.band_exists(band):
            raise ValueError('Requested band does not exist.')
        rs = self.get_raster(band)
        if subset is None:
            return rs.ReadAsArray()
        return rs.ReadAsArray(xoff=subset[0], yoff=subset[1],
                              win_xsize=subset[2], win_ysize=subset[3])

    def read_all_bands(self):
        images = self.ds.ReadAsArray()
        images = images.transpose((1, 2, 0))
        return images

    def read_bands(self, bands, subset):
        images = []
        for band in bands:
            images.append(self.read_band(band, subset))
        return np.dstack(images)

    def band_exists(self, band):
        if band > self.num_bands or band < 1:
            return False
        return True

    def create_bounds(self, bounds):
        """
        Extracts the pixel and UTM boundaries for a desired lat/lon AOI.
        :param ls_file: The landsat file location.
            A baseline file or prior collection.
        :param bounds: The lat/lon bounds [min_lon, min_lat, max_lon, max_lat]
        :return:
        """
        llas = [[bounds[0], bounds[1], 0.0],
                [bounds[0], bounds[3], 0.0],
                [bounds[2], bounds[3], 0.0],
                [bounds[2], bounds[1], 0.0]]
        llas = np.array(llas)
        ls_pix_bounds = self.lla_to_pixel(llas)
        rs = self.get_raster(1)
        xsize = rs.XSize
        ysize = rs.YSize
        ls_pix_bounds[ls_pix_bounds[:, 0] < 0, 0] = 0
        ls_pix_bounds[ls_pix_bounds[:, 0] > (xsize - 1), 0] = xsize - 1
        ls_pix_bounds[ls_pix_bounds[:, 1] < 0, 1] = 0
        ls_pix_bounds[ls_pix_bounds[:, 1] > (ysize - 1), 1] = ysize - 1
        pix_mins = np.floor(ls_pix_bounds.min(0)).astype('int')
        pix_maxs = np.ceil(ls_pix_bounds.max(0)).astype('int')
        ls_gdal_bounds = np.array([[pix_mins[0], pix_mins[1]],
                                   [pix_mins[0], pix_maxs[1]],
                                   [pix_maxs[0], pix_maxs[1]],
                                   [pix_maxs[0], pix_mins[1]]])
        ls_pix_bounds = [pix_mins[0], pix_mins[1],
                         pix_maxs[0] - pix_mins[0],
                         pix_maxs[1] - pix_mins[1]]
        ls_bounds_meters = self.pixel_to_geo_units(ls_gdal_bounds.astype('float32'))
        new_bounds = [ls_bounds_meters.min(0)[0], ls_bounds_meters.min(0)[1],
                      ls_bounds_meters.max(0)[0], ls_bounds_meters.max(0)[1]]
        return ls_pix_bounds, new_bounds

    def pixel_to_geo_units(self, pixels):
        """
        Converts pixels to a position in the tiffs spatial reference system
        :param pixels: A N x 2 array of pixel positions
        :return: Lon Lat Alt array equal in size to the pixels array
        """
        # Get the Geotransform vector
        gt = self.ds.GetGeoTransform()
        points = np.zeros(pixels.shape, dtype='float32')
        points[:, 0] = (pixels[:, 0] * gt[1]) + gt[0]
        points[:, 1] = (pixels[:, 1] * gt[5]) + gt[3]
        return points

    def pixel_to_lla(self, pixels):
        """
        Converts pixels to a LLA position
        :param pixels: A N x 2 array of pixel positions
        :return: Lon Lat Alt array equal in size to the pixels array
        """
        new_cs = osr.SpatialReference()
        new_cs.ImportFromEPSG(4326)
        band_cs = osr.SpatialReference()
        band_cs.ImportFromWkt(self.ds.GetProjectionRef())
        if int(gdal.__version__[0]) >= 3:
            # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
            new_cs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            band_cs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        tx = osr.CoordinateTransformation(band_cs, new_cs)
        # Get the Geotransform vector
        gt = self.ds.GetGeoTransform()
        points = np.zeros(pixels.shape, dtype='float32')
        points[:, 0] = (pixels[:, 0] * gt[1]) + gt[0]
        points[:, 1] = (pixels[:, 1] * gt[5]) + gt[3]
        # Work out the boundaries of the new dataset in the target projection
        band_pos = np.array(tx.TransformPoints(points))
        return band_pos

    def lla_to_pixel(self, llas, no_neg_lons=False):
        """
        Converts LLA Lon/Lat/Alt to pixel position
        :param llas: An N x 3 Lon/Lat/Alt array
        :return: The band pixel locations as an Nx2 array (x, y)
        """
        new_cs = osr.SpatialReference()
        new_cs.ImportFromEPSG(4326)
        band_cs = osr.SpatialReference()
        band_cs.ImportFromWkt(self.ds.GetProjectionRef())
        if int(gdal.__version__[0]) >= 3:
            # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
            new_cs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            band_cs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        tx = osr.CoordinateTransformation(new_cs, band_cs)
        # Get the Geotransform vector
        gt = list(self.ds.GetGeoTransform())
        if no_neg_lons:
            gt[0] = 0
            gt[3] = 90
        # Work out the boundaries of the new dataset in the target projection
        band_pos = np.array(tx.TransformPoints(llas))
        if no_neg_lons:
            band_pos = no_neg_lons_converter(band_pos)
        band_pix = np.zeros_like(band_pos)
        band_pix[:, 0] = (band_pos[:, 0] - gt[0]) / gt[1]
        band_pix[:, 1] = (band_pos[:, 1] - gt[3]) / gt[5]
        return band_pix

    def chip_to_pixel(self, pixel, radii):
        """
        Chips the data to a LLA boundary
        :param pixel: The center point of the extraction site
        :param radii: The radius in pixels to extract
        :return: A new Geotiff object
        """
        pixel_srcwin = [pixel[0] - radii[0], pixel[1] - radii[1],
                        radii[0] * 2, radii[1] * 2]
        chip_ds = gdal.Translate('', self.ds, format='MEM', srcWin=pixel_srcwin)
        return Geotiff(chip_ds)

    def chip_to_pixel_bbox(self, bbox):
        """
        Chips the data to a LLA boundary
        :param bbox: [min_x, min_y, max_x, max_y]
        :return: A new Geotiff object
        """
        pixel_srcwin = [bbox[0], bbox[1], bbox[2] - bbox[0] + 1,
                        bbox[3] - bbox[1] - 1]
        chip_ds = gdal.Translate('', self.ds, format='MEM', srcWin=pixel_srcwin)
        return Geotiff(chip_ds)

    def chip_to_lla(self, lla_bounds):
        """
        Chips the data to a LLA boundary
        :param lla_bounds: [min_lon, min_lat, max_lon, max_lat]
        :return: A new Geotiff object
        """
        # Changed this code up so it doesn't go nuts when the geotiff is not
        # WGS84 projected but some other projection, you'll just get crazy
        # answers without knowing something bad went wrong.
        llas = []
        for lon in [lla_bounds[0], lla_bounds[2]]:
            for lat in [lla_bounds[1], lla_bounds[3]]:
                llas.append([lon, lat])
        llas = np.array(llas)
        pixels = self.lla_to_pixel(llas)[:, 0:2]
        min_x, min_y = pixels.min(0)
        max_x, max_y = pixels.max(0)

        # Check the minimum values of the pixels
        if min_x < 0 or max_x < 0:
            logger.warning('Minimum x pixel was below 0.')
            min_x = 0
        if min_y < 0 or max_y < 0:
            logger.warning('Minimum y pixel was below 0.')
            min_y = 0

        # Check the maximum values of the pixels
        x_size = self.get_xsize(1) - 1
        y_size = self.get_ysize(1) - 1
        if max_x > x_size or min_x > x_size:
            logger.warning('Maximum x pixel was > %d.' % x_size)
            min_x = x_size
        if max_y > y_size or min_y > y_size:
            logger.warning('Maximum y pixel was > %d.' % y_size)
            min_y = y_size

        # Compute the ranges to pass to the Translate command
        x_range = max_x - min_x + 1
        y_range = max_y - min_y + 1

        # Make sure we don't have bogus ranges
        if (x_range <= 0) or (y_range <= 0):
            return None

        # Generate the source window from the starting pixel positions and
        # the extent of the extracted area desired
        src_win = [min_x, min_y, x_range, y_range]

        chip_ds = gdal.Translate('', self.ds, format='MEM', srcWin=src_win)

        # Make sure that the GDAL function worked otherwise return None
        if chip_ds is None:
            return None

        return Geotiff(chip_ds)

    def chip_to_lla_point(self, lla_point, radius=0.005):
        """
        Chips the data to a LLA point based on a radius
        :param lla_point: [Longitude, Latitude]
        :param radius: Radius of extraction in degrees
        :return: A new Geotiff object
        """
        lla_bounds = [lla_point[0] - radius, lla_point[1] - radius,
                      lla_point[0] + radius, lla_point[1] + radius]
        return self.chip_to_lla(lla_bounds)

    def chip_to_pix(self, pix_bounds):
        """
        Chips the data to a LLA boundary
        :param lla_bounds: [min_x, min_y, x_range, y_range]
        :return: A new Geotiff object
        """
        pix_srcwin = [pix_bounds[0], pix_bounds[1], pix_bounds[2], pix_bounds[3]]
        chip_ds = gdal.Translate('', self.ds, format='MEM', srcWin=pix_srcwin)
        return Geotiff(chip_ds)

    def mask_from_lines(self, lines, band, line_width=1):
        """
        Draw lines on the image based on the list "lines" passed to the
        function. The thickness of the lines is determined by user.
        :param lines:
        :param band:
        :return: masked image, the indices of the polygons used
        """
        inds = []
        counter = 1
        img = Image.new('I', (self.get_xsize(band), self.get_ysize(band)))
        draw = ImageDraw.Draw(img)
        for ix, line in enumerate(lines):
            coords = np.array(line.coords.xy).T
            xys = self.lla_to_pixel(coords)[:, 0:2]
            ls = LineString(xys)
            if ls.length < 1:
                continue
            inds.append(ix + 1)
            draw.line(xys.flatten().tolist(), fill=ix + 1, width=line_width)
            counter += 1
        del draw
        return np.array(img), inds

    def mask_from_polygons(self, polys, band, min_object_size=2,
                           small_objects_to_pixel=True, no_neg_lons=False):
        """
        Creates a mask the same size as the geotiff band desired.
        The mask is a label of which pixels belong to the polygons.
        :param polys: The shapely Polygons to mask to
        :param band: The geotiff band to create a mask
        :param min_object_size: The smallest in pixels an objects area can be
        :param small_objects_to_pixel: If object is less than a pixel,
        label that single pixel not matter what its size
        :param no_neg_lons: Flag on whether to convert negative latitudes for GRIB2 primarily
        :return: masked image, the indices of the polygons used
        """
        if not self.band_exists(band):
            raise ValueError('Requested band does not exist.')
        inds = []
        counter = 1
        img = Image.new('I', (self.get_xsize(band), self.get_ysize(band)))
        draw = ImageDraw.Draw(img)
        if not isinstance(polys, list):
            coords = np.array(polys.boundary.xy).T
            xys = geo_conv.lla_to_pixel(self.ds, coords)[:, 0:2]
            poly = Polygon(xys)
            if poly.area < 2:
                print('Clipped area too small')
            else:
                inds.append(1)
                draw.polygon(xys.flatten().tolist(), fill=1)
        else:
            for ix, poly in enumerate(polys):
                # Get the outer ring of the polygon
                coords = np.array(poly.boundary.xy).T
                xys = self.lla_to_pixel(coords, no_neg_lons=no_neg_lons)[:, 0:2]
                poly = Polygon(xys)
                if poly.area < min_object_size and small_objects_to_pixel:
                    center_pt = xys.mean(0)
                    x_min, y_min = np.floor(center_pt)
                    x_max, y_max = np.ceil(center_pt)
                    xys = np.array([[x_min, y_max], [x_max, y_max],
                                    [x_max, y_min], [x_min, y_min],
                                    [x_min, y_max]])
                elif poly.area < min_object_size:
                    logger.debug('Clipped area too small')
                    continue
                inds.append(ix + 1)
                draw.polygon(xys.flatten().tolist(), fill=ix + 1)
                counter += 1
        del draw
        return np.array(img), inds

    def chip_to_polygon(self, poly, band, return_offsets=False):
        """
        Chips the geotiff to a shapely polygon
        :param poly: A shapely polygon object
        :param band: The band to be chipped (1 to N)
        :param return_offsets: If offsets are desired to be returned
        :return: A new Geotiff object
        """
        xsize = self.get_xsize(band)
        ysize = self.get_ysize(band)
        coords = np.array(poly.bounds).reshape((2, 2))
        xys = self.lla_to_pixel(coords)[:, 0:2]

        xmin, ymin = xys.min(0)
        xmax, ymax = xys.max(0)
        xs_offset = 0
        if xmin < 0:
            xs_offset = int(0 - xmin)
            xmin = 0
        ys_offset = 0
        if ymin < 0:
            ys_offset = int(0 - ymin)
            ymin = 0
        xe_offset = 0
        if xmax > (xsize - 1):
            xe_offset = int(xmax - (xsize - 1))
            xmax = xsize - 1
        ye_offset = 0
        if ymax > (ysize - 1):
            ye_offset = int(ymax - (ysize - 1))
            ymax = ysize - 1

        x_range = xmax - xmin
        y_range = ymax - ymin
        if x_range == 0:
            return None
        if y_range == 0:
            return None

        srcwin = [int(np.floor(xmin)), int(np.floor(ymin)),
                  int(np.ceil(x_range) + 1), int(np.ceil(y_range) + 1)]
        chip_ds = gdal.Translate('', self.ds, format='MEM', srcWin=srcwin)

        if return_offsets:
            return Geotiff(chip_ds), \
                   (xs_offset, ys_offset, xe_offset, ye_offset)
        else:
            return Geotiff(chip_ds)

    def chip_to_polygon_exterior(self, poly, band, return_offsets=False):
        """
        Chips the geotiff to a shapely polygon
        :param poly: A shapely polygon object
        :param band: The band to be chipped (1 to N)
        :param return_offsets: If offsets are desired to be returned
        :return: A new Geotiff object
        """
        xsize = self.get_xsize(band)
        ysize = self.get_ysize(band)
        coords = np.transpose(np.array(poly.exterior.coords.xy))
        # coords = np.array(poly.bounds).reshape((2, 2))
        xys = geo_conv.lla_to_pixel(self.ds, coords)[:, 0:2]

        xmin, ymin = xys.min(0)
        xmax, ymax = xys.max(0)
        xs_offset = 0
        if xmin < 0:
            xs_offset = int(0 - xmin)
            xmin = 0
        ys_offset = 0
        if ymin < 0:
            ys_offset = int(0 - ymin)
            ymin = 0
        xe_offset = 0
        if xmax > (xsize - 1):
            xe_offset = int(xmax - (xsize - 1))
            xmax = xsize - 1
        ye_offset = 0
        if ymax > (ysize - 1):
            ye_offset = int(ymax - (ysize - 1))
            ymax = ysize - 1

        x_range = xmax - xmin
        y_range = ymax - ymin
        if x_range == 0:
            return None
        if y_range == 0:
            return None

        srcwin = [int(np.floor(xmin)), int(np.floor(ymin)),
                  int(np.ceil(x_range) + 1), int(np.ceil(y_range) + 1)]
        chip_ds = gdal.Translate('', self.ds, format='MEM', srcWin=srcwin)

        if return_offsets:
            return Geotiff(chip_ds), \
                   (xs_offset, ys_offset, xe_offset, ye_offset)
        else:
            return Geotiff(chip_ds)

    def write_file(self, filename, gdal_type=gdal.GDT_Float32,
                   options=['COMPRESS=DEFLATE'], metadata=None, nodataval=0):
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(filename, self.get_xsize(1), self.get_ysize(1),
                               self.num_bands, gdal_type, options=options)
        out_ds.SetProjection(self.projection)
        out_ds.SetGeoTransform(self.geotransform)
        if metadata is None:
            out_ds.SetMetadata(self.metadata)
            out_ds.GetRasterBand(1).SetNoDataValue(nodataval)
        else:
            out_ds.SetMetadata(metadata)
        for b in range(1, self.num_bands + 1):
            out_ds.GetRasterBand(b).WriteArray(self.get_raster(b).ReadAsArray())
            out_ds.GetRasterBand(b).SetNoDataValue(nodataval)
        out_ds.FlushCache()
        out_ds = None

    def write_new_raster_file(self, filename, image, num_bands,
                              options=['COMPRESS=DEFLATE'], gdal_type=gdal.GDT_Float32,
                              gt=None, metadata=None, nodataval=0):
        """
        Writes out a new rasterfile using information from the current geotiff
        :param filename: Output filename
        :param image: The image data to be written to the geotiff
        :param num_bands: The number of bands in the geotiff
        :param options: The geotiff options to use (such as compression)
        :param gdal_type: The gdal type (byte - 1, uint16 - 2, int16 - 3, uint32 - 4, int32 - 5,
        float32 - 6)
        :param gt: The geotransform that describes the new geotiff (if None, it copies the current geotiff)
        :param metadata: The metadata that is desired to be placed into the geotiff
        :param nodataval: The value to set the "nodata" value to in a geotiff
        :return:
        """
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(filename, image.shape[1], image.shape[0],
                               num_bands, gdal_type, options=options)
        # out_ds.
        out_ds.SetProjection(self.projection)
        if metadata is None:
            out_ds.SetMetadata(self.metadata)
        else:
            out_ds.SetMetadata(metadata)
        if gt is None:
            out_ds.SetGeoTransform(self.geotransform)
            out_ds.SetProjection(self.projection)
        else:
            out_ds.SetGeoTransform(gt)
            if np.max(np.abs(gt)) < 180:
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(4326)
                out_ds.SetProjection(srs.ExportToWkt())
            else:
                out_ds.SetProjection(self.projection)

        if num_bands == 1:
            out_ds.GetRasterBand(1).WriteArray(image)
            out_ds.GetRasterBand(1).SetNoDataValue(nodataval)
        else:
            for b in range(1, num_bands + 1):
                out_ds.GetRasterBand(b).WriteArray(image[:, :, b - 1])
                out_ds.GetRasterBand(b).SetNoDataValue(nodataval)
        out_ds.FlushCache()
        out_ds = None

    def write_grib2_subset(self, filename, images, metas, options=['COMPRESS=DEFLATE'],
                           gdal_type=gdal.GDT_Float32):
        """
        Writes out a new raster file using information from the current GRIB2
        :param filename: Output filename
        :param images: A list of images to write to the new file
        :param metas: A list of metadata dictionaries to write to the new file
        :param options: The geotiff options to use (such as compression)
        :param gdal_type: The gdal type (byte - 1, uint16 - 2, int16 - 3, uint32 - 4, int32 - 5,
        float32 - 6)
        :return:
        """
        driver = gdal.GetDriverByName('GTiff')
        num_bands = len(images)
        out_ds = driver.Create(filename, self.get_xsize(1), self.get_ysize(1),
                               num_bands, gdal_type, options=options)
        out_ds.SetProjection(self.projection)
        out_ds.SetGeoTransform(self.geotransform)
        for ix, b in enumerate(images):
            out_ds.GetRasterBand(ix + 1).WriteArray(images[ix])
            out_ds.GetRasterBand(ix + 1).SetMetadata(metas[ix])
        out_ds.FlushCache()
        out_ds = None

    def set_nodata_value(self, no_data_val):
        """
        Sets no data value in Geotiff object band metadata
        :param self: Geotiff object
        :param no_data_val: No data value
        :return: None, object updated in memory
        """
        for ii in range(self.ds.RasterCount):
            self.ds.GetRasterBand(ii + 1).SetNoDataValue(no_data_val)

    def create_inclusion_exclusion_mask(inex_path, inex_type, tiff_class, band=1):
        """
        Create binary mask based on input json and tiff_class
        :param inex_path: Path to inclusion / exclusion json
        :param inex_type: 0 - create exclusion mask, 1 - create inclusion mask
        :param tiff_class: mask input band class object
        :param band: band from mask input to use  - default = 1
        :return:
        """

        coords1 = tiff_class.cornercoords
        poly_tif = Polygon(coords1)
        mask = tiff_class.mask_from_polygons(poly_tif, band)[0]

        if inex_type == 1:
            mask *= 0

        with open(inex_path, 'r') as fid:
            inex_meta = json.load(fid)

        for xx in range(len(inex_meta['features'])):
            coords = np.array(inex_meta['features'][xx]['geometry']['coordinates'][0])
            poly = Polygon(coords)
            if poly.intersects(poly_tif):
                tmp_mask = tiff_class.mask_from_polygons(poly, band)[0]
                if inex_type == 1:
                    mask[tmp_mask == 1] = 1
                else:
                    mask[tmp_mask == 1] = 0
        return mask

    def resize(self, new_shape, warp_algorithm=gdal.GRIORA_Bilinear):
        """
        Resizes the geotiff based on a desired output size (should be multiple of existing output size)
        :param new_shape: New output size (y, x). YX chosen to match the numpy output of .shape
        :param warp_algorithm: The type of gdal alg to use for warping. Options are:
        gdal.GRIORA_NearestNeighbour - 0
        gdal.GRIORA_Bilinear - 1
        gdal.GRIORA_Cubic - 2
        etc...
        :return:
        """
        warp_opts = gdal.WarpOptions(width=new_shape[1], height=new_shape[0],
                                     resampleAlg=warp_algorithm)
        tf = tempfile.NamedTemporaryFile()
        tf_file = tf.name + '.tif'
        new_ds = gdal.Warp(tf_file, self.ds, format='MEM',
                           options=warp_opts)
        return Geotiff(new_ds)


def label_distance(label_image):
    driver = gdal.GetDriverByName('MEM')
    tmp_ds = driver.Create('', label_image.shape[1], label_image.shape[0], 1, gdal.GDT_Byte)
    tmp_ds.GetRasterBand(1).WriteArray(label_image)
    out_ds = driver.Create('', label_image.shape[1], label_image.shape[0], 1,
                           gdal.GDT_Float32)
    gdal.ComputeProximity(tmp_ds.GetRasterBand(1), out_ds.GetRasterBand(1))
    dist_img = out_ds.GetRasterBand(1).ReadAsArray()
    out_ds = None
    tmp_ds = None
    return dist_img


def no_neg_lons_converter(llas):
    for ix in np.arange(len(llas)):
        if llas[ix][0] < 0:
            llas[ix][0] += 360
    return llas


def write_new_geotiff(filename, image, num_bands, epsg, gdal_type, gt, metadata,
                      nodataval=0, options=['COMPRESS=DEFLATE']):
    """
    Writes out a new rasterfile using information from the current geotiff
    :param filename: Output filename
    :param image: The image data to be written to the geotiff
    :param num_bands: The number of bands in the geotiff
    :param epsg: The code for the projection desired
    :param gdal_type: The gdal type (byte - 1, uint16 - 2, int16 - 3, uint32 - 4, int32 - 5,
    float32 - 6)
    :param gt: The geotransform that describes the new geotiff (if None, it copies the current geotiff)
    :param metadata: The metadata that is desired to be placed into the geotiff
    :param nodataval: The value to set the "nodata" value to in a geotiff
    :param options: The geotiff options to use (such as compression)
    :return:
    """
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(filename, image.shape[1], image.shape[0],
                           num_bands, gdal_type, options=options)

    out_ds.SetMetadata(metadata)
    out_ds.SetGeoTransform(gt)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    out_ds.SetProjection(srs.ExportToWkt())

    if num_bands == 1:
        out_ds.GetRasterBand(1).WriteArray(image)
        out_ds.GetRasterBand(1).SetNoDataValue(nodataval)
    else:
        for b in range(1, num_bands + 1):
            out_ds.GetRasterBand(b).WriteArray(image[:, :, b - 1])
            out_ds.GetRasterBand(b).SetNoDataValue(nodataval)
    out_ds.FlushCache()
    out_ds = None
