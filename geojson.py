
import json
import os

import numpy as np
import osgeo.gdal as gdal
import osgeo.ogr as ogr
import xmltodict

from PIL import Image, ImageDraw
from shapely.geometry import Polygon

import geodetic_conversion as geo_conv


def polygon_to_raster(tiff_file, geojson_dict):
    """
    Gets the polygons from a geojson file and
    :param tiff_file: The tiff to use as the raster source size and projection
    :param geojson_dict: The GEOJSON dictionary
    :return: Raster Image, Dictionary Providing Map of Polygon Name to Raster
        Value
    """
    poly_map = {}
    counter = 1
    ds = gdal.Open(tiff_file)
    rs = ds.GetRasterBand(1)
    img = Image.new('I', (rs.XSize, rs.YSize))
    draw = ImageDraw.Draw(img)
    for feature in geojson_dict['features']:
        if feature['geometry']['type'] != 'Polygon':
            continue
        name = feature['properties']['name']
        if 'model_type' in feature['properties']:
            model_type = feature['properties']['model_type']
        else:
            model_type = 'NORMAL'           
        if 'algorithm' in feature['properties']:
            algorithm  = feature['properties']['algorithm']
        else:
            algorithm  = 'on_off'
        if 'default_status' in feature['properties']:
            default  = feature['properties']['default_status']
        else:
            default  = 40    
        if 'target id' in feature['properties']:
            target_id  = feature['properties']['target id']
        else:
            target_id  = counter
#        model_type = feature['properties'].get('model_type', 'normal')
#        algorithm = feature['properties'].get('algorithm', 'on_off')        
        poly_map[name] = {'value': counter, 'model_type': model_type, 
            'algorithm': algorithm,'default_status': default, 
            'target_id': target_id}        
        # Get the outer ring of the polygon
        coords = np.array(feature['geometry']['coordinates'][0])
        xys = geo_conv.lla_to_pixel(tiff_file, coords)[:, 0:2]
        draw.polygon(xys.flatten().tolist(), fill=counter)
        counter += 1
    del draw
    return np.array(img), poly_map 


def polygon_to_raster_generic(tiff_file, geojson_dict):
    """
    The generic version of the above
    :param tiff_file: The tiff to use as the raster source size and projection
    :param geojson_dict: The GEOJSON dictionary
    :return: Raster Image
    """
    if not isinstance(geojson_dict, dict):
        with open(geojson_dict, 'r') as fid:
            geojson_dict = json.load(fid)
    counter = 1
    ds = gdal.Open(tiff_file)
    rs = ds.GetRasterBand(1)
    img = Image.new('I', (rs.XSize, rs.YSize))
    draw = ImageDraw.Draw(img)
    for feature in geojson_dict['features']:
        if feature['geometry']['type'] != 'Polygon':
            continue
        # Get the outer ring of the polygon
        coords = np.array(feature['geometry']['coordinates'][0])
        xys = geo_conv.lla_to_pixel(tiff_file, coords)[:, 0:2]
        poly = Polygon(xys)
        if poly.area < 2:
            print('Clipped area too small')
            continue
        draw.polygon(xys.flatten().tolist(), fill=counter)
        counter += 1
    del draw
    return np.array(img)


def load_shapefile_info(shp_file):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(shp_file, 0)
    polys = []
    poly_map = {}
    # counter = 1
    for lyr in range(ds.GetLayerCount()):
        layer = ds.GetLayer(lyr)
        for ft in range(layer.GetFeatureCount()):
            feat = layer.GetFeature(ft)
            geo_ref = feat.GetGeometryRef()
            poly_ref = geo_ref.GetGeometryRef(0)
            points = np.array(poly_ref.GetPoints())
            bounds = np.hstack([points.min(0), points.max(0)])
            polys.append([lyr+1,ft+1,points, bounds])
            keys = feat.keys()
            if 'id' in keys:
                name = feat['id']
                keys.remove('id')
            else:
                name = ft+1

            poly_map[ft+1] = {'layer':lyr+1,'id': name}
            for key in keys:
                poly_map[ft+1].update({key:feat[key]})
    return polys, poly_map


def shp_to_raster(tif_fil, shp_fil):
    """
    Gets the polygons from a shapefile and returns a raster mask
    :param tif_file: The tif to use as the raster source size and projection
    :param shp_fil: The shapefile
    :return: Raster image, dictionary of shapefile metadata, polygon points
    """
    # poly_map = {}
    counter = 1
    ds = gdal.Open(tif_fil)
    rs = ds.GetRasterBand(1)
    img = Image.new('I', (rs.XSize, rs.YSize))
    draw = ImageDraw.Draw(img)

    polys, poly_map = load_shapefile_info(shp_fil)

    for ft in range(len(polys)):
        # Get the outer ring of the polygon
        coords = np.array(polys[ft][2])
        xys = geo_conv.lla_to_pixel(tif_fil, coords)[:, 0:2]
        draw.polygon(xys.flatten().tolist(), fill=counter)
        counter += 1
    del draw
    return img, poly_map, polys


class GeoJSON(object):
    def __init__(self, file_path, mode='r'):
        self.file_path = file_path
        self.mode = mode
        if mode == 'r' or mode == 'a':
            with open(file_path, mode) as fid:
                self.json_data = json.load(fid)
        else:
            self.json_data = dict()
            self.json_data['type'] = 'FeatureCollection'
            self.json_data['crs'] = dict()
            self.json_data['crs']['type'] = "name"
            self.json_data['crs']['properties'] = dict()
            self.json_data['crs']['properties']['name'] = \
                'urn:ogc:def:crs:OGC:1.3:CRS84'
            self.json_data['features'] = []

    @property
    def features(self):
        return self.json_data['features']

    def add_feature(self, feature):
        self.json_data['features'].append(feature)

    def add_features(self, features):
        self.json_data['features'].extend(features)

    def save(self, output_file, overwrite=False, indent=3):
        if os.path.exists(output_file) and not overwrite:
            raise IOError('File already exists. Set overwrite to True if you '
                          'would like to overwrite the file.')
        with open(output_file, 'w') as fid:
            json.dump(self.json_data, fid, indent=indent)


def create_feature(properties: dict, coords: np.ndarray, geometry_type: str):
    feat = dict()
    feat['type'] = 'Feature'
    feat['properties'] = properties
    feat['geometry'] = dict()
    feat['geometry']['type'] = geometry_type
    if geometry_type == 'Point' or geometry_type == 'LineString':
        feat['geometry']['coordinates'] = coords.tolist()
    else:
        feat['geometry']['coordinates'] = [coords.tolist()]
    return feat


def from_polygon_kml(kml_file, output_file=None, overwrite=True):
    if output_file is None:
        output_file = os.path.splitext(kml_file)[0] + '.geojson'

    geojson_obj = GeoJSON(output_file, 'w')

    with open(kml_file, 'rb') as fid:
        xml_data = xmltodict.parse(fid)

    if isinstance(xml_data['kml']['Document']['Placemark'], list):
        for pm in xml_data['kml']['Document']['Placemark']:
            if 'Polygon' not in pm:
                continue
            xml_coords = pm['Polygon']['outerBoundaryIs']['LinearRing'][
                'coordinates']
            coords = np.array(
                [np.array(c.split(','))[0:2].astype('float32') for c in
                 xml_coords.split(' ')])
            props = dict()
            if 'name' in pm:
                props['Name'] = pm['name']
            feat = create_feature(props, coords, 'Polygon')
            geojson_obj.add_feature(feat)
    else:
        pm = xml_data['kml']['Document']['Placemark']
        if 'Polygon' not in pm:
            print('KML does not contain Polygon.')
            return
        xml_coords = pm['Polygon']['outerBoundaryIs']['LinearRing'][
            'coordinates']
        coords = np.array(
            [np.array(c.split(','))[0:2].astype('float32') for c in
             xml_coords.split(' ')])
        props = dict()
        if 'name' in pm:
            props['Name'] = pm['name']
        feat = create_feature(props, coords, 'Polygon')
        geojson_obj.add_feature(feat)

    geojson_obj.save(output_file, overwrite=overwrite)


def from_linestring_kml(kml_file, output_file=None, overwrite=True, return_json_obj=False):
    if output_file is None:
        output_file = os.path.splitext(kml_file)[0] + '.geojson'

    geojson_obj = GeoJSON(output_file, 'w')

    with open(kml_file, 'rb') as fid:
        xml_data = xmltodict.parse(fid)

    if isinstance(xml_data['kml']['Document']['Placemark'], list):
        for pm in xml_data['kml']['Document']['Placemark']:
            if 'LineString' not in pm:
                continue
            xml_coords = pm['LineString']['coordinates']
            coords = np.array(
                [np.array(c.split(','))[0:2].astype('float32') for c in
                 xml_coords.split(' ')])
            props = dict()
            if 'name' in pm:
                props['Name'] = pm['name']
            feat = create_feature(props, coords, 'LineString')
            geojson_obj.add_feature(feat)
    else:
        pm = xml_data['kml']['Document']['Placemark']
        if 'LineString' not in pm:
            print('KML does not contain LineString.')
            return
        xml_coords = pm['LineString']['coordinates']
        coords = np.array(
            [np.array(c.split(','))[0:2].astype('float32') for c in
             xml_coords.split(' ')])
        props = dict()
        if 'name' in pm:
            props['Name'] = pm['name']
        feat = create_feature(props, coords, 'LineString')
        geojson_obj.add_feature(feat)

    if return_json_obj:
        return geojson_obj

    geojson_obj.save(output_file, overwrite=overwrite)
