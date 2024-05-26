#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import utm
import statistics
import time
import logging
import sys

import numpy as np
import pandas as pd
import geopandas as gpd


import points as rd_points
import kernels as rd_kern
import projector as projector
import geotiff as geotiff
import geojson as geojson

from shapely import wkt
from shapely.geometry import Point, Polygon
from geopy import distance


from typing import List  # , Dict
CoordList = List[List[float]]

# logging.basicConfig(level=logging.INFO,
#                     format="%(asctime)s [%(levelname)s] %(message)s",
#                     datefmt='%m-%d %H:%M',
#                     handlers=[logging.FileHandler("debug.log"),
#                               logging.StreamHandler(sys.stdout)])


class ClusterFlares:
    """
    Docstring goes here
    """


    def __init__(self,
                 df_file=None,
                 output_dir='./',
                 dpp_value=0.006,
                 geojson_outline=None,
                 perc_valid=0.01,
                 verbose=True,
                 log_file=False):
        """
        Docstring goes here
        """

        if log_file:
            print('Logging flare cluster info to debug.log')
            logging.basicConfig(level=logging.INFO,
                                format="%(asctime)s [%(levelname)s] %(message)s",
                                datefmt='%m-%d %H:%M',
                                handlers=[logging.FileHandler("debug.log"),
                                          logging.StreamHandler(sys.stdout)])

        else:
            logging.basicConfig(stream=sys.stdout,
                                level=logging.INFO,
                                format="%(asctime)s [%(levelname)s] %(message)s",
                                datefmt='%m-%d %H:%M')




        self.verbose = verbose
        self.output_dir = output_dir
        self.dpp_value = dpp_value
        self.perc_valid = perc_valid
        self.mainDF = self.prepare_dataframe(df_file)

        output_file = df_file.replace('.csv', '_clusters_v2.json')
        self.output_file = self.output_dir + output_file
        self.basename = os.path.splitext(os.path.basename(df_file))[0]

        # heatmap
        self.csv_basename = os.path.splitext(os.path.basename(df_file))[0]
        # _heatmap.tif
        self.createHeatmap()

        self.refinery_polys = self.makeRefineryPolys(geojson_outline)

        if isinstance(self.mainDF, pd.DataFrame):
            pass
        else:
            self.mainDF = pd.read_csv(self.mainDF)

        self.num_collects, self.num_valid = self.setCollectionNumbers()
        self.lonlats, self.idxs, self.oob_idxs = self.filter_lonlats_indices()

        if self.verbose:
            logging.info('Num valid: {}'.format(self.num_valid))
            logging.info('# of lonlats: {}'.format(len(self.lonlats)))
            logging.info('In bounds indices: {}'.format(len(self.idxs)))
            logging.info('Out of bounds indices: {}'.format(len(self.oob_idxs)))

        self.intensities = self.mainDF['frp'].values[self.idxs]
        if self.verbose:
            # intensity_str = 'Intensities: {}\n'.format(len(self.intensities)), self.intensities, '\n' + '=' * 100
            intensity_str = 'Intensities # : {}\n'.format(len(self.intensities)) + '=' * 100
            logging.info(intensity_str)
        self.filteredDF = self.filter_oob_idx_from_df()

        hc_start = time.time()

        if len(self.lonlats) == 1:
            self.clusters = np.array(self.lonlats)
            self.cluster_inds = [np.array([0])]
        else:
            self.clusters, self.cluster_inds = rd_points.cluster_hierarchical(self.lonlats, dpp_value)


        if self.verbose:
            logging.info('{} seconds for hierarchical clustering'.format(time.time()-hc_start))
            # logging.info('HC length: ',len(self.clusters), ' || HC type: ', type(self.clusters), self.clusters, ' || Cluster inds: ', self.cluster_inds, ' || C-Ind type: ', type(self.cluster_inds),'\n')

        self.clstr_points_pairs, self.threshholded_clusters = self.pairClusterWithPoints(self.clusters, self.cluster_inds, points_threshhold=3)
        # logging.info('Len hierarchical clusters: {}'.format(len(self.clstr_points_pairs)))
        self.clstr_points_pairs = [pair for pair in self.clstr_points_pairs if pair[1]]
        # logging.info('Len after filtering empties: {}'.format(len(self.clstr_points_pairs)))

        self.cluster_inds = [idx_sublist[1] for idx_sublist in self.clstr_points_pairs]

        self.cluster_points_d = self.getClusterPointsDict(self.clstr_points_pairs, self.filteredDF)

        self.cluster_df_index_pairs = self.pair_df_idx_w_cluster()

        # hcluster_polygon_basename = os.path.splitext(os.path.basename(self.output_file))[0]
        # hierarchical_cluster_polygon_file = output_dir + hcluster_polygon_basename + '_hierarchical_poly.json'
        # cluster_polygon_json = geojson.GeoJSON(hierarchical_cluster_polygon_file, mode='w')

        if self.verbose:
            logging.info('# of hierarchical clusters: {}'.format(len(self.clusters)))
            logging.info('# of threshholded clusters: {}'.format(len(self.threshholded_clusters)))

        if len(self.lonlats) == 1:
            gmm_centroids = [self.lonlats]
            self.gm_idxs = [np.array([0])]


        else:
            self.gmix = self.make_gmm(self.lonlats, len(self.clusters))
            # self.gm_idxs = self.getGMMIdxs3(self.lonlats, self.gmix)
            self.gm_idxs = self.getGMMIdxs4(self.lonlats, self.gmix)
            gmm_centroids = [[self.gmix.means_[i][0], self.gmix.means_[i][1]] for i in range(len(self.gmix.means_))]



        self.gmm_cluster_points_lol = [[gmm_centroids[i], list(self.gm_idxs[i])] for i in range(len(gmm_centroids))]
        self.gmm_cluster_points_lol = [pair for pair in self.gmm_cluster_points_lol if pair[1]]
        self.gmm_cluster_points_lol = [pair for pair in self.gmm_cluster_points_lol if len(pair[1]) > 4]

        if self.verbose:
            # logging.info('Len gmm clusters: {}'.format(len(self.gmix.means_)))
            logging.info('Len gmm clusters: {}'.format(len(gmm_centroids)))
            logging.info('Len gmm cluster after filtering: {}'.format(len(self.gmm_cluster_points_lol)))
        # logging.info(self.gmm_cluster_points_lol)
        count=0
        for pair in self.gmm_cluster_points_lol:
            count= count+len(pair[1])
        logging.info(count)

        # logging.info('='*100+'\nGMM Points Pair List:')
        # logging.info(gmm_cluster_points_lol)

        self.gmm_cluster_point_dict = self.getClusterPointsDict(self.gmm_cluster_points_lol, self.filteredDF)

        # logging.info(self.gmm_cluster_point_dict)

        # logging.info('='*30+'\nGMM Centroids:')
        # logging.info(gmm_centroids)

        # logging.info('='*30)
        # logging.info(len(self.gm_idxs))
        # logging.info(100*'='+'\n'+'GMM Indices: {}'.format(self.gm_idxs))
        # logging.info(100*'=')

    def spinning_cursor(self,):
        """
        Helper function that yields a spinning
        cursor. Used to provide visual feedback
        when a loop is running.
        """
        while True:
            for cursor in '|/\\':
                yield cursor


    def prepare_dataframe(self, df_file):
        """
        Load a CSV file into a Pandas dataframe.
        Use Shapely to convert the string GMTCO point
        into a shapely geometry object. Add a latitude and
        longitude column to the dataframe based off the
        Shapely geometry point.

        Args:
            df_file: a CSV file

        Returns:
            mainDF: a Pandas dataframe
        """
        mainDF = pd.read_csv(df_file)
        if len(mainDF.index) == 0:
            logging.info('Empty Dataframe')
            logging.info('Exiting with no data to cluster.')
            exit()

        mainDF['gmtco_point'] = mainDF['gmtco_point'].apply(wkt.loads)
        mainDF = self.addLatLon2Dataframe(mainDF)
        if self.verbose:
            logging.info('Length of dataframe: ')
            logging.info(len(mainDF['gmtco_point'].to_list()))
            logging.info('=' * 100)
        return mainDF


    def addLatLon2Dataframe(self, mainDF):
        """
        Take a dataframe with a column of Shapely geometry points
        and use that column to add latitude and longitude columns as
        floats.

        Args:
            mainDF: a Pandas dataframe with a Shapely points geometry column

        Returns:
            mainDF: the dataframe with lat and lon columns added
        """
        gdf = gpd.GeoDataFrame(mainDF, geometry='gmtco_point')
        mainDF['latitude'] = gdf.geometry.y
        mainDF['longitude'] = gdf.geometry.x
        return mainDF


    def define_gkern(self):
        """

        Args:

        Returns:
        """
        gkern = rd_kern.gaussian_2d(shape=(5, 5), sigma=0.8)
        gkern /= gkern.sum()
        return gkern


    def createHeatmap(self,):
        """
        Make and export a heatmap in .TIF format

        Args:

        Returns:
        """
        gkern_value = self.define_gkern()
        heatmap_outpt = self.output_dir + self.csv_basename + '_heatmap.tif'
        values = np.ones(len(self.mainDF))
        swir_img, swir_weights = projector.grid_resample_mercator_python_1d(values, self.mainDF['longitude'].values,
                                                                            self.mainDF['latitude'].values,
                                                                            self.dpp_value,
                                                                            kernel=gkern_value.astype('float32'))

        gt = [self.mainDF['longitude'].values.min(), self.dpp_value, 0, self.mainDF['latitude'].values.max(), 0, - self.dpp_value]
        if self.verbose:
            logging.info('\nWriting heatmap to: {}\n'.format(heatmap_outpt))
        geotiff.write_new_geotiff(heatmap_outpt, swir_weights, 1, 4326, 6, gt, {})


    def makeRefineryPolys(self, outlne):
        """
        Take in a GeoJson file (or a dictionary) and return
        a list of Polygon objects that define boundaries/borders
        which will be used to exclude points from further processing.

        Args:
            outlne: A geojson object or file that defines a polygon
                    with borders. Used to exclude lat-lon points outside
                    the polygon.

        Returns:
             refinery_plys: A list of Shapely Polygon objects. Multiple
                            polygons are possible to account for islands, etc.
        """
        refinery_plys = []

        if outlne:
            # logging.info('\nType outline: {}\n'.format(type(outlne)))
            if isinstance(outlne, dict):
                refinery_plys.append(Polygon(np.array(outlne['geometry']['coordinates']).squeeze()))
            else:
                with open(outlne) as fid:
                    json_data = json.load(fid)
                for feat in json_data['features']:
                    # logging.info('\nFeature: {}\n'.format(feat))
                    # logging.info('\nFeature type: {}\n'.format(feat['geometry']['type']))
                    if feat['geometry']['type'] == 'MultiPolygon':
                        for coord_lst in feat['geometry']['coordinates']:
                            refinery_plys.append(Polygon(np.array(coord_lst).squeeze()))
                    else:
                        refinery_plys.append(Polygon(np.array(feat['geometry']['coordinates']).squeeze()))
            return refinery_plys
        else:
            return refinery_plys



    def setCollectionNumbers(self,):
        """
        Determines the number of unique points collected in the
        dataset. Also uses the perc_valid parameter to define a minimum
        number of points required to consider a cluster valid.

        Args:

        Returns:
            num_collects: Number of unique points observed
            num_valid: Threshhold minimum number of points needed
                       to create a valid cluster
        """
        u_collects = np.unique(self.mainDF['FID'].values)
        num_collects = len(u_collects)
        num_valid = int(num_collects * self.perc_valid)
        if self.verbose:
            logging.info('\n# Collects: {}'.format(num_collects))
            logging.info('# Valid: {}\n'.format(num_valid))
        return num_collects, num_valid


    def longitudeCheck(self):
        """
        Check that the dataframe has a longitude column form
        the GMTCO Points conversion. If not use the 'site_point'
        column to make one. Return a nested list of longitude-latitude
        points to be used for clustering.

        Args:

        Returns:
            lonlats: A nested np.array of longitude-latitude points of type float.
        """
        if 'longitude' not in self.mainDF:
            lonlats = np.zeros((len(self.mainDF), 2))
            for ix in range(len(self.mainDF)):
                lonlats[ix] = np.array(wkt.loads(self.mainDF['site_point'][ix]).xy).T[0]
        else:
            lonlats = np.hstack((self.mainDF['longitude'].values[:, None], self.mainDF['latitude'].values[:, None]))
        if self.verbose:
            latlon_msg = 'Lat-Long # Conversion: {}\n'.format(len(lonlats)) + '=' * 100
            logging.info(latlon_msg)
        return lonlats


    def filter_lonlats_indices(self):
        """
        Removes lon-lat points from the array if they
        are not within the bounds of the refinery Polygons
        loaded from Geojson. Adds the indices of these
        in-bounds points to an array. Also tracks out
        of bounds indices.

        Args:

        Returns:
            lonlats: A filtered np.array of lon-lat points
                     within the bounds of the Polygon(s)
            idxs: Indices for the dataframe points inside the
                  Polygon(s)
            oob_idxs: Indices of points outside the bounds of the
                      Polygon(s) that will be filtered
        """
        lonlats = self.longitudeCheck()
        idxs = self.filterIdxs(lonlats)
        oob_idxs = list(set([i for i in range(len(lonlats))]) - set(idxs))
        lonlats = np.array(lonlats)
        lonlats = lonlats[np.array(idxs)]
        if self.verbose:
            oob_msg = 'Out of Bounds Points: {}\n'.format(len(oob_idxs)) + '\n' + '=' * 100
            logging.info(oob_msg)
        return lonlats, idxs, oob_idxs


    def filterIdxs(self, LonLats: list):
        """
        Helper function that checks the lon-lat points
        to see if they are within the bounds of a Shapely
        Polygon(s)

        Args:
            lonlats: list of longitude-latitude points

        Returns:
            indices: the indices of the lon-lat points that
                     are within the bounds of the Shapely Polygon(s)
        """
        # indices = [i for i in range(len(LonLats))]
        indices = []
        # logging.info('# of Lat-Lons to check: {}'.format(len(LonLats)))
        if self.refinery_polys:
            for ix in range(len(LonLats)):
                for poly in self.refinery_polys:
                    if Point(LonLats[ix]).within(poly):
                        indices.append(ix)
                        break
        else:
            indices = [ix for ix in range(len(LonLats))]
        # in_bounds_indices = np.array(in_bounds_indices)
        return indices


    def filter_oob_idx_from_df(self):
        """
        Creates a filtered dataframe from the original that
        removes entries/rows that contain lat-lon points which
        were outside the bounds of the Shapely Polygon(s)

        Args:

        Returns:
            filterDF: filtered dataframe with out of bounds indices dropped
        """
        filterDF = self.mainDF.drop(index=self.oob_idxs)
        filterDF = filterDF.reset_index(drop=True)
        return filterDF


    def set_json_output_objects(self):
        """
        Initializes the geojson objects and output filenames that will be used for
        recording the clusters:

        Args:

        Returns:
            out_point_json: output path for json file that will contain the cluster
                            centroid as a point
            out_poly_json: output path for json file that will contain a bounding box
                           polygon for the cluster
            output_json_pt: json object that contains the cluster centroid as a geographic
                            coordinate point
            output_json_poly: json object that contains a bounding box polygon for each cluster
        """
        self.out_point_json = self.output_dir + os.path.splitext(os.path.basename(self.output_file))[0] + '_pt.json'
        self.out_poly_json = self.output_dir + os.path.splitext(os.path.basename(self.output_file))[0] + '_poly.json'
        self.output_json_pt = geojson.GeoJSON(self.out_point_json, mode='w')
        self.output_json_poly = geojson.GeoJSON(self.out_poly_json, mode='w')

        if len(self.lonlats) <= 0:
            self.output_json_pt.save(self.out_point_json, overwrite=True)
            self.output_json_poly.save(self.out_poly_json, overwrite=True)
            logging.info('No lat-lon points were detected in dataframe file!')
            logging.info('Exiting...')
            exit()


    def getClusterPointsDict(self, cluster_point_pair_lst, main_dataframe, utm_zone=None):
        """
        Create a dictionary containing key information about each cluster.

        Args:
            cluster_point_pair_lst: Nested list where the first item is the lon-lat centroid
                                    for a cluster, and the second item is the indices of the
                                    associated points in the dataframe for that cluster
            main_dataframe: pandas dataframe with lon-lat points
            utm_zone: UTM coordinate zone  # Delete this it is out of date

        Returns:
            clstr_pts_dict: A dictionary where the key is a cluster number and the value is a dictionary
                            containing various items associated with that cluster- centroid, constituent points, etc.
        """
        for pair in cluster_point_pair_lst:
            if not pair[1]:
                logging.info('No points associated with GMM cluster centroid:')
                logging.info(pair, '\n')
        clstr_pts_dict = {}
        cluster_no = 0
        for pair in cluster_point_pair_lst:
            clstr_pts_dict[cluster_no] = {'lon_lat_centroid': pair[0], 'points_df_idxs': pair[1], 'utm_zone': utm_zone}
            cluster_constituent_points = []
            # cluster_constituent_utms = []
            for idx in pair[1]:
                row = main_dataframe.loc[idx]
                lat = row['latitude']
                lon = row['longitude']
                # utm_point = list(utm_list[idx])
                cluster_constituent_points.append([lon, lat])
                # cluster_constituent_utms.append(utm_point)
            clstr_pts_dict[cluster_no]['points_lon_lats'] = cluster_constituent_points
            # clstr_pts_dict[cluster_no]['points_utms'] = cluster_constituent_utms
            cluster_no += 1
        clstr_pts_dict = self.addSpatialStats(clstr_pts_dict)
        return clstr_pts_dict


    def addSpatialStats(self, clstr_dict:dict):
        """
        Helper function that adds spatial information to the cluster-points dictionary.

        Args:
            clstr_dict: A dictionary where the key is a cluster number and the value is a dictionary
                        containing various items associated with that cluster- centroid, constituent points, etc.

        Returns:
            clstr_dict: The same dictionary but with spatial distance info added
        """
        for k,v in clstr_dict.items():
            # centroid_utm = v['utm_centroid']
            centroid_lola = v['lon_lat_centroid']
            # point_utm_lst = v['points_utms']
            point_lonlat_lst = v['points_lon_lats']
            if not point_lonlat_lst:
                logging.info('Empty lonlat list: ')
                logging.info(k,' || ',v)
            mean2centroid, centroid_distances = self.getDistances2Centroid([centroid_lola[1], centroid_lola[0]], point_lonlat_lst)
            distance_stdev = self.getStandardDeviation(mean2centroid, centroid_distances)
            clstr_dict[k]['mean_meters_to_centroid'] = mean2centroid
            clstr_dict[k]['stddev_meters_to_centroid'] = distance_stdev
            clstr_dict[k]['distances_to_centroid'] = centroid_distances
            try:
                cluster_polygon = self.defineClusterPolygon(v['points_lon_lats'])
            except Exception as e:
                logging.info(e)
                logging.info('Points lonlats:')
                logging.info('\n',v['points_lon_lats'], '\n')
            clstr_dict[k]['cluster_polygon'] = cluster_polygon
        return clstr_dict


    def getDistances2Centroid(self, centroid_lola_coord, points_lola_coords:CoordList):
        """
        Take a centroid lon-lat coordinate point and list of coordinate points and calculate
        the distance to the centroid for each. Uses the geopy distance function that accounts
        for non-Euclidean space

        Args:
            centroid_lola_coord: cluster centroid coordinate in decimal degrees
            points_lola_coords: Nested list of lon-lat coordinate points in decimal degrees

        Returns:
            mean2centroid: the mean distance to the cluster centroid for all associated points
            centroid_distances: A list of distances to the cluster centroid for each associated point
        """
        centroid_distances = []
        for p in points_lola_coords:
            distance2center = distance.distance(centroid_lola_coord, [p[1], p[0]]).m
            centroid_distances.append(distance2center)
        try:
            mean2centroid = sum(centroid_distances) / len(centroid_distances)
        except ZeroDivisionError as e:
            logging.info('No associated points: {}'.format(e))
            mean2centroid = 0.0
        return mean2centroid, centroid_distances


    def getStandardDeviation(self, mean_value, values_array: list):
        """
        Calculate standard deviation for an array of values.

        Args:
            mean_value: the mean for the array of values
            values_array: an array of values

        Returns:
            stdev: standard deviation for all values to the mean
        """
        try:
            variance = sum([((d - mean_value) ** 2) for d in values_array]) / len(values_array)
        except ZeroDivisionError as e:
            variance = 0.0
            if self.verbose:
                logging.info('Couldn\'t get Std. Dev. no points in array: {}'.format(e))
        stdev = variance ** 0.5
        return stdev


    def bounding_box(self, points):
        """
        Create a bounding box using an array of coordinate points

        Args:
            points: a list pf coordinates (x,y style)

        Returns:
            bounding_box: a bounding box defined by the minimum x,y and maximum x,y points,
                          i.e. the extreme south-west and north-east points
        """
        try:
            x_coordinates, y_coordinates = zip(*points)
        except ValueError as e:
            logging.info(e)
            logging.info('Points:')
            logging.info('\n',points,'\n')
        return [[min(x_coordinates), min(y_coordinates)], [max(x_coordinates), max(y_coordinates)]]


    def defineClusterPolygon(self, lon_lat_points:list):
        """
        Defines a rectangular cluster polygon using a bounding box.

        Args:
            lon_lat_points: A list of lon-lat coordinate points

        Returns:
            cluster_polygon: np.array of coordinates defining a polygon, the start coordinate
                             must be the same as the end coordinate.
        """
        bbox = self.bounding_box(lon_lat_points)
        box_sw = bbox[0]
        box_nw = [bbox[0][0], bbox[1][1]]
        box_ne = bbox[1]
        box_se = [bbox[1][0], bbox[0][1]]
        bbox_end = bbox[0]
        cluster_polygon = np.array([box_sw, box_nw, box_ne, box_se, bbox_end])
        return cluster_polygon


    def pair_df_idx_w_cluster(self):
        """
        Pairs a cluster with a point's index in the dataframe, such that
        every point gets associated with one or more clusters. Returns this as
        a nested list, e.g. [[13, 3914], [6, 4018]] means cluster 3 has a
        constituent point at row 3914 in the dataframe and cluster 6 at row 4018.

        Args:

        Returns:
            cluster_df_index_pairs: A nested list where the first item is a
                                    cluster number and the second is the index for
                                    a row in the dataframe. E.g. [[13, 3914], [6, 4018]]
        """
        placeholder_indices = [i for i in range(len(self.mainDF))]
        cluster_ind_count = 0
        cluster_idxs = 0
        cluster_df_index_pairs = []

        for c in self.cluster_inds:
            cluster_ind_count = cluster_ind_count + len(c)
            self.cluster_points_d[cluster_idxs]['cluster_point_indices'] = list(c)
            for idx in c:
                cluster_df_index_pairs.append([cluster_idxs, idx])
                if idx in placeholder_indices:
                    placeholder_indices.remove(idx)
            cluster_idxs+=1
        for idx in placeholder_indices:
            cluster_df_index_pairs.append([None, idx])

        cluster_df_index_pairs = sorted(cluster_df_index_pairs, key=lambda x: x[1])
        sorted_clstr_idxs = [i[0] for i in cluster_df_index_pairs]
        self.mainDF['cluster_id'] = sorted_clstr_idxs

        if self.verbose:
            logging.info('Combined length of cluster indices: {}'.format(cluster_ind_count))
        return cluster_df_index_pairs


    def getRefineryBounds(self, refinery_plys):
        """
        Unused method that is used to get the boundaries of
        refinery polygons. The result was originally used
        to estimate the UTM grid zone.

        Args:
            refinery_polys: Shapely polygons defining the borders
                            of an area

        Returns:
            bounds: np.array of coordinate points that define the outline
                    of a Shapely polygon(s)
        """
        bounds = []
        for p in refinery_plys:
            b = np.array(p.boundary.xy).T
            bounds.append(b)
        return bounds


    def getUtmPolys(self, refinery_plys, utm_zone=None):
        """
        Redefines a list of lon-lat coordinate points into
        a list of the same points in the UTM coordinate system

        Args:
            refinery_plys: np.array of coordinate points defining a
                           Shapely Polygon(s) boundary
            utm_zone: UTM grid zone number

        Returns:
            utm_polys: np.array of the same polygon(s) boundary but converted
                       to using UTM coordinate system
        """
        utm_polys = []
        for p in refinery_plys:
            coords = np.array(p.boundary.xy).T
            utm_coords = np.zeros_like(coords)
            for ix in range(len(coords)):
                utm_vals = utm.from_latlon(coords[ix, 1], coords[ix, 0], utm_zone)
                utm_coords[ix] = np.array([utm_vals[0], utm_vals[1]])
            utm_polys.append(Polygon(utm_coords))
        return utm_polys


    def getLonLatMeans(self, area_bounds):
        """
        Get the mean longitude and latitude values
        for a polygon.

        Args:
            area_bounds: Array of coordinates defining a
                         polygon

        Returns:
            lon_mean: Longitude mean value
            lat_mean: Latitude mean value
        """
        area_mins = area_bounds.min(0)
        area_maxs = area_bounds.max(0)
        lla_bounds = [area_mins[0], area_mins[1], area_maxs[0], area_maxs[1]]
        lon_mean = (lla_bounds[0] + lla_bounds[2]) / 2
        lat_mean = (lla_bounds[1] + lla_bounds[3]) / 2
        return lon_mean, lat_mean


    def addIntensityStats2Json(self, frp_intensities: list, json_properties):
        """
        Take a dictionary and add various statistics about the flare/fire signal's
        spectral intensity

        Args:
            frp_intensities: list of spectral intensity values for the
                             flare/fire signals
            json_properties: A dictionary that will contain information
                             written to the flare cluster geojson output

        Returns:
            json_properties: Same dictionary with the spectral intensity stats added
        """
        json_properties['intensity_mean'] = float(statistics.mean(frp_intensities))
        json_properties['intensity_stddev'] = self.getStandardDeviation(float(statistics.mean(frp_intensities)),
                                                                        frp_intensities)
        json_properties['intensity_max'] = float(max(frp_intensities))
        json_properties['intensity_min'] = float(min(frp_intensities))
        return json_properties


    def setGeojsonProps(self,
                        mainDF=None,
                        file_basename=None,
                        num_collects=None,
                        signal_type=None,
                        cluster_num=None,
                        cluster_points=None):
        """
        Initialize a dictionary that will contain various properties
        for the cluster output geojson file.

        Args:
            mainDF: the dataframe loaded from the input CSV
            file_basename: filename that will be used to generate a unique name
            num_collects: the number of valid points collected in the dataset
            signal_type: static string value-- 'flare'
            cluster_num: the cluster's integer value (only unique to each run of the program)
            cluster_points: the coordinate points associated to the cluster

        Return:
            geojson_props: dictionary containing properties associated with the cluster
        """
        geojson_props = dict()
        geojson_props['order'] = mainDF['uuid'][0]
        geojson_props['name'] = file_basename.split(' ')[0] + '_{:04d}'.format(cluster_num)
        geojson_props['num_collects'] = int(num_collects)
        geojson_props['num_in_cluster'] = len(cluster_points)
        geojson_props['signal_type'] = signal_type
        return geojson_props


    def make_gmm(self, points_lola, cluster_num):
        """
        Runs the Expectation-Maximization algorithm
        to generate a Gaussian Mixture Model for flare clusters.

        Args:
            points_lola: Lon-Lat coordinate points from the flare collects
            cluster_num: The number of normal distributions to fit to the points.

        Returns:
            gmm: Gaussian Mixture Model from sklearn
        """
        if self.verbose:
            logging.info('Fitting GMM via EM algorithm...')
            start = time.time()
        gmm = rd_points.gaussian_mixture(points_lola, cluster_num)
        if self.verbose:
            logging.info('{} seconds for fitting GMM'.format(time.time()-start))
        return gmm


    def pairClusterWithPoints(self, clstrs:list, clstr_indices:list, points_threshhold=3):
        """
        Create a nested list that groups a cluster centroid coordinate with the
        dataframe indices of the points associated of that cluster.


        Args:
            clstrs: centroid points of clusters
            clstr_indices: List of dataframe indices for points associated with a cluster
            points_threshhold: Minimum number of points that must be associated with a cluster

        Returns:
            cluster_points_pair_lol: A nested list where the first item is the coordinate point for
                                     a cluster centroid. The second item is another list of all dataframe
                                     indices for the points associated with that cluster.
                                     E.g. [[-104.607, 38.128], [3, 7, 8, 149, 210, 211]]
            threshhold_clusters: np.array of cluster centroid points that has been threshholded. Only clusters
                                 with more than a certain number of associated points are included.
        """
        cluster_points_pair_lol = []
        threshhold_clusters = []
        for i in range(len(clstrs)):
            if len(clstr_indices[i]) > points_threshhold:
                threshhold_clusters.append(clstrs[i])
                cluster_points_pair_lol.append([list(clstrs[i]), list(clstr_indices[i])])
        return cluster_points_pair_lol, threshhold_clusters


    def getGMMIdxs3(self, latlons: list, gmm, semi_soft=False):
        """
        Get the indices of the points associated with each cluster/normal distribution
        in the Gaussian Mixture Model. Though GMMs are soft-clustering, do a hard
        assignment of each point to a single cluster by choosing the one with
        highest probability.

        Args:
            latlons: List of lat-lon coordinate points.
            gmm: Sklearn Gaussian Mixture Model object
        Returns:
            gmm_idxs: List of np.arrays where each array is
                      the indices of the lat-lon points assigned to that
                      GMM cluster
        """
        spinner = self.spinning_cursor()
        start_time = time.time()
        gmm_idxs = [[] for i in range(len(gmm.means_))]

        idx_count = 0
        for pnt in latlons:
            if self.verbose:
                logging.info('\rCalculating GMM Membership ' + next(spinner) + ' ', end='', flush=True)
            x = [pnt]
            if semi_soft:
                prob_matrix = gmm.predict_proba(x).tolist()
                prob_matrix = [float(f'v:.4f') for v in prob_matrix]
                possible_clusters = [idx for idx in range(len(prob_matrix)) if prob_matrix[idx] > 0.4]
                for clst in possible_clusters:
                    gmm_idxs[clst].append(idx_count)
            else:
                predicted_cluster = gmm.predict(x)[0]
                gmm_idxs[predicted_cluster].append(idx_count)
            idx_count += 1

        gmm_idxs = [np.array(lst) for lst in gmm_idxs]
        # gm_idxs = getGMMIdxs3(lonlats, gmix)
        if self.verbose:
            logging.info('\n{} seconds for cluster idx calculations\n'.format(time.time() - start_time))
            # logging.info(gmm_idxs)

        return gmm_idxs


    def getGMMIdxs4(self, latlons: list, gmm, semi_soft=False):
        """
        Get the indices of the points associated with each cluster/normal distribution
        in the Gaussian Mixture Model. Though GMMs are soft-clustering, do a hard
        assignment of each point to a single cluster by choosing the one with
        highest probability.

        Args:
            latlons: List of lat-lon coordinate points.
            gmm: Sklearn Gaussian Mixture Model object
        Returns:
            gmm_idxs: List of np.arrays where each array is
                      the indices of the lat-lon points assigned to that
                      GMM cluster
        """
        spinner = self.spinning_cursor()
        start_time = time.time()
        gmm_idxs = [[] for i in range(len(gmm.means_))]

        if semi_soft:
            idx_count = 0
            for pnt in latlons:
                if self.verbose:
                    logging.info('\rCalculating GMM Membership ' + next(spinner) + ' ', end='', flush=True)
                x = [pnt]
                prob_matrix = gmm.predict_proba(x).tolist()
                prob_matrix = [float(f'v:.4f') for v in prob_matrix]
                possible_clusters = [idx for idx in range(len(prob_matrix)) if prob_matrix[idx] > 0.4]
                for clst in possible_clusters:
                    gmm_idxs[clst].append(idx_count)

                idx_count += 1

        else:
            if self.verbose:
                print('\rCalculating GMM Membership ' + next(spinner) + ' ', end='', flush=True)
            all_predictions = gmm.predict(latlons)
            for i in range(len(all_predictions)):
                if self.verbose:
                    print('\rCalculating GMM Membership ' + next(spinner) + ' ', end='', flush=True)
                gmm_idxs[all_predictions[i]].append(i)


        gmm_idxs = [np.array(lst) for lst in gmm_idxs]
        # gm_idxs = getGMMIdxs3(lonlats, gmix)
        if self.verbose:
            logging.info('\n{} seconds for cluster idx calculations\n'.format(time.time() - start_time))
            # logging.info(gmm_idxs)

        return gmm_idxs
