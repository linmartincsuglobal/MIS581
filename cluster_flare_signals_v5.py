#!/usr/bin/env python
# -*- coding: utf-8 -*-


# import glob
# import json
import os
import argparse
import csv
import pickle
import time
import logging
import sys

import numpy as np
from pathlib import Path


import geojson as geojson
from flare_clusterer_dbs import ClusterFlares


# logging.basicConfig(level=logging.INFO,
#                     format="%(asctime)s [%(levelname)s] %(message)s",
#                     datefmt='%m-%d %H:%M',
#                     handlers=[logging.FileHandler("debug.log"),
#                               logging.StreamHandler(sys.stdout)])

# I/O functions below


def loadPickleModel(pkl_file:str):
    """
    Loads a pickle binary file of a trained
    sklearn model

    Args:
        pkl_file: path to a pickled classifier model

    Returns:
        model: a trained sklearn model
    """
    with open(pkl_file, 'rb') as f:
        model = pickle.load(f)
    return model


def returnClassifierPrediction(model,input_features)->str:
    """
    Takes a pretrained sklearn binary classifier and
    input features and predicts if the source is a wildfire
    or manmade

    Args:
        model: a trained SVM classifier
        input_features: an array of input features used to
                        make a prediction

    Returns:
        prediction: a string value for the classifier prediction.
                    The model predicts 0 or 1, which is then explicitly
                    turned into a string value for that integer.
    """
    cluster_classification = model.predict(input_features)[0]
    if cluster_classification == 0:
        prediction = 'wildfire'
    else:
        prediction = 'manmade'
    return prediction



def returnClassifierPrediction2(model,input_data):
    """
    Takes a pretrained sklearn binary classifier and
    input features and predicts if the source is a wildfire
    or manmade

    Args:
        model: a trained SVM classifier
        input_features: an array of input features used to
                        make a prediction

    Returns:
        prediction: a string value for the classifier prediction.
                    The model predicts 0 or 1, which is then explicitly
                    turned into a string value for that integer.
    """
    cluster_classifications = model.predict(input_data)
    cluster_classifications = list(cluster_classifications)
    cluster_classifications = ['wildfire' if i==0 else 'manmade' for i in cluster_classifications]

    return cluster_classifications



def writeClusterPoints2CSV(filepath:str, cluster_points:list):
    """
    Writes the centroid points for the hierarchical clusters
    to a CSV file

    Args:
        filepath: the output file
        cluster_points: The lat-lon coordinates of the cluster
                        centroid

    Returns:
    """
    with open(filepath, 'w') as outcsv:
        # configure writer to write standard csv file
        writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(['latitude', 'longitude'])
        writer.writerows(cluster_points)



def main(output_file,
         csv_file,
         signal_type,
         basename,
         outline,
         dpp=0.006,
         mpp=600,
         perc_valid=0.01,
         output_dir='./',
         verbose=True,
         model='./lsvc_model_001.pkl'):

    start_time = time.time()

    cluster_maker = ClusterFlares(df_file=csv_file,
                                  output_dir=output_dir,
                                  dpp_value=dpp,
                                  geojson_outline=outline,
                                  verbose=verbose,
                                  perc_valid=perc_valid)

    if verbose:
        # logging.info('Hierarchical Clusters: {}'.format(len(cluster_maker.clusters)))

        logging.info('DataFrame length: {}'.format(len(cluster_maker.mainDF)))
        # logging.info('Cluster_inds: {}'.format(len(cluster_maker.cluster_inds)))

    # hcluster_polygon_basename = os.path.splitext(os.path.basename(cluster_maker.output_file))[0]

    ## hierarchical_cluster_polygon_file = output_file.replace('.json', '_hierarchical_poly.json')

    # hierarchical_cluster_polygon_file = output_dir + hcluster_polygon_basename + '_hierarchical_poly.json'
    # cluster_polygon_json = geojson.GeoJSON(hierarchical_cluster_polygon_file, mode='w')

    '''
    ###
    point_length=0
    for k,v in cluster_maker.cluster_points_d.items():
        json_props = cluster_maker.setGeojsonProps(mainDF=cluster_maker.mainDF,
                                                   file_basename=cluster_maker.basename,
                                                   num_collects=cluster_maker.num_collects,
                                                   signal_type=signal_type,
                                                   cluster_num=k,
                                                   cluster_points=v['points_df_idxs'])
        json_props['cluster_id'] = k
        json_props['lon_lat_centroid'] = v['lon_lat_centroid']
        json_props['distance_mean'] = v['mean_meters_to_centroid']
        json_props['distance_stddev'] = v['stddev_meters_to_centroid']
        json_props['distance_max'] = max(v['distances_to_centroid'])
        json_props['distance_min'] = min(v['distances_to_centroid'])

        point_indices = v['cluster_point_indices']
        point_length = point_length+len(point_indices)
        cluster_intensities = []
        for point in point_indices:
            cluster_intensities.append(cluster_maker.mainDF['frp'][point])

        json_props = cluster_maker.addIntensityStats2Json(cluster_intensities, json_props)
        polygon_bbox = v['cluster_polygon']
        bbox_feat = geojson.create_feature(json_props, polygon_bbox, geometry_type='Polygon')
        cluster_polygon_json.add_feature(bbox_feat)

    cluster_polygon_json.save(hierarchical_cluster_polygon_file, overwrite=True)

    if verbose:
        logging.info('Point length: {}'.format(point_length))
        logging.info('Number of threshholded hierarchical clusters: ')
        logging.info(len(cluster_maker.threshholded_clusters))
        logging.info('=' * 100)

    clstr_file_path = os.path.splitext(os.path.basename(csv_file))[0]
    clstr_file_path = output_dir + clstr_file_path + '_hierarchical_clusters.csv'
    # clstr_file_path = output_dir + clstr_file_path
    writeClusterPoints2CSV(clstr_file_path, cluster_maker.threshholded_clusters)

    csv_file_new = os.path.splitext(os.path.basename(csv_file))[0]
    csv_file_new = csv_file_new + '_mod2.csv'
    csv_file_new = output_dir + csv_file_new
    cluster_maker.mainDF.to_csv(path_or_buf=csv_file_new, sep=',', na_rep='',
                                float_format=None, columns=None, header=True,
                                index=False, mode='w', encoding='utf8')
    '''

    # total_in_gm_clusters = []
    # filtered_gm_clstr_idxs = []
    classifier_model = loadPickleModel(model)

    cluster_maker.set_json_output_objects()
    # for ix in range(len(cluster_maker.gm_idxs)):
    #     point_indices = list(cluster_maker.gm_idxs[ix])
    #     point_lola_coords = [list(cluster_maker.lonlats[idx]) for idx in point_indices]
    #     mean2centroid, centroid_distances = cluster_maker.getDistances2Centroid([cluster_maker.gmix.means_[ix][1], cluster_maker.gmix.means_[ix][0]], point_lola_coords)
    #     distance_stdev = cluster_maker.getStandardDeviation(mean2centroid, centroid_distances)
    #     cluster_poly_points = [[cluster_maker.gmix.means_[ix][0] - .0003, cluster_maker.gmix.means_[ix][1] - .0003],
    #                            [cluster_maker.gmix.means_[ix][0] + .0003, cluster_maker.gmix.means_[ix][1] + .0003]]
    #     # clust_poly = defineClusterPolygon(cluster_poly_points)
    #     try:
    #         clust_poly = cluster_maker.defineClusterPolygon(point_lola_coords)
    #     except ValueError as e:
    #         clust_poly = cluster_maker.defineClusterPolygon(cluster_poly_points)
    #         if verbose:
    #             logging.info('Value Error when defining cluster polygon: {}'.format(e))
    #             logging.info('Continuing...')
    #     # logging.info(clust_poly)
    #     clust_lla = np.array([cluster_maker.gmix.means_[ix][0], cluster_maker.gmix.means_[ix][1]])  # [::-1]
    #############################################################################
    #     if len(cluster_maker.gm_idxs[ix]) >= cluster_maker.num_valid:
    #############################################################################
    #         json_props = cluster_maker.setGeojsonProps(mainDF=cluster_maker.mainDF,
    #                                                    file_basename=basename,
    #                                                    num_collects=cluster_maker.num_collects,
    #                                                    signal_type=signal_type,
    #                                                    cluster_num=ix+1,
    #                                                    cluster_points=cluster_maker.gm_idxs[ix])

    #         filtered_gm_clstr_idxs.append(cluster_maker.gm_idxs[ix])
    #         total_in_gm_clusters.append(len(cluster_maker.gm_idxs[ix]))
    #         json_props['percent_visible'] = int(len(cluster_maker.gm_idxs[ix]) / cluster_maker.num_collects * 100)
    #         json_props['lon_lat_centroid'] = list(cluster_maker.gmix.means_[ix])
    #         # json_props['utm_zone'] = utm_zone
    #         json_props['distance_min'] = min(centroid_distances)
    #         json_props['distance_max'] = max(centroid_distances)
    #         json_props['distance_mean'] = mean2centroid
    #         json_props['distance_stddev'] = distance_stdev
    #         json_props = cluster_maker.addIntensityStats2Json(list(cluster_maker.intensities[cluster_maker.gm_idxs[ix]]), json_props)
    #         classifier_input = [[mean2centroid,
    #                              distance_stdev,
    #                              json_props['intensity_mean'],
    #                              json_props['intensity_stddev']]]

    #         json_props['model_prediction'] = returnClassifierPrediction(svm_classifier,
    #                                                                     classifier_input)
    #         feat1 = geojson.create_feature(json_props, clust_lla, geometry_type='Point')
    #         cluster_maker.output_json_pt.add_feature(feat1)
    #         feat2 = geojson.create_feature(json_props, clust_poly, geometry_type='Polygon')
    #         cluster_maker.output_json_poly.add_feature(feat2)

#############################################################################################################
    batch_input_data = []
    json_prop_dict = {}
    # for k,v in cluster_maker.gmm_cluster_point_dict.items():
    for k,v in cluster_maker.dbs_cluster_points_dict.items():
        json_props = cluster_maker.setGeojsonProps(mainDF=cluster_maker.filteredDF,
                                                   file_basename=cluster_maker.basename,
                                                   num_collects=cluster_maker.num_collects,
                                                   signal_type=signal_type,
                                                   cluster_num=k,
                                                   cluster_points=v['points_df_idxs'])
        json_props['cluster_id'] = k
        json_props['lon_lat_centroid'] = v['lon_lat_centroid']
        json_props['distance_mean'] = v['mean_meters_to_centroid']
        json_props['distance_stddev'] = v['stddev_meters_to_centroid']
        json_props['distance_max'] = max(v['distances_to_centroid'])
        json_props['distance_min'] = min(v['distances_to_centroid'])


        point_indices = v['points_df_idxs']
        # point_length = point_length+len(point_indices)
        cluster_intensities = []
        for point in point_indices:
            cluster_intensities.append(cluster_maker.filteredDF['frp'][point])

        json_props = cluster_maker.addIntensityStats2Json(cluster_intensities, json_props)

        # classifier_input = [[json_props['distance_mean'],
        #                      json_props['distance_stddev'],
        #                      json_props['intensity_mean'],
        #                      json_props['intensity_stddev'],
        #                      json_props['distance_max'],
        #                      json_props['distance_min'],
        #                      json_props['intensity_max'],
        #                      json_props['intensity_min']]]

        classifier_input = [json_props['distance_mean'],
                            json_props['distance_stddev'],
                            json_props['intensity_mean'],
                            json_props['intensity_stddev'],
                            json_props['distance_max'],
                            json_props['distance_min'],
                            json_props['intensity_max'],
                            json_props['intensity_min']]


        json_props['points_lon_lats'] = v['points_lon_lats']

        # batch_input_data[k] = classifier_input
        batch_input_data.append(classifier_input)
        # json_props['model_prediction'] = returnClassifierPrediction(classifier_model,
        #                                                             classifier_input)
        json_prop_dict[k] = json_props
        # json_prop_dicts.append(json_props)

    start_p = time.time()
    batch_predictions = returnClassifierPrediction2(classifier_model, batch_input_data)
    end_p = time.time()

    if verbose:
        logging.info('{} seconds for cluster batch predictions'.format(end_p-start_p))

    for k,v in json_prop_dict.items():
        # for i in range(len(json_prop_dicts)):
        cluster_class = batch_predictions[k]
        # cluster_class = batch_predictions[i]
        v['model_prediction'] = cluster_class
        # json_prop_dicts[i]['model_prediction'] = cluster_class
        # clust_poly = defineClusterPolygon(cluster_poly_points)

        try:
            clust_poly = cluster_maker.defineClusterPolygon(v['points_lon_lats'])
            # clust_poly = cluster_maker.defineClusterPolygon(json_prop_dicts[i]['points_lon_lats'])
        except ValueError as e:
            cluster_poly_points = [[v['lon_lat_centroid'][0] - .0003, v['lon_lat_centroid'][1] - .0003],
                                   [v['lon_lat_centroid'][0] + .0003, v['lon_lat_centroid'][1] + .0003]]
            # cluster_poly_points = [[json_prop_dicts[i]['lon_lat_centroid'][0] - .0003, json_prop_dicts[i]['lon_lat_centroid'][1] - .0003],
            #                        [json_prop_dicts[i]['lon_lat_centroid'][0] + .0003, json_prop_dicts[i]['lon_lat_centroid'][1] + .0003]]
            clust_poly = cluster_maker.defineClusterPolygon(cluster_poly_points)
            if verbose:
                logging.info('Value Error when defining cluster polygon: {}'.format(e))
                logging.info('Continuing...')

        v.pop('points_lon_lats')
        # json_prop_dicts[i].pop('points_lon_lats')

        feat1 = geojson.create_feature(v, np.array(v['lon_lat_centroid']), geometry_type='Point')
        # feat1 = geojson.create_feature(json_prop_dicts[i], np.array(json_prop_dicts[i]['lon_lat_centroid']), geometry_type='Point')

        cluster_maker.output_json_pt.add_feature(feat1)
        feat2 = geojson.create_feature(v, clust_poly, geometry_type='Polygon')
        # feat2 = geojson.create_feature(json_prop_dicts[i], clust_poly, geometry_type='Polygon')
        cluster_maker.output_json_poly.add_feature(feat2)



#############################################################################################################


    cluster_maker.output_json_pt.save(cluster_maker.out_point_json, overwrite=True)
    cluster_maker.output_json_poly.save(cluster_maker.out_poly_json, overwrite=True)


    idx_total=0
    # for lst in cluster_maker.gm_idxs:
    for lst in cluster_maker.db_idxs:
        for idx in lst:
            idx_total+=1

    if verbose:
        logging.info('GMM Member Idx totals: {}'.format(idx_total))
        logging.info('# of Lon-Lats: {}'.format(len(cluster_maker.lonlats)))
        # logging.info('Total in GM clusters: {}'.format(len(total_in_gm_clusters)))
        # logging.info('# of GMM clusters:', len(cluster_maker.gmix.means_))

    # filtered_gm_clstr_idxs = [list(a) for a in filtered_gm_clstr_idxs]
    # fltr_clstr_d = {}
    # clstr_num = 0
    # for c in filtered_gm_clstr_idxs:
    #     fltr_clstr_d[clstr_num] = c
    #     clstr_num+=1
    logging.info('\n{} seconds for runtime\n'.format(time.time() - start_time))

    logging.info('\nDone!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=('Generate point and poly geojsons for flare data'))
    parser.add_argument('-f', '--flare_file',
                        help='CSV with NASA flare data',
                        required=True)
    parser.add_argument('-b', '--bound',
                        help='Geojson defining boundaries for area being considered',)
    parser.add_argument('-o', '--out',
                        help='Output directory for cluster files',
                        default='./cluster_output')
    parser.add_argument('-m', '--model',
                        help='PKL binary classifier file to load model from',
                        default='./lsvc_model_001.pkl')
    parser.add_argument('-p', '--percent_valid',
                        default=0.0025,
                        help=('Threshhold for Gaussian Mixture Model clustering. '
                              'A cluster must be composed of a minimum fraction of '
                              'total points in the dataframe to be considered valid.'))
    parser.add_argument('-v', '--verbose',
                        action='store_false',
                        help='Print feedback to terminal. Default is true, option flag silences.')
    parser.add_argument('-l', '--logging',
                        action='store_true',
                        help='Write info to log file. Default is false, option flag writes to file.')
    args = parser.parse_args()


    file = args.flare_file
    output_directory = args.out
    model_file = args.model

    basename = os.path.splitext(os.path.basename(file))[0]
    full_path = str(Path.cwd().joinpath(output_directory))
    if not full_path.endswith(os.path.sep):
        full_path += os.path.sep

    if not os.path.isdir(full_path):
        os.makedirs(full_path)


    if args.logging:
        # print('Logging flare cluster info to debug.log')
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s",
                            datefmt='%m-%d %H:%M',
                            handlers=[logging.FileHandler(full_path+'debug.log'),
                                      logging.StreamHandler(sys.stdout)])

    else:
        logging.basicConfig(stream=sys.stdout,
                            level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s",
                            datefmt='%m-%d %H:%M')


    # files = sorted(glob.glob(r'F:\projects\veridian\global_refinery_monitoring\signal_detection_real_names\*.csv'))
    signal_type = 'flare'
    # input_dir = r'./'
    # outline = r"./Permian_Basin.geojson"
    if args.bound:
        outline = args.bound
    else:
        outline = None
    # file = r"./nasa_fire_10012021_03_03_2022.csv"

    logging.info(full_path)

    # Path(output_directory+basename).mkdir(parents=True, exist_ok=True)
    # output_directory = output_directory + basename

    output_file = file.replace('.csv', '_clusters_v2.json')
    output_file = full_path + output_file


    main(output_file,
         file,
         signal_type,
         basename,
         outline,
         perc_valid=float(args.percent_valid),
         output_dir=full_path,
         verbose=args.verbose,
         model=model_file)
