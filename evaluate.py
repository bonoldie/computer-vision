import sys
import os

from utils.visibility import load_visibility, visualize_visibility

import open3d as o3d
from matplotlib import pyplot as plt
import cv2

import numpy as np
from pathlib import Path

from loguru import logger
logger.add('logs/log_{time}.log', compression='zip', rotation='1 hour')

matches_savepath = os.path.join(os.getcwd(), 'matches')

distances_to_check = [3,5,7]

if __name__ == '__main__':

    vis_map_path = os.path.join('downloads/dante_dataset/dante_dataset/Visibility.txt')
    logger.info(f'Loading visibility map: {vis_map_path}')
    vis_map = load_visibility(vis_map_path)

    _SAM1005_vis = vis_map['_SAM1005.JPG']

   
    distances_to_check.insert(0, 0)
    distances_boundaries = [[distances_to_check[i], distances_to_check[i+1]] for i in range(len(distances_to_check) - 1)]

    logger.info('Masking matches based on visibility of reference image')

    ##############
    ## LiftFeat ##
    ##############

    reference__SAM1005_matches = np.load(os.path.join(matches_savepath, 'LiftFeat', 'reference__SAM1005_matches.npy'))
    target__SAM1007_matches = np.load(os.path.join(matches_savepath, 'LiftFeat', 'target__SAM1007_matches.npy'))

    masking_result_LF = {}

    for dist in distances_to_check:
        masking_result_LF[dist] = []
     

    logger.info('LiftFeat matches')
    logger.info(f'Total matches: {len(reference__SAM1005_matches)}')

    for ref_match in reference__SAM1005_matches:
        for vis_point in _SAM1005_vis:
            norm = np.linalg.norm(ref_match - np.array([vis_point['w'], vis_point['h']]))
            
            for dist in distances_boundaries:    
                if norm >= dist[0] and norm <= dist[1]:
                    masking_result_LF[dist[1]].append(ref_match)

    for dist_boundary in distances_boundaries:
        logger.success(f'Matches inside range ({dist_boundary[0]} - {dist_boundary[1]}): {len(masking_result_LF[dist_boundary[1]])}')

    


    #########
    ## RDD ##
    #########
 
    reference__SAM1005_matches = np.load(os.path.join(matches_savepath, 'RDD', 'reference__SAM1005_matches.npy'))
    target__SAM1007_matches = np.load(os.path.join(matches_savepath, 'RDD', 'target__SAM1007_matches.npy'))

    masking_result_RDD = {}

    for dist in distances_to_check:
        masking_result_RDD[dist] = []
        
    logger.info('RDD matches')
    logger.info(f'Total matches: {len(reference__SAM1005_matches)}')

    for ref_match in reference__SAM1005_matches:
        for vis_point in _SAM1005_vis:
            norm = np.linalg.norm(ref_match - np.array([vis_point['w'], vis_point['h']]))
            
            for dist in distances_boundaries:    
                if norm >= dist[0] and norm <= dist[1]:
                    masking_result_RDD[dist[1]].append(ref_match)

    for dist_boundary in distances_boundaries:
        logger.success(f'Matches inside range ({dist_boundary[0]} - {dist_boundary[1]}): {len(masking_result_RDD[dist_boundary[1]])}')

    



