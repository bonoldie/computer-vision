import sys
import os
from pathlib import Path

import matplotlib.pyplot

from utils.visibility import *
from utils.view import *

import numpy as np
from scipy.spatial.distance import cdist

import open3d as o3d
import matplotlib
matplotlib.use('TkAgg')

import cv2

from loguru import logger
logger.add('logs/log_{time}.log', compression='zip', rotation='1 hour')

matches_savepath = os.path.join(os.getcwd(), 'matches')

distances_to_check = [3,5,7,15,30,50,100,200]

if __name__ == '__main__':
    
    raw_dataset_dir = os.path.join("downloads", "dante_dataset", "dante_dataset", "photos")
    ref_image_name = "_SAM1005.JPG"
    target_image_name = "_SAM1007.JPG"

    ref_image = cv2.imread(os.path.join(raw_dataset_dir, ref_image_name))
    target_image = cv2.imread(os.path.join(raw_dataset_dir, target_image_name))

    vis_map_path = os.path.join('downloads/dante_dataset/dante_dataset/Visibility.txt')
    logger.info(f'Loading visibility map: {vis_map_path}')
    vis_map = load_visibility(vis_map_path)

    ref_vis = vis_map[ref_image_name]
    ref_vis_array = convert_coords(np.asarray([*map( lambda vis_entry: [vis_entry['w'], vis_entry['h']], ref_vis)]), width=ref_image.shape[1], height=ref_image.shape[0] )
    
    target_vis = vis_map[target_image_name]

    # distances_to_check.insert(0, 0)
    # distances_boundaries = [[distances_to_check[i], distances_to_check[i+1]] for i in range(len(distances_to_check) - 1)]

    logger.info('Masking matches based on visibility of reference image')

    masking_result =  dict([['RoMa', {}], ['Mast3r', {}], ['LiftFeat', {}], ['RDD', {}]])

    ##########
    ## RoMa ##
    ##########

    # load matches
    reference__SAM1005_matches_RoMa = np.load(os.path.join(matches_savepath, 'RoMa', 'reference__SAM1005_matches.npy'))
    target__SAM1007_matches_RoMa  = np.load(os.path.join(matches_savepath, 'RoMa', 'target__SAM1007_matches.npy'))

    # view_2D_matches(reference__SAM1005_matches_RoMa[:, :],"".join([ref_image_name, '_RoMa_matches']), shape=ref_image.shape)# 
    # view_2D_matches(target__SAM1007_matches_RoMa[:, :],"".join([target_image_name, '_RoMa_matches']), shape=target_image.shape)# 

    # [-1,1] -> [0,1]
    reference__SAM1005_matches_RoMa = (reference__SAM1005_matches_RoMa + 1)/2

    reference__SAM1005_matches_RoMa_tmp = np.copy(reference__SAM1005_matches_RoMa)
    
    reference__SAM1005_matches_RoMa[:, 0] = reference__SAM1005_matches_RoMa_tmp[:, 1] * ref_image.shape[1]
    reference__SAM1005_matches_RoMa[:, 1] = (1 - reference__SAM1005_matches_RoMa_tmp[:, 0]) * ref_image.shape[0]    

    # view_2D_matches(reference__SAM1005_matches_RoMa, "".join([target_image_name, '_RoMa_matches']), shape=ref_image.shape, bg=ref_image)

    dist = cdist(reference__SAM1005_matches_RoMa, ref_vis_array)
    masking_result['RoMa'] = { 
        'reference_matches': reference__SAM1005_matches_RoMa,
        'target_matches': target__SAM1007_matches_RoMa,
        **{ dist_boundary: {'mask': np.any(dist <= dist_boundary, axis=1), 'masked_matches': reference__SAM1005_matches_RoMa[np.any(dist <= dist_boundary, axis=1)] } for dist_boundary in distances_to_check}
    } 

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit(0)

    ##############
    ## Mast3r ##
    ##############
    reference__SAM1005_matches_Mast3r = np.load(os.path.join(matches_savepath, 'Mast3r', 'reference__SAM1005_matches.npy'))
    target__SAM1007_matches_Mast3r = np.load(os.path.join(matches_savepath, 'Mast3r', 'target__SAM1007_matches.npy'))

    # 
    # view_2D_matches(target__SAM1007_matches_Mast3r,"".join([target_image_name, '_Mast3r_matches']), shape=target_image.shape)

    dist = cdist(reference__SAM1005_matches_Mast3r, ref_vis_array)    
    masking_result['Mast3r'] = {
        'reference_matches': reference__SAM1005_matches_Mast3r,
        'target_matches': target__SAM1007_matches_Mast3r, 
        **{dist_boundary: {'mask': np.any(dist <= dist_boundary, axis=1), 'masked_matches': reference__SAM1005_matches_Mast3r[np.any(dist <= dist_boundary, axis=1)] } for dist_boundary in distances_to_check}
    }


    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit(0)

    # view_2D_matches([],"".join([ref_image_name]), shape=ref_image.shape, bg=ref_image)
    # view_2D_matches(ref_vis_array,"".join([ref_image_name]), shape=ref_image.shape, bg=ref_image)
    # view_2D_matches(reference__SAM1005_matches_Mast3r,"".join([ref_image_name, '_Mast3r_matches']), shape=ref_image.shape, bg=ref_image)
    # view_2D_matches(masking_result['Mast3r'][3]['masked_matches'],"".join([ref_image_name, '_Mast3r_matches_after(3px)']), shape=ref_image.shape, bg=ref_image)
    # view_2D_matches(masking_result['Mast3r'][5]['masked_matches'],"".join([ref_image_name, '_Mast3r_matches(5px)']), shape=ref_image.shape, bg=ref_image)
    # view_2D_matches(masking_result['Mast3r'][7]['masked_matches'],"".join([ref_image_name, '_Mast3r_matches(7px)']), shape=ref_image.shape, bg=ref_image)

    ##############
    ## LiftFeat ##
    ##############

    reference__SAM1005_matches_LiftFeat = np.load(os.path.join(matches_savepath, 'LiftFeat', 'reference__SAM1005_matches.npy'))
    target__SAM1007_matches_LiftFeat = np.load(os.path.join(matches_savepath, 'LiftFeat', 'target__SAM1007_matches.npy'))

    # view_2D_matches(reference__SAM1005_matches_LiftFeat[:, :],"".join([ref_image_name, '_LiftFeat_matches']), shape=ref_image.shape)# 
    # view_2D_matches(target__SAM1007_matches_LiftFeat[:, :],"".join([target_image_name, '_LiftFeat_matches']), shape=target_image.shape)# 

    dist = cdist(reference__SAM1005_matches_LiftFeat, ref_vis_array)    
    masking_result['LiftFeat'] = {
        'reference_matches': reference__SAM1005_matches_LiftFeat,
        'target_matches': target__SAM1007_matches_LiftFeat, 
        **{ dist_boundary: {'mask': np.any(dist <= dist_boundary, axis=1), 'masked_matches': reference__SAM1005_matches_LiftFeat[np.any(dist <= dist_boundary, axis=1)] } for dist_boundary in distances_to_check}
    }

    # view_2D_matches(reference__SAM1005_matches_LiftFeat, "".join([target_image_name, '_LiftFeat_matches']), shape=ref_image.shape, bg=ref_image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit(0)

    #########
    ## RDD ##
    #########
 
    reference__SAM1005_matches_RDD = np.load(os.path.join(matches_savepath, 'RDD', 'reference__SAM1005_matches.npy'))
    target__SAM1007_matches_RDD = np.load(os.path.join(matches_savepath, 'RDD', 'target__SAM1007_matches.npy'))

    # view_2D_matches(reference__SAM1005_matches_RDD[:, :],"".join([ref_image_name, '_RDD_matches']), shape=ref_image.shape)# 
    # view_2D_matches(target__SAM1007_matches_RDD[:, :],"".join([target_image_name, '_RDD_matches']), shape=target_image.shape)# 

    dist = cdist(reference__SAM1005_matches_RDD, ref_vis_array)    
    masking_result['RDD'] = {
        'reference_matches': reference__SAM1005_matches_RDD,
        'target_matches': target__SAM1007_matches_RDD, 
        **{ dist_boundary: {'mask': np.any(dist <= dist_boundary, axis=1), 'masked_matches': reference__SAM1005_matches_RDD[np.any(dist <= dist_boundary, axis=1)] } for dist_boundary in distances_to_check}
    }

    view_2D_matches(reference__SAM1005_matches_RDD, "".join([target_image_name, '_RDD_matches']), shape=ref_image.shape, bg=ref_image)


    # Plot results
    distances_labels = tuple([*map(str, distances_to_check)])
    values = { }

    for model in masking_result.keys():
        values[model] = []
        for label in distances_to_check:
            values[model].append(len(masking_result[model][label]['masked_matches'])/len(masking_result[model]['reference_matches']) * 100)

        values[model] = tuple(values[model]) 

    x = np.arange(len(distances_labels))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = matplotlib.pyplot.subplots(layout='constrained')

    for attribute, measurement in values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=4)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('valid matches/total matches (%)')
    ax.set_title('')
    ax.set_xticks(x + width, distances_labels)
    ax.legend(loc='upper left', ncols=min(len(masking_result.keys()),5))
    # ax.set_ylim(0, 20)

    matplotlib.pyplot.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()



