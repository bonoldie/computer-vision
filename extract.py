import sys
import os

import torch
torch.cuda.empty_cache()

from rdd.RDD.utils.misc import read_config
from rdd.RDD.RDD import build
from rdd.RDD.RDD_helper import RDD_helper
from LiftFeat.models.liftfeat_wrapper import LiftFeat

from utils.visibility import load_visibility, visualize_visibility

import open3d as o3d
from matplotlib import pyplot as plt
import cv2

import numpy as np
from pathlib import Path



from loguru import logger
logger.add('logs/log_{time}.log', compression='zip', rotation='1 hour')

matches_savepath = os.path.join(os.getcwd(), 'matches')
 
# liftfeat_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'LiftFeat'))
# if liftfeat_path not in sys.path:
#     sys.path.insert(0, liftfeat_path)

def setupRDD():
    logger.info('Setting up RDD...')
    RDD_model = build(weights='./downloads/RDD-v2.pth', config=read_config('./configs/rdd.yaml'))
    RDD_model.eval()
    RDD = RDD_helper(RDD_model)
    return RDD

def setupLiftFeat():
    logger.info('Setting up LiftFeat...')
    liftfeat = LiftFeat(detect_threshold=0.05)   
    liftfeat.cpu() 
    return liftfeat


def draw_matches(ref_points, dst_points, img0, img1):
    
    # Prepare keypoints and matches for drawMatches function
    keypoints0 = [cv2.KeyPoint(p[0], p[1], 1000) for p in ref_points]
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 1000) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(ref_points))]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img0, keypoints0, img1, keypoints1, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches

if __name__ == '__main__':
    # Load visibility map

    vis_map_path = 'downloads/dante_dataset/dante_dataset/Visibility.txt'
    logger.info(f'Loading visibility map: {vis_map_path}')
    vis_map = load_visibility(vis_map_path)
    #vis_map: {"photo_name": [ {"index": "1234", "w": "12.2", "h": "799.5" }, ...], ...}
    
    # image_path = os.path.join(
    # os.getcwd(), "downloads", "dante_dataset", "dante_dataset", "photos", "_SAM1001.JPG")
    
    # visualize_visibility(vis_map,image_path,radius=5)
    
    # pc = o3d.io.read_point_cloud('downloads/dante_dataset/dante_dataset/SamPointCloud.ply')

    _SAM1005 = cv2.imread('downloads/dante_dataset/dante_dataset/photos/_SAM1005.JPG')
    _SAM1007 = cv2.imread('downloads/dante_dataset/dante_dataset/photos/_SAM1007.JPG')

    ref = _SAM1005
    target = _SAM1007

    # Models instances
    RDD = setupRDD()
    LF = setupLiftFeat()
   
    logger.info(f'Matches save dir: {matches_savepath}')

    Path(os.path.join(matches_savepath, 'RDD')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(matches_savepath, 'LiftFeat')).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # RDD.cpu()

        ######################
        ## LiftFeat section ##
        ######################
        logger.info('Running inference on LiftFeat...')

        ref_H,ref_W = ref.shape[:2]
        ref_LF = cv2.resize(ref, (round(ref_W/3),round(ref_H/3)))

        target_H,target_W = ref.shape[:2]
        target_LF = cv2.resize(ref, (round(target_W/3),round(target_H/3)))

        ref_matches, target_matches = LF.match_liftfeat(ref_LF, target_LF)
        
        logger.log('SUCCESS' if len(ref_matches) > 0 else 'WARNING',f'LiftFeat returned {len(ref_matches)} matches')
    
        np.save(os.path.join(matches_savepath, 'LiftFeat', 'reference__SAM1005_matches'),np.asanyarray(ref_matches))
        np.save(os.path.join(matches_savepath, 'LiftFeat', 'target__SAM1007_matches'),np.asanyarray(target_matches))

        logger.success(f'LiftFeat matches saved')


        #################
        ## RDD section ##
        #################

        logger.info('Running inference on RDD...')
        ref_matches, target_matches, conf = RDD.match(ref, target, resize=1024)
        
        logger.log('SUCCESS' if len(ref_matches) > 0 else 'WARNING',f'RDD returned {len(ref_matches)} matches')
    
        np.save(os.path.join(matches_savepath, 'RDD', 'reference__SAM1005_matches'),np.asanyarray(ref_matches))
        np.save(os.path.join(matches_savepath, 'RDD', 'target__SAM1007_matches'),np.asanyarray(target_matches))

        logger.success(f'RDD matches saved')




        #canvas = draw_matches(_SAM1005_matches, _SAM1007_matches, _SAM1005,_SAM1007)
        #plt.figure(figsize=(12,12))
        #plt.imshow(canvas[..., ::-1]), plt.show()



