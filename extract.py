import sys
import os
import gc

import torch
torch.device
torch.cuda.empty_cache()

from rdd.RDD.utils.misc import read_config
from rdd.RDD.RDD import build
from rdd.RDD.RDD_helper import RDD_helper
from LiftFeat.models.liftfeat_wrapper import LiftFeat
from mast3r.mast3r.model import AsymmetricMASt3R
from mast3r.mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.dust3r.dust3r.inference import inference
from mast3r.dust3r.dust3r.utils.image import load_images
#from RoMa.romatch import roma_outdoor

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

def setupMaste3r():
    logger.info('Setting up Mast3r...')
    return AsymmetricMASt3R.from_pretrained('downloads/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth').to('cpu')


#def setupRoMa():
#    logger.info('Setting up RoMa(RoMa outdoor)...')
#    return roma_outdoor(device='cpu', coarse_res=int(420), upsample_res=int(560))

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

    ref_H,ref_W = ref.shape[:2]
    target_H,target_W = target.shape[:2]   
   
    logger.info(f'Matches save dir: {matches_savepath}')

    Path(os.path.join(matches_savepath, 'RDD')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(matches_savepath, 'LiftFeat')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(matches_savepath, 'Mast3r')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(matches_savepath, 'RoMa')).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # RDD.cpu()

        ##########
        ## RoMa ##
        ##########
        #RoMa = setupRoMa()
        #logger.info('Running inference on RoMa...')

        #warp, certainty = RoMa.match('downloads/dante_dataset/dante_dataset/photos/_SAM1005.JPG', 'downloads/dante_dataset/dante_dataset/photos/_SAM1007.JPG', device='cpu')
        ## Sample matches for estimation
        #matches, certainty = RoMa.sample(warp, certainty)
        ## Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
        #kptsA, kptsB = RoMa.to_pixel_coordinates(matches, ref_H, ref_W, target_H, target_W)
#
        #logger.log('SUCCESS' if len(kptsA) > 0 else 'WARNING',f'RoMa returned {len(kptsA)} matches')
    #
        #np.save(os.path.join(matches_savepath, 'RoMa', 'reference__SAM1005_matches'),np.asanyarray(kptsA))
        #np.save(os.path.join(matches_savepath, 'RoMa', 'target__SAM1007_matches'),np.asanyarray(kptsB))
#
        #logger.success(f'RoMa matches saved')
#
        #del RoMa, warp, certainty, kptsA, kptsB
        #gc.collect()

        ############
        ## Mast3r ##
        ############
        #Mast3r = setupMaste3r()
        #logger.info('Running inference on Mast3r...')
        #
        #images = load_images(['downloads/dante_dataset/dante_dataset/photos/_SAM1005.JPG', 'downloads/dante_dataset/dante_dataset/photos/_SAM1007.JPG'], size=512)
        ##ref_Mast3r, = cv2.resize(ref, (512,512))
        ##target_Mast3r = cv2.resize(target, (512,512))
        #output = inference([tuple(images)], Mast3r, 'cpu', batch_size=1, verbose=False)
#
        ## at this stage, you have the raw dust3r predictions
        #view1, pred1 = output['view1'], output['pred1']
        #view2, pred2 = output['view2'], output['pred2']
#
        #desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
#
        ## find 2D-2D matches between the two images
        #matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8, device='cpu', dist='dot', block_size=2**13)
#
        ## ignore small border around the edge
        #H0, W0 = view1['true_shape'][0]
        #valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        #    matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)
#
        #H1, W1 = view2['true_shape'][0]
        #valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        #    matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)
#
        #valid_matches = valid_matches_im0 & valid_matches_im1
        #matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]
#
        #logger.log('SUCCESS' if len(matches_im0) > 0 else 'WARNING',f'Mast3r returned {len(matches_im0)} matches')
    #
        #np.save(os.path.join(matches_savepath, 'Mast3r', 'reference__SAM1005_matches'),np.asanyarray(matches_im0))
        #np.save(os.path.join(matches_savepath, 'Mast3r', 'target__SAM1007_matches'),np.asanyarray(matches_im1))
#
        #logger.success(f'Mast3r matches saved')
#
        #del images, output, view1, view2, pred1, pred2, desc1, desc2, matches_im0, matches_im1, valid_matches_im0, valid_matches_im1, valid_matches, Mast3r
        #gc.collect() 

        ######################
        ## LiftFeat section ##
        ######################
        LF = setupLiftFeat()
        logger.info('Running inference on LiftFeat...')

        ref_H,ref_W = ref.shape[:2]
        ref_LF = cv2.resize(ref, (round(ref_W/5),round(ref_H/5)))

        target_H,target_W = target.shape[:2]
        target_LF = cv2.resize(ref, (round(target_W/5),round(target_H/5)))

        ref_matches, target_matches = LF.match_liftfeat(ref_LF, target_LF)
        
        logger.log('SUCCESS' if len(ref_matches) > 0 else 'WARNING',f'LiftFeat returned {len(ref_matches)} matches')
    
        np.save(os.path.join(matches_savepath, 'LiftFeat', 'reference__SAM1005_matches'),np.asanyarray(ref_matches))
        np.save(os.path.join(matches_savepath, 'LiftFeat', 'target__SAM1007_matches'),np.asanyarray(target_matches))

        logger.success(f'LiftFeat matches saved')

        del LF
        gc.collect()

        #################
        ## RDD section ##
        #################
        RDD = setupRDD()

        logger.info('Running inference on RDD...')
        ref_matches, target_matches, conf = RDD.match(ref, target, resize=1024)
        
        logger.log('SUCCESS' if len(ref_matches) > 0 else 'WARNING',f'RDD returned {len(ref_matches)} matches')
    
        np.save(os.path.join(matches_savepath, 'RDD', 'reference__SAM1005_matches'),np.asanyarray(ref_matches))
        np.save(os.path.join(matches_savepath, 'RDD', 'target__SAM1007_matches'),np.asanyarray(target_matches))

        logger.success(f'RDD matches saved')


        #canvas = draw_matches(_SAM1005_matches, _SAM1007_matches, _SAM1005,_SAM1007)
        #plt.figure(figsize=(12,12))
        #plt.imshow(canvas[..., ::-1]), plt.show()



