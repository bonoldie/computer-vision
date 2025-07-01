import sys
import os

import torch
torch.cuda.empty_cache()

from rdd.RDD.utils.misc import read_config
from rdd.RDD.RDD import build
from rdd.RDD.RDD_helper import RDD_helper
from LiftFeat.models.liftfeat_wrapper import LiftFeat

from utils.load_visibility import load_visibility 

import open3d as o3d
from matplotlib import pyplot as plt
import cv2
import kornia

from PIL import Image
import numpy as np


liftfeat_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'LiftFeat'))
if liftfeat_path not in sys.path:
    sys.path.insert(0, liftfeat_path)

def setupRDD():
    RDD_model = build(weights='./downloads/RDD-v2.pth', config=read_config('./rdd/configs/default.yaml'))
    RDD_model.eval()
    RDD = RDD_helper(RDD_model)
    return RDD

def setupLiftFeat():
    liftfeat = LiftFeat(detect_threshold=0.05)    
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

    vis_map = load_visibility('downloads/dante_dataset/dante_dataset/Visibility.txt')
    # vis_map: {"photo_name": [ {"index": "1234", "w": "12.2", "h": "799.5" }, ...], ...}
    # print(vis_map["_SAM1001.JPG"])

    pc = o3d.io.read_point_cloud('downloads/dante_dataset/dante_dataset/SamPointCloud.ply')

    _SAM1005 = cv2.imread('downloads/dante_dataset/dante_dataset/photos/_SAM1005.JPG')
    _SAM1007 = cv2.imread('downloads/dante_dataset/dante_dataset/photos/_SAM1007.JPG')

    # Models instances
    RDD = setupRDD()
    LF = setupLiftFeat()
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))


    with torch.no_grad():
        # RDD.cpu()

        in1 = torch.tensor(_SAM1005)
        in2 = torch.tensor(_SAM1007)

        in1 = in1.transpose(0, 2)
        in1 = in1[None,:,:,:]
        
        in2 = in2.transpose(0, 2)
        in2 = in2[None,:,:,:]
        
        _SAM1005_matches, _SAM1007_matches, conf = RDD.match(in1, in2, resize=1024)

        print(_SAM1005_matches)
        print(_SAM1007_matches)

        canvas = draw_matches(_SAM1005_matches, _SAM1007_matches, _SAM1005,_SAM1007)
        plt.figure(figsize=(12,12))
        plt.imshow(canvas[..., ::-1]), plt.show()



