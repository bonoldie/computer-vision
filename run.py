from src.extract import evaluateRDD
from utils.visibility import load_visibility
from utils.view import convert_coords
import cv2
from pathlib import Path
import open3d as o3d
import numpy as np
from scipy.spatial.distance import cdist
import json

pairs_path = [
    # (reference_path, target_path)
    ('downloads/dante_dataset/dante_dataset/photos/_SAM1005.JPG', 'downloads/dante_dataset/dante_dataset/photos/_SAM1007.JPG'),
]


def loadPairs(pairs_path):
    pairs = []

    for pair_path in pairs_path:
        pairs.append(
            (cv2.imread(pair_path[0]), cv2.imread(pair_path[1]))
        )
    
    return pairs

camera_matrix = np.array([
    [5794.23954825, 5.00502053223e-12, 2792.29091924],
    [0.0,           5794.23954825,     1817.35747811],
    [0.0,           0.0,               1.0]
])
dist_coeffs = np.array([
    -0.049194332605,   # k1
     0.0277098399156,  # k2
     0.0,              # p1
     0.0,              # p2
     0.12963081329     # k3
])

def deep_match(reference, target, visibility_reference,  model_pointcloud, match_max_distance=3):

    reference_vis_array = convert_coords(np.asarray([*map( lambda vis_entry: [vis_entry['w'], vis_entry['h']], visibility_reference)]), width=reference.shape[1], height=reference.shape[0] )
    
    deep_match_results = {
        'reference_image': reference,
        'target_image': target,
        'by_model': {}
    }

    # TODO add other models
    deep_match_results['by_model']['RDD'] = {'matches': evaluateRDD(reference, target)}


    pc_array = np.asarray(model_pointcloud.points)

    for model in  deep_match_results['by_model'].keys():
        reference_matches = deep_match_results['by_model'][model]['matches']['reference_matches']
        target_matches = deep_match_results['by_model'][model]['matches']['target_matches']

        dist = cdist(reference_matches, reference_vis_array)
        dist_mask = dist <= match_max_distance

        dist_indices = np.argwhere(dist_mask == True)

        deep_match_results['by_model'][model]['pose_data'] = {'target_points': [], 'model_points': []}

        for ref_index, vis_index in dist_indices:
            target_2d_point = target_matches[ref_index]
            model_3d_point_index = visibility_reference[vis_index]['index']
            model_3d_point = pc_array[model_3d_point_index]
            
            deep_match_results['by_model'][model]['pose_data']['target_points'].append(target_2d_point)
            deep_match_results['by_model'][model]['pose_data']['model_points'].append(model_3d_point)

        deep_match_results['by_model'][model]['pose_data']['target_points'] = np.asanyarray(deep_match_results['by_model'][model]['pose_data']['target_points'])
        deep_match_results['by_model'][model]['pose_data']['model_points'] = np.asanyarray(deep_match_results['by_model'][model]['pose_data']['model_points'])
        
        print(deep_match_results['by_model'][model]['pose_data']['target_points'].shape)
        print(deep_match_results['by_model'][model]['pose_data']['model_points'].shape)

        success, rvec, tvec = cv2.solvePnP(deep_match_results['by_model'][model]['pose_data']['model_points'], deep_match_results['by_model'][model]['pose_data']['target_points'], cameraMatrix=camera_matrix, distCoeffs=dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        deep_match_results['by_model'][model]['pose_data']['success'] = success
        rmat, _ = cv2.Rodrigues(rvec)
        deep_match_results['by_model'][model]['pose_data']['R'] = rmat 
        deep_match_results['by_model'][model]['pose_data']['t'] = tvec


        # matches_mask = np.any(dist <= match_max_distance, axis=1)
        # point_index_mask = np.any(dist <= match_max_distance, axis=2)

    
    return deep_match_results
    
    

if __name__ == '__main__':

    visibility_map_path = 'downloads/dante_dataset/dante_dataset/Visibility.txt'
    visibility_map = load_visibility(visibility_map_path)

    pc = o3d.io.read_point_cloud('downloads/dante_dataset/dante_dataset/SamPointCloud.ply')

    # each pair is a tuple (reference image, target image)
    pairs = loadPairs(pairs_path)

    pairs_match_results = [] 
    
    for (pair_index, pair) in enumerate(pairs):
        
        visibility_reference = visibility_map.get(Path(pairs_path[pair_index][0]).name)

        pairs_match_results.append(deep_match(pair[0], pair[1], visibility_reference, pc))

    
    print(pairs_match_results[0]['by_model']['RDD']['pose_data']['R'])
    print(pairs_match_results[0]['by_model']['RDD']['pose_data']['t'])



