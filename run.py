from src.extract import evaluateRDD
from utils.visibility import load_visibility
from utils.view import convert_coords, view_2D_matches
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

def draw_matches(ref_points, dst_points, img0, img1):
    
    # Prepare keypoints and matches for drawMatches function
    keypoints0 = [cv2.KeyPoint(p[0], p[1], 1000) for p in ref_points]
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 1000) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(ref_points))]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img0, keypoints0, img1, keypoints1, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches


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

def deep_match(reference, target, visibility_reference,  model_pointcloud, match_max_distance=15):

    reference_vis_array = convert_coords(np.asarray([*map( lambda vis_entry: [vis_entry['w'], vis_entry['h']], visibility_reference)]), width=reference.shape[1], height=reference.shape[0] )
    
    deep_match_results = {
        'reference_image': reference,
        'target_image': target,
        'by_model': {}
    }

    # TODO add other models
    deep_match_results['by_model']['RDD'] = {'matches': evaluateRDD(reference, target)}

    # draw_matches(deep_match_results['by_model']['RDD']['matches']['reference_matches'], deep_match_results['by_model']['RDD']['matches']['target_matches'],reference,target)
    # view_2D_matches(deep_match_results['by_model']['RDD']['matches']['reference_matches'],'a', bg_image=reference)
    #view_2D_matches(reference_vis_array,'a', bg_image=reference)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

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
        
        #view_2D_matches(deep_match_results['by_model'][model]['pose_data']['target_points'],'a', bg_image=target)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        print(deep_match_results['by_model'][model]['pose_data']['target_points'].shape)
        print(deep_match_results['by_model'][model]['pose_data']['model_points'].shape)

        success, rvec, tvec = cv2.solvePnP(deep_match_results['by_model'][model]['pose_data']['model_points'], deep_match_results['by_model'][model]['pose_data']['target_points'], cameraMatrix=camera_matrix, distCoeffs=dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        
        #due_D = np.asarray([*map( lambda vis_entry: [vis_entry['w'], vis_entry['h']], visibility_reference)])
        #tre_D = np.asarray([*map( lambda vis_entry: pc_array[vis_entry['index'], :], visibility_reference)])

        #print(due_D.shape)
        #print(tre_D.shape)
        #success, rvec, tvec = cv2.solvePnP(tre_D, due_D, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        deep_match_results['by_model'][model]['pose_data']['success'] = success
        rmat, _ = cv2.Rodrigues(rvec)
        deep_match_results['by_model'][model]['pose_data']['R'] = rmat 
        deep_match_results['by_model'][model]['pose_data']['t'] = tvec

        target_reproj, _ = cv2.projectPoints(pc_array, rvec, tvec, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)

        deep_match_results['by_model'][model]['target_reproj']  = np.array(target_reproj, dtype=np.float32).astype(int)
        deep_match_results['by_model'][model]['target_reproj'] = np.squeeze(deep_match_results['by_model'][model]['target_reproj'], axis=1)
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

    print(pairs_match_results[0]['by_model']['RDD']['pose_data']['success'])
    print(pairs_match_results[0]['by_model']['RDD']['pose_data']['R'])
    print(pairs_match_results[0]['by_model']['RDD']['pose_data']['t'])


    print(pairs_match_results[0]['by_model']['RDD']['target_reproj'])

    view_2D_matches(pairs_match_results[0]['by_model']['RDD']['target_reproj'],'a', bg_image=pairs_match_results[0]['target_image'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

