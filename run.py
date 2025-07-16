from src.extract import evaluateRDD
from utils.visibility import load_visibility
from utils.view import convert_coords
import cv2
from pathlib import Path
import open3d as o3d
import numpy as np
from scipy.spatial.distance import cdist

pairs_path = [
    # (reference_path, target_path)
    ('downloads/dante_dataset/dante_dataset/photos/_SAM1005.JPG', 'downloads/dante_dataset/dante_dataset/photos/_SAM1007.JPG'),
]


def loadPairs():
    pairs = []

    for pair_path in pairs_path:
        pairs.append(
            (cv2.imread(pair_path[0]), cv2.imread(pair_path[1]))
        )
    
    return pairs

def deep_match(reference, target, visibility_reference,  model_pointcloud, match_max_distance=3):

    reference_vis_array = convert_coords(np.asarray([*map( lambda vis_entry: [vis_entry['w'], vis_entry['h']], visibility_reference)]), width=reference.shape[1], height=reference.shape[0] )
    

    deep_match_results = {
        'reference_image': reference,
        'target_image': target,
        'by_model': {}
    }

    # TODO add other models
    deep_match_results['by_model']['RDD'] = {'matches': evaluateRDD(reference, target)}


    for model in  deep_match_results['by_model'].keys():
        reference_matches = deep_match_results['by_model'][model]['matches']['reference_matches']
        target_matches = deep_match_results['by_model'][model]['matches']['target_matches']

        dist = cdist(reference_matches, reference_vis_array)
        dist_mask = dist <= match_max_distance

        dist_indices = np.argwhere(dist_mask == True)

        deep_match_results['by_model'][model]['pose_data'] = np.array([])

        for indices in dist_indices:
            pass





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
        
        visibility_reference = visibility_map.get(Path(pairs_path[pair_index]).stem)

        pairs_match_results.append(deep_match(pair[0], pair[1], visibility_reference, pc))



