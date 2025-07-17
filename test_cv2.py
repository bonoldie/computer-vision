from src.extract import evaluateRDD
from utils.visibility import load_visibility
from utils.view import view_2D_matches, compute_camera_matrix, draw_matches, zephyr_to_cv2, cv2_to_zephyr
import cv2
from pathlib import Path
import open3d as o3d
import numpy as np
from scipy.spatial.distance import cdist
import json


dante_camera_matrix = compute_camera_matrix(
    5794.23954825, 5794.23954825, 2792.29091924, 1817.35747811)

dante_dist_coeffs = np.array([
    -0.049194332605,   # k1
    0.0277098399156,  # k2
    0.0,              # p1
    0.0,              # p2
    0.12963081329     # k3
])

def loadPairs(pairs_path):
    pairs = []

    for pair_path in pairs_path:
        pairs.append(
            (cv2.imread(pair_path[0]), cv2.imread(pair_path[1]))
        )

    return pairs


def deep_match(reference, target, visibility_reference,  model_pointcloud, match_max_distance=15):

    width, height = reference.shape[0], reference.shape[1]
    model_pointcloud_points = np.asarray(model_pointcloud.points)

    reference_vis_array = np.asarray(
        [*map(lambda vis_entry: [vis_entry['w'], vis_entry['h']], visibility_reference)]
    )

    deep_match_results = {
        'reference_image': reference,
        'target_image': target,
        'by_model': {}
    }

    RDD_matches = evaluateRDD(reference, target)

    deep_match_results['by_model'] = {
        'RDD': {
            'matches': {
                'reference_matches':  cv2_to_zephyr(RDD_matches['reference_matches'], width, height),
                'target_matches':  cv2_to_zephyr(RDD_matches['target_matches'], width, height)
            }
        },
        # TODO add other models
    }

    # draw_matches(deep_match_results['by_model']['RDD']['matches']['reference_matches'], deep_match_results['by_model']['RDD']['matches']['target_matches'],reference,target)
    # view_2D_matches(deep_match_results['by_model']['RDD']['matches']['reference_matches'],'a', bg_image=reference)
    # view_2D_matches(reference_vis_array,'a', bg_image=reference)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for model in deep_match_results['by_model'].keys():
        reference_matches = deep_match_results['by_model'][model]['matches']['reference_matches']
        target_matches = deep_match_results['by_model'][model]['matches']['target_matches']

        dist = cdist(reference_matches, reference_vis_array)

        dist_indices = np.argwhere((dist <= match_max_distance) == True)

        target_points = np.asanyarray(target_matches[dist_indices[:, 0]])

        model_points = np.asanyarray(
            [*map(lambda index: model_pointcloud_points[visibility_reference[index]['index']], dist_indices[:, 1])])

        print(target_points.shape)
        print(model_points.shape)

        success, r_vec, t_vec = cv2.solvePnP(
            model_points, target_points, cameraMatrix=dante_camera_matrix, distCoeffs=dante_dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        R, _ = cv2.Rodrigues(r_vec)

        deep_match_results['by_model'][model]['pose'] = {
            'success': success,
            'R': R,
            't': t_vec
        }

        deep_match_results['by_model'][model]['model_target_reprojection'] = np.squeeze(
            np.array(target_reproj, dtype=np.float32).astype(int), axis=1)

    return deep_match_results


if __name__ == '__main__':

    visibility_map_path = 'downloads/dante_dataset/dante_dataset/Visibility.txt'
    visibility_map = load_visibility(visibility_map_path)

    pc = o3d.io.read_point_cloud(
        'downloads/dante_dataset/dante_dataset/SamPointCloud.ply')

    pairs_path = [
        # (reference_path, target_path)
        ('downloads/dante_dataset/dante_dataset/photos/_SAM1005.JPG',
         'downloads/dante_dataset/dante_dataset/photos/_SAM1006.JPG'),
    ]

    # each pair is a tuple (reference image, target image)
    pairs = loadPairs(pairs_path)

    pairs_match_results = []

    for (pair_index, pair) in enumerate(pairs):

        width, height = pair[0].shape[0], pair[0].shape[1]

        visibility_reference = visibility_map.get(
            Path(pairs_path[pair_index][0]).name)

        model_pointcloud_points = np.asarray(pc.points)

        reference_vis_array = np.asarray(
            [*map(lambda vis_entry: [vis_entry['w'],
                  vis_entry['h']], visibility_reference)]
        )

        reference_vis_array = cv2_to_zephyr(reference_vis_array, width, height)

        model_points = np.asanyarray(
            [*map(lambda visibility_entry: model_pointcloud_points[visibility_entry['index']], visibility_reference)])

        success, r_vec, t_vec = cv2.solvePnP(
            model_points, reference_vis_array, cameraMatrix=dante_camera_matrix, distCoeffs=dante_dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        R, _ = cv2.Rodrigues(r_vec)

        print(R)
        print(t_vec)

        ref_reproj, _ = cv2.projectPoints(
            model_pointcloud_points, r_vec, t_vec, cameraMatrix=dante_camera_matrix, distCoeffs=dante_dist_coeffs)

        ref_reproj = np.squeeze(
            np.array(ref_reproj, dtype=np.float32).astype(int), axis=1)

        view_2D_matches(ref_reproj, 'a', bg_image=pair[0])
        cv2.waitKey(0)

    cv2.destroyAllWindows()
