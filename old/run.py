from src.extract import evaluateRDD
from utils.visibility import load_visibility
from utils.view import view_2D_matches, compute_camera_matrix, draw_matches, zephyr_to_cv2, cv2_to_zephyr
import cv2
from pathlib import Path
import open3d as o3d
import numpy as np
from scipy.spatial.distance import cdist
import json

__SAM1007_T = np.array([
    [0.08404846995145755, 0.1946269345254674, 0.9772708918842394, -80.91637301148417],
    [-0.962014309988296, 0.2714960996071781, 0.02866710023339282, -10.21486893928423],
    [-0.259744500184983, -0.9425570791974044,  0.2100523063297199, 56.7239534506907],
    [0, 0, 0, 1.0]
])

dante_camera_matrix = compute_camera_matrix(
    5794.23954825, 5794.23954825, 2792.29091924, 1817.35747811)

dante_dist_coeffs = np.array([
    -0.049194332605,   # k1
    0.0277098399156,  # k2
    0.0,              # p1
    0.0,              # p2
    0.12963081329     # k3
])

# dante_camera_matrix = np.array([
#    [5794.23954825, 5.00502053223e-12, 2792.29091924],
#    [0.0,           5794.23954825,     1817.35747811],
#    [0.0,           0.0,               1.0]
# ])
# dante_dist_coeffs = np.array([
#    -0.049194332605,   # k1
#     0.0277098399156,  # k2
#     0.0,              # p1
#     0.0,              # p2
#     0.12963081329     # k3
# ])


def undistorted_pixels_to_distorted(undistorted_pixel, camera_matrix, dist_coeffs):
    """
    Reapply distortion to undistorted pixel coordinates.

    Parameters:
        undistorted_pixel: Nx2 array of undistorted pixel coords (u, v)
        camera_matrix: 3x3 intrinsic matrix
        dist_coeffs: distortion coefficients [k1, k2, p1, p2, k3]

    Returns:
        distorted_pixel: Nx2 array of distorted pixel coords
    """
    # Inverse of camera matrix
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # Normalize pixel coordinates
    x = (undistorted_pixel[:, 0] - cx) / fx
    y = (undistorted_pixel[:, 1] - cy) / fy

    # Apply distortion
    k1, k2, p1, p2, k3 = dist_coeffs
    r2 = x**2 + y**2
    r4 = r2**2
    r6 = r2**3

    radial = 1 + k1*r2 + k2*r4 + k3*r6
    x_distorted = x * radial + 2*p1*x*y + p2*(r2 + 2*x**2)
    y_distorted = y * radial + p1*(r2 + 2*y**2) + 2*p2*x*y

    # Convert back to pixel coordinates
    u_distorted = fx * x_distorted + cx
    v_distorted = fy * y_distorted + cy

    return np.stack([u_distorted, v_distorted], axis=1)


def loadPairs(pairs_path):
    pairs = []

    for pair_path in pairs_path:
        pairs.append(
            (cv2.imread(pair_path[0]), cv2.imread(pair_path[1]))
        )

    return pairs


def deep_match(reference, target, visibility_reference,  model_pointcloud, match_max_distance=15):

    width, height = reference.shape[1], reference.shape[0]
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
                'reference_matches': cv2_to_zephyr(RDD_matches['reference_matches'], width, height),
                'target_matches': cv2_to_zephyr(RDD_matches['target_matches'], width, height),
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

        target_reproj, _ = cv2.projectPoints(
            model_pointcloud_points, r_vec, t_vec, cameraMatrix=dante_camera_matrix, distCoeffs=dante_dist_coeffs)
        target_reproj = np.squeeze(
            np.array(target_reproj, dtype=np.float32).astype(int), axis=1)

        # target_reproj = cv2.undistortPoints(target_reproj, dante_camera_matrix, dante_dist_coeffs)
        # target_reproj = cv2.convertPointsToHomogeneous(target_reproj) @ dante_camera_matrix.T
        # target_reproj = target_reproj[:, 0, :2] / target_reproj[:, 0, 2:]

        # target_reproj = zephyr_to_cv2(target_reproj, width, height)

        # target_reproj = undistorted_pixels_to_distorted(target_reproj, dante_camera_matrix, dante_dist_coeffs)

        deep_match_results['by_model'][model]['model_target_reprojection'] = target_reproj
    return deep_match_results


if __name__ == '__main__':

    visibility_map_path = 'downloads/dante_dataset/dante_dataset/Visibility.txt'
    visibility_map = load_visibility(visibility_map_path)

    pc = o3d.io.read_point_cloud(
        'downloads/dante_dataset/dante_dataset/SamPointCloud.ply')

    pc1 = o3d.io.read_point_cloud(
        'downloads/dante_dataset/dante_dataset/SamPointCloud.ply')

    pc2 = o3d.io.read_point_cloud(
        'downloads/dante_dataset/dante_dataset/SamPointCloud.ply')

    pairs_path = [
        # (reference_path, target_path)
        ('downloads/dante_dataset/dante_dataset/photos/_SAM1005.JPG',
         'downloads/dante_dataset/dante_dataset/photos/_SAM1007.JPG'),
    ]

    # each pair is a tuple (reference image, target image)
    pairs = loadPairs(pairs_path)

    pairs_match_results = []

    for (pair_index, pair) in enumerate(pairs):

        visibility_reference = visibility_map.get(
            Path(pairs_path[pair_index][0]).name)

        pairs_match_results.append(deep_match(
            pair[0], pair[1], visibility_reference, pc))

    print(pairs_match_results[0]['by_model']['RDD']['pose']['success'])
    print(pairs_match_results[0]['by_model']['RDD']['pose']['R'])
    print(pairs_match_results[0]['by_model']['RDD']['pose']['t'])

    # print(pairs_match_results[0]['by_model']['RDD']['target_reproj'])

    # T = np.eye(4)
    # T[:3, :3] = pairs_match_results[0]['by_model']['RDD']['pose']['R']
    # T[:3, 3] = pairs_match_results[0]['by_model']['RDD']['pose']['t'].flatten()
    # pc1.transform(np.linalg.inv(T))
    # pc2.transform(np.linalg.inv(__SAM1007_T))
    # distances = np.linalg.norm(np.asarray(pc1.points) - np.asarray(pc2.points), axis=1)  # Shape: (N1, N2)
    # print(np.sum(distances))


    # Visualize the transformed point cloud
    # o3d.visualization.draw_geometries([pc])

    view_2D_matches(pairs_match_results[0]['by_model']['RDD']['model_target_reprojection'], 'a', bg_image=pairs_match_results[0]['target_image'])
    
    theta = np.pi / 2
    R_fix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    # Combined rotation
    R_corrected = R_fix @ __SAM1007_T[:3, :3]
    t_corrected = R_fix @  __SAM1007_T[:3, 3]


    r_vec, _ = cv2.Rodrigues(R_corrected)
    # print(__SAM1007_T[:3, 3])
    original_target_reproj, _ = cv2.projectPoints(np.asarray(pc.points), r_vec,t_corrected, cameraMatrix=dante_camera_matrix, distCoeffs=dante_dist_coeffs)
    original_target_reproj = np.squeeze(np.array(original_target_reproj, dtype=np.float32).astype(int), axis=1)

    view_2D_matches(original_target_reproj,'b', bg_image=pairs_match_results[0]['target_image'])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
