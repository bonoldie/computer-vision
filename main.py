
import open3d as o3d
import copy
import cv2
import numpy as np

from utils.visibility import load_visibility
from utils.xmp import parse_xmp_camera
from utils.view import draw_matches, view_2D_matches, zephyr_to_cv2

from src.extract import evaluateRDD, evaluateLiftFeat, evaluateMast3r, evaluateRoMa
from scipy.spatial.distance import cdist

from loguru import logger
logger.add('logs/log_{time}.log', compression='zip', rotation='1 hour')

if __name__ == '__main__':
    visibility_map_path = 'downloads/dante_rework/Visibility.txt'
    logger.info(f'Loading visibility map ({visibility_map_path})')
    visibility_map = load_visibility(visibility_map_path)

    # loads reworked point cloud
    point_cloud_path = 'downloads/dante_rework/SamPointCloud.ply'
    logger.info(f'Loading point cloud using Open3D ({point_cloud_path})')
    pc = o3d.io.read_point_cloud(point_cloud_path)
    pc_points = np.asarray(pc.points)

    # loads media
    reference_image = '_SAM1005.JPG'
    target_image = '_SAM1007.JPG'


    reference_image_path = f'downloads/dante_dataset/dante_dataset/photos/{reference_image}'
    target_image_path = f'downloads/dante_dataset/dante_dataset/photos/{target_image}'

    reference_K, reference_R, reference_t, dist_coeffs = parse_xmp_camera(f'downloads/dante_rework/extrinsics/{reference_image}.xmp')
    target_K, target_R, target_t, _ = parse_xmp_camera(f'downloads/dante_rework/extrinsics/{target_image}.xmp')
    target_T = np.identity(4)
    target_T[:3, :3] = np.array(target_R)
    target_T[:3, 3] = np.array(target_t).flatten()

    reference = cv2.imread(reference_image_path)

    reference_visibility = visibility_map[reference_image]
    reference_2D_feature_points = np.asarray([*map( lambda vis_entry: [vis_entry['w'], vis_entry['h']], reference_visibility)])
    reference_2D_feature_points = zephyr_to_cv2(reference_2D_feature_points,width=reference.shape[1], height=reference.shape[0])

    if False:
        view_2D_matches(reference_2D_feature_points, 'test', bg_image=reference_image_path)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # extracting 2D matches

    RDD_matches = evaluateRDD(reference_image_path, target_image_path)
    LiftFeat_matches = evaluateLiftFeat(reference_image_path, target_image_path)
    Mast3r_matches = evaluateMast3r(reference_image_path, target_image_path)

    if False: 
        # show the matches 

        RDD_canvas = draw_matches(RDD_matches['reference_matches'], RDD_matches['target_matches'], reference_image_path, target_image_path )
        LiftFeat_canvas = draw_matches(LiftFeat_matches['reference_matches'], LiftFeat_matches['target_matches'], reference_image_path, target_image_path )
        Mast3r_canvas = draw_matches(Mast3r_matches['reference_matches'], Mast3r_matches['target_matches'], reference_image_path, target_image_path )
        
        cv2.namedWindow("RDD matches", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RDD matches", 800, 600)
        cv2.imshow("RDD matches", RDD_canvas)

        cv2.namedWindow("LiftFeat matches", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("LiftFeat matches", 800, 600)
        cv2.imshow("LiftFeat matches", LiftFeat_canvas)

        cv2.namedWindow("Mast3r matches", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Mast3r matches", 800, 600)
        cv2.imshow("Mast3r matches", Mast3r_canvas)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
    # 2D-2D matches to 2D-3D matches
    
    reference_2D_feature_points_to_RDD_matches_distance = cdist(RDD_matches['reference_matches'], reference_2D_feature_points)
    indexes = np.argmin(reference_2D_feature_points_to_RDD_matches_distance, axis=1)

    target_3D_matches = np.asarray([pc_points[idx] for idx in [reference_visibility[idx]['index'] for idx in indexes]])

    is_target_pose_ok, target_rvec_estimated, target_t_estimated, inliers = cv2.solvePnPRansac(target_3D_matches, RDD_matches['target_matches'], cameraMatrix=target_K, distCoeffs=dist_coeffs, iterationsCount=100000, confidence=0.5)
    
    target_R_estimated, _ = cv2.Rodrigues(target_rvec_estimated)

    target_T_estimated = np.identity(4)
    target_T_estimated[:3, :3] = target_R_estimated
    target_T_estimated[:3, 3] = target_t_estimated.flatten()

    print(is_target_pose_ok)
    print(target_T)
    print(np.linalg.inv(target_T))
    print(target_T_estimated)
    print(np.linalg.inv(target_T_estimated))

