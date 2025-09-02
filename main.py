
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import copy
import cv2
import numpy as np

from utils.visibility import load_visibility
from utils.xmp import parse_xmp_camera
from utils.view import draw_matches, view_2D_matches, zephyr_to_cv2, create_camera_frustum
from utils.metrics import chamfer_distance, hausdorff_distance, symmetric_partial_rmse, log_se3

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

    reference_K, reference_R, reference_t, dist_coeffs = parse_xmp_camera(
        f'downloads/dante_rework/extrinsics/{reference_image}.xmp')
    target_K, target_R, target_t, _ = parse_xmp_camera(
        f'downloads/dante_rework/extrinsics/{target_image}.xmp')
    target_T = np.identity(4)
    target_T[:3, :3] = np.array(target_R)
    target_T[:3, 3] = np.array(target_t).flatten()

    reference = cv2.rotate(cv2.imread(reference_image_path), cv2.ROTATE_90_CLOCKWISE)
    target = cv2.rotate(cv2.imread(target_image_path), cv2.ROTATE_90_CLOCKWISE)

    reference_visibility = visibility_map[reference_image]
    target_visibility = visibility_map[target_image]
    reference_2D_feature_points = np.asarray([*map(lambda vis_entry: [vis_entry['w'], vis_entry['h']], reference_visibility)])
    reference_2D_feature_points = reference_2D_feature_points[:, ::-1]

    target_2D_feature_points = np.asarray([*map(lambda vis_entry: [vis_entry['w'], vis_entry['h']], target_visibility)])
    target_2D_feature_points = target_2D_feature_points[:, ::-1]
    target_3D_feature_points = np.asarray([*map(lambda vis_entry: pc_points[vis_entry['index'], :], target_visibility)])


    # Open3D section
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("Dante scene",  int(1024/2), int(768/2))
    dante_scene_widget = gui.SceneWidget()
    dante_scene_widget.scene = rendering.Open3DScene(window.renderer)
    window.add_child(dante_scene_widget)

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsics.set_intrinsics(reference.shape[1], reference.shape[0], 5859.98222223, 5859.98222223, 2766.82815988, 1833.34832266)

    # this will eventually be used in Open3D
    geometries = []
    geometries.append(pc)

    target_camera_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    target_camera_axis.transform(np.linalg.inv(target_T))
    frustum_target, _ = create_camera_frustum(camera_intrinsics,  np.linalg.inv(target_T), reference, color=[0, 1, 0])

    dante_scene_widget.scene.add_geometry("Dante point cloud", pc, rendering.MaterialRecord())
    dante_scene_widget.scene.add_geometry("target camera axis", target_camera_axis, rendering.MaterialRecord())
    dante_scene_widget.scene.add_geometry("target camera frustum", frustum_target, rendering.MaterialRecord())

    geometries.append(target_camera_axis)
    geometries.append(frustum_target)

    if False:
        view_2D_matches(reference_2D_feature_points, 'test',
                        bg_image=reference)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # extracting 2D matches

    extraction_matching_result = dict()
    extraction_matching_result['RDD'] = evaluateRDD(reference, target)
    extraction_matching_result['LiftFeat'] = evaluateLiftFeat(reference, target)
    extraction_matching_result['Mast3r'] = evaluateMast3r(reference, target)

    # 2D-2D matches to 2D-3D matches

    for extractor in extraction_matching_result.keys():

        logger.info(f'Working on matches from {extractor}')
        # select the matches for the current extractor
        matches = extraction_matching_result[extractor]

        if False:
            # show the matches
            canvas = draw_matches(matches['reference_matches'], matches['target_matches'], reference, target)
 
            cv2.namedWindow(f"{extractor} matches", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"{extractor} matches", 800, 600)
            cv2.imshow(f"{extractor} matches", canvas)

            cv2.waitKey(0)
            cv2.destroyAllWindows()


        reference_2D_feature_points_to_extractor_matches_distance = cdist(
            matches['reference_matches'], reference_2D_feature_points)
        indexes = np.argmin(
            reference_2D_feature_points_to_extractor_matches_distance, axis=1)

        filtered_indexes = []

        filtered_reference_matches = []
        filtered_target_matches = []
        filtered_target_3D_matches = []

        logger.debug(
            reference_2D_feature_points_to_extractor_matches_distance.shape)

        for match_idx, feat_idx in enumerate(indexes.tolist()):
            # since the 2D-2D matches to features on the reference images may be not even close we filter them base on the euclidean norm
            # this is useful even when using the solvePnP with RANSAC
            if (reference_2D_feature_points_to_extractor_matches_distance[match_idx, feat_idx] < 100):
                filtered_indexes.append([match_idx, feat_idx])
                filtered_reference_matches.append(
                    matches['reference_matches'][match_idx, :])
                filtered_target_matches.append(
                    matches['target_matches'][match_idx, :])
                filtered_target_3D_matches.append(
                    pc_points[reference_visibility[feat_idx]['index']])
            
        if False:
            extractor_filtered_canvas = draw_matches(
                filtered_reference_matches, filtered_target_matches, reference, target)
            cv2.namedWindow("extractor filtered matches", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("extractor filtered matches", 800, 600)
            cv2.imshow("extractor filtered matches", extractor_filtered_canvas)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if False:
            # shows reference matches grouped by feature point

            bg = view_2D_matches(reference_2D_feature_points, 'Grouped matches',
                                 bg_image=reference, show=False).copy()
            img_h, img_w = bg.shape[:2]

            for match_idx, feat_idx in filtered_indexes:
                color = np.random.randint(0, 255, 3).tolist()

                cv2.circle(bg, (int(matches['reference_matches'][match_idx, 0]), int(matches['reference_matches'][match_idx, 1])), radius=3,
                           color=tuple(color), thickness=-1)

                cv2.line(bg,
                         (
                            int(matches['reference_matches'][match_idx, 0]),
                            int(matches['reference_matches'][match_idx, 1])
                        ), (
                            int(reference_2D_feature_points[feat_idx, 0]),
                            int(reference_2D_feature_points[feat_idx, 1])
                        ),
                    color=tuple(color),
                    thickness=2
                )

            cv2.namedWindow(f'{extractor} grouped matches', cv2.WINDOW_KEEPRATIO)
            cv2.imshow(f'{extractor} grouped matches', bg)
            cv2.resizeWindow(f'{extractor} grouped matches', img_w // 2, img_h // 2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        filtered_target_3D_matches = np.asarray(
            filtered_target_3D_matches, dtype=np.float32)
        filtered_target_matches = np.asarray(
            filtered_target_matches, dtype=np.float32)

        # logger.info(filtered_target_matches.shape)
        # logger.info(filtered_target_3D_matches.shape)

        is_target_pose_ok, target_rvec_estimated, target_t_estimated, inliers = cv2.solvePnPRansac(
            filtered_target_3D_matches, filtered_target_matches, cameraMatrix=target_K, distCoeffs=dist_coeffs, reprojectionError=60, iterationsCount=100000,  flags=cv2.SOLVEPNP_P3P)

        target_3D_inliers = filtered_target_3D_matches[inliers, :]
        target_3D_inliers = target_3D_inliers.squeeze()
        target_2D_matches_inliers = filtered_target_matches[inliers, :]
        target_2D_matches_inliers = target_2D_matches_inliers.squeeze()
        
        print(target_3D_inliers.shape)
        print(target_2D_matches_inliers.shape)

        target_R_estimated, _ = cv2.Rodrigues(target_rvec_estimated)
        target_T_estimated = np.identity(4)
        target_T_estimated[:3, :3] = target_R_estimated
        target_T_estimated[:3, 3] = target_t_estimated.flatten()

        logger.success(f"pose estimation completed") if is_target_pose_ok else logger.error('pose estimation failed')

        target_estimated_reproj, _ = cv2.projectPoints(
            pc_points, target_rvec_estimated, target_t_estimated, cameraMatrix=reference_K, distCoeffs=dist_coeffs)
        target_estimated_reproj = target_estimated_reproj.squeeze()
        
        # print([matches['target_matches'].shape, target_estimated_reproj.shape])
        
        if False:
            # Compute and print errors
            chamfer_error = chamfer_distance(target_2D_feature_points, target_estimated_reproj)
            hausdorff_error = hausdorff_distance(target_2D_feature_points, target_estimated_reproj)
            symmetric_partial_rmse_error = symmetric_partial_rmse(target_2D_feature_points, target_estimated_reproj)

            log_se3_error = log_se3(target_T @ np.linalg.inv(target_T_estimated))
            logger.info(f'\nErrors ({extractor}):\n    chamfer: {chamfer_error}\n    hausdorff: {hausdorff_error}\n    symmetric partial RMSE: {symmetric_partial_rmse_error}\n    log_se3: {log_se3_error}\n    log_se3 norm: {np.linalg.norm(log_se3_error)}')

        # logger.info(f'Mean distance ({extractor}): {np.linalg.norm(np.mean(target_2D_feature_points - target_estimated_reproj, axis=0))}')

        # print(target_T)
        # print(np.linalg.inv(target_T))
        # print(target_T_estimated)
        # print(np.linalg.inv(target_T_estimated))

        if True:
            img = view_2D_matches(target_2D_feature_points, f'Target estimated reprojection ({extractor})', bg_image=target, show=False, color=(0,255,0))
            view_2D_matches(target_estimated_reproj, f'Target estimated reprojection ({extractor})', bg_image=img)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        # Open3D geometries
        target_camera_axis_estimated = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        target_camera_axis_estimated.transform(np.linalg.inv(target_T_estimated))
        frustum_target_estimated, _ = create_camera_frustum(camera_intrinsics, np.linalg.inv(target_T_estimated), reference)

        # frame_label = o3d.visualization.gui.Label3D(extractor, target_t_estimated)

        geometries.append(target_camera_axis_estimated)
        geometries.append(frustum_target_estimated)
        
        # label = gui.Label3D(extractor)
        label = dante_scene_widget.add_3d_label(np.linalg.inv(target_T_estimated)[:3, 3], extractor)
        label.scale = 1.0

        dante_scene_widget.scene.add_geometry(f"{extractor} estimated target camera axis", target_camera_axis_estimated, rendering.MaterialRecord())
        dante_scene_widget.scene.add_geometry(f"{extractor} estimated target camera frustum", frustum_target_estimated, rendering.MaterialRecord())

        # geometries.append(frame_label)

    if False:
        # shows poses in Open3D
        camera = dante_scene_widget.scene.camera
        camera.look_at([0, 1, 0], np.linalg.inv(target_T)[:3, 3], np.linalg.inv(target_R)[:3, 2])

        gui.Application.instance.run()
        # o3d.visualization.draw_geometries(geometries)