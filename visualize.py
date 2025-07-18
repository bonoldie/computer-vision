import open3d as o3d
import cv2
import numpy as np

from utils.visibility import load_visibility
from utils.view import compute_camera_matrix, view_2D_matches, zephyr_to_cv2
 

_SAM1005_zephyr_R = np.reshape(np.fromstring('1.38777878078e-17 -0.157592817354 0.98750417919 0.997723318127 -0.0665974785439 -0.010628090993 0.0674401991883 0.985255946326 0.157234028644', sep=' ', dtype=np.float64), (3,3))
_SAM1005_zephyr_t = np.fromstring('-3.63418597003 1.1289630973 6.95687936484', sep=' ',dtype=np.float64)

dante_camera_matrix = compute_camera_matrix(
    5794.23954825, 5794.23954825, 2792.29091924, 1817.35747811)

dante_dist_coeffs = np.array([
    -0.049194332605,   # k1
    0.0277098399156,  # k2
    0.0,              # p1
    0.0,              # p2
    0.12963081329     # k3
])

def rotation_matrix(axis: str, angle_degrees: float) -> np.ndarray:
    """
    Returns a 3D rotation matrix for a given axis ('x', 'y', or 'z') and angle in degrees.
    """
    angle = np.radians(angle_degrees)
    c, s = np.cos(angle), np.sin(angle)

    if axis.lower() == 'x':
        return np.array([
            [1, 0,  0],
            [0, c, -s],
            [0, s,  c]
        ])
    elif axis.lower() == 'y':
        return np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c]
        ])
    elif axis.lower() == 'z':
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    

if __name__ == '__main__':

    # loads visibility
    visibility_map_path = 'downloads/dante_rework/Visibility.txt'
    visibility_map = load_visibility(visibility_map_path)

    # loads reworked point cloud
    pc = o3d.io.read_point_cloud('downloads/dante_rework/SamPointCloud.ply')
    pc_points = np.asarray(pc.points)

    pc2 = o3d.io.read_point_cloud('downloads/dante_rework/SamPointCloud.ply')
    pc2_rotation = pc2.get_rotation_matrix_from_axis_angle([0, -np.pi/2, 0])
    pc2.rotate(pc2_rotation, center=(0, 0, 0))
    pc2_points = np.asarray(pc2.points)

    # loads media
    _SAM1005 = cv2.imread('downloads/dante_dataset/dante_dataset/photos/_SAM1005.JPG')

    _SAM1005_visibility = visibility_map.get('_SAM1005.JPG')

    _SAM1005_visibility_array = np.asarray([*map(lambda vis_entry: [vis_entry['w'], vis_entry['h']], _SAM1005_visibility)])    
    _SAM1005_model_points = np.asanyarray([*map(lambda visibility_entry: pc_points[visibility_entry['index']], _SAM1005_visibility)])


    success, r_vec, t_vec = cv2.solvePnP(_SAM1005_model_points, zephyr_to_cv2(_SAM1005_visibility_array, width=_SAM1005.shape[1], height=_SAM1005.shape[0]), cameraMatrix=dante_camera_matrix, distCoeffs=dante_dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    _SAM1005_model_reproj, _ = cv2.projectPoints(pc_points, r_vec, t_vec, cameraMatrix=dante_camera_matrix, distCoeffs=dante_dist_coeffs)
    _SAM1005_model_reproj = np.asarray(_SAM1005_model_reproj).squeeze(axis=1)

    zephyr_r_vec, _ = cv2.Rodrigues(_SAM1005_zephyr_R)
    zephyr_t_vec = _SAM1005_zephyr_t
    _SAM1005_model_zephyr_reproj, _ = cv2.projectPoints(pc_points, zephyr_r_vec, zephyr_t_vec, cameraMatrix=dante_camera_matrix, distCoeffs=dante_dist_coeffs)
    _SAM1005_model_zephyr_reproj = np.asarray(_SAM1005_model_zephyr_reproj).squeeze(axis=1)
    
    view_2D_matches(_SAM1005_model_reproj, plot_name='cv2 pose', bg_image=_SAM1005)

    view_2D_matches(_SAM1005_model_zephyr_reproj, plot_name='zephyr pose', bg_image=cv2.rotate(_SAM1005,  cv2.ROTATE_90_CLOCKWISE)) # with cv2.rotate(_SAM1005,  cv2.ROTATE_90_CLOCKWISE) the projection is aligned

    print('####################\nzephyr\n')
    print(zephyr_r_vec)
    print(zephyr_t_vec)

    print('####################\cv2\n')
    print(r_vec)
    print(t_vec)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)
    # o3d.visualization.draw_geometries([pc, pc2, axis])
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
