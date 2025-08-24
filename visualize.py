import open3d as o3d
import copy
import cv2
import numpy as np

from utils.visibility import load_visibility
from utils.view import compute_camera_matrix, view_2D_matches, zephyr_to_cv2


_SAM1005_zephyr_R = np.reshape(np.fromstring(
    '1.38777878078e-17 -0.157592817354 0.98750417919 0.997723318127 -0.0665974785439 -0.010628090993 0.0674401991883 0.985255946326 0.157234028644', sep=' ', dtype=np.float64), (3, 3))
_SAM1005_zephyr_t = np.fromstring(
    '-3.63418597003 1.1289630973 6.95687936484', sep=' ', dtype=np.float64)


_SAM1073_zephyr_R = np.reshape(np.fromstring(
    '-0.0376026577346 -0.11992473137 0.99207061187 -0.987749875042 -0.145998407495 -0.0550876516489 0.151447101281 -0.981989065018 -0.112965710278', sep=' ', dtype=np.float64), (3, 3))
_SAM1073_zephyr_t = np.fromstring(
    '-6.26692379871 -0.986993936906 6.41601378017', sep=' ', dtype=np.float64)


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
            [0, c, s],
            [0, -s,  c]
        ])
    elif axis.lower() == 'y':
        return np.array([
            [c, 0, -s],
            [0, 1, 0],
            [s, 0, c]
        ])
    elif axis.lower() == 'z':
        return np.array([
            [c, s, 0],
            [-s,  c, 0],
            [0,  0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

def create_camera_frustum(intrinsic, extrinsic, img, scale=0.3, color=[1, 0, 0]):
    """
    Create a camera frustum with an image at its center.
    - intrinsic: o3d.camera.PinholeCameraIntrinsic
    - extrinsic: 4x4 np.array (T_camera_to_world)
    - img_path: path to the image
    """
    width = intrinsic.width
    height = intrinsic.height
    fx = intrinsic.get_focal_length()[0]
    fy = intrinsic.get_focal_length()[1]
    cx = intrinsic.get_principal_point()[0]
    cy = intrinsic.get_principal_point()[1]

    # Compute corners of the image plane in camera space
    z = scale
    corners = np.array([
        [(0 - cx) * z / fx, (0 - cy) * z / fy, z],
        [(width - cx) * z / fx, (0 - cy) * z / fy, z],
        [(width - cx) * z / fx, (height - cy) * z / fy, z],
        [(0 - cx) * z / fx, (height - cy) * z / fy, z]
    ])
    origin = np.array([[0, 0, 0]])
    points_cam = np.vstack((origin, corners))

    # Apply camera-to-world transform
    points_world = (extrinsic @ np.hstack((points_cam, np.ones((5,1)))).T).T[:, :3]

    # Define frustum lines
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # from origin to corners
        [1, 2], [2, 3], [3, 4], [4, 1]   # image plane edges
    ]
    colors = [color for _ in lines]  # red lines

    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_world),
        lines=o3d.utility.Vector2iVector(lines)
    )
    frustum.colors = o3d.utility.Vector3dVector(colors)

    # Optional: add the image as a textured plane at the image plane
    # Load image and convert to Open3D format
    img_resized = np.resize(img,(int(width * 0.2), int(height * 0.2)))
    img_np = np.asarray(img_resized) / 255.0
    if img_np.ndim == 2:  # grayscale
        img_np = np.repeat(img_np[:, :, None], 3, axis=2)

    texture = o3d.geometry.Image((img_np * 255).astype(np.uint8))

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points_world[1:])
    mesh.triangles = o3d.utility.Vector3iVector([[0,1,2],[0,2,3]])
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([1,1,1])

    return frustum, mesh


if __name__ == '__main__':

    # loads visibility
    visibility_map_path = 'downloads/dante_rework/Visibility.txt'
    visibility_map = load_visibility(visibility_map_path)

    # loads reworked point cloud
    pc = o3d.io.read_point_cloud('downloads/dante_rework/SamPointCloud.ply')
    # pc_original = o3d.io.read_point_cloud('downloads/dante_dataset/dante_dataset/SamPointCloud.ply')
    pc_points = np.asarray(pc.points)

    # loads media
    _SAM1005 = cv2.imread(
        'downloads/dante_dataset/dante_dataset/photos/_SAM1005.JPG')

    _SAM1073 = cv2.imread(
        'downloads/dante_dataset/dante_dataset/photos/_SAM1073.JPG')
    
    # _SAM1005 = _SAM1073


    _SAM1005_visibility = visibility_map.get('_SAM1005.JPG')
    _SAM1073_visibility = visibility_map.get('_SAM1073.JPG')

    #_SAM1005_visibility = _SAM1073_visibility

    #_SAM1005_zephyr_R = _SAM1073_zephyr_R
    #_SAM1005_zephyr_t = _SAM1073_zephyr_t

    # T_fix = np.identity(4)
    # T_fix[:3, :3] = np.linalg.inv(_SAM1005_zephyr_R) @ rotation_matrix('z', 90.0)


    # _SAM1005_zephyr_R = _SAM1005_zephyr_R
    

    _SAM1005_visibility_array = np.asarray(
        [*map(lambda vis_entry: [vis_entry['w'], vis_entry['h']], _SAM1005_visibility)])
    _SAM1005_model_points = np.asanyarray(
        [*map(lambda visibility_entry: pc_points[visibility_entry['index']], _SAM1005_visibility)])

    success, r_vec, t_vec = cv2.solvePnP(_SAM1005_model_points, _SAM1005_visibility_array[:, ::-1], cameraMatrix=dante_camera_matrix, distCoeffs=dante_dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    _SAM1005_model_reproj, _ = cv2.projectPoints(
        pc_points, r_vec, t_vec, cameraMatrix=dante_camera_matrix, distCoeffs=dante_dist_coeffs)
    _SAM1005_model_reproj = np.asarray(_SAM1005_model_reproj).squeeze(axis=1)

    zephyr_r_vec, _ = cv2.Rodrigues(_SAM1005_zephyr_R)
    zephyr_t_vec = _SAM1005_zephyr_t
    _SAM1005_model_zephyr_reproj, _ = cv2.projectPoints(
        pc_points, zephyr_r_vec, zephyr_t_vec, cameraMatrix=dante_camera_matrix, distCoeffs=dante_dist_coeffs)
    _SAM1005_model_zephyr_reproj = np.asarray(
        _SAM1005_model_zephyr_reproj).squeeze(axis=1)
    

    print('####################\nzephyr\n')
    print(zephyr_r_vec)
    print(zephyr_t_vec)

    print('####################\ncv2\n')
    print(r_vec)
    print(t_vec)

    # view_2D_matches(zephyr_to_cv2(_SAM1005_visibility_array, width=_SAM1005.shape[1], height=_SAM1005.shape[0]), plot_name='vis', bg_image=_SAM1005)

    view_2D_matches(_SAM1005_model_reproj, plot_name='cv2 pose', bg_image=cv2.rotate(_SAM1005,  cv2.ROTATE_90_CLOCKWISE))

    # view_2D_matches(_SAM1005_model_zephyr_reproj, plot_name='zephyr pose', bg_image=cv2.rotate(_SAM1005,  cv2.ROTATE_90_CLOCKWISE)) # with cv2.rotate(_SAM1005,  cv2.ROTATE_90_CLOCKWISE) the projection is aligned
    view_2D_matches(_SAM1005_model_zephyr_reproj, plot_name='zephyr pose', bg_image=cv2.rotate(_SAM1005,  cv2.ROTATE_90_CLOCKWISE)) # with cv2.rotate(_SAM1005,  cv2.ROTATE_90_CLOCKWISE) the projection is aligned

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    geometries = []

    # cameras POV

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsics.set_intrinsics(_SAM1005.shape[1], _SAM1005.shape[0], 5794.23954825, 5794.23954825, 2792.29091924, 1817.35747811)

    cv2_transformation = np.identity(4)
    R_cv2, _ = cv2.Rodrigues(r_vec)
    cv2_transformation[:3, :3] = np.asarray(R_cv2)
    cv2_transformation[:3, 3] = np.asarray(t_vec).flatten()


    zephyr_transformation = np.identity(4)
    zephyr_transformation[:3, :3] = _SAM1005_zephyr_R
    zephyr_transformation[:3, 3] = _SAM1005_zephyr_t.flatten()
    
    # WARNING: these are Word-to-camera transformations
    cv2_camera_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    cv2_camera_axis.transform(np.linalg.inv(cv2_transformation))

    T_fix = np.identity(4)
    T_fix[:3, :3] = rotation_matrix('y', 90)

    zephyr_camera_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    zephyr_camera_axis.transform(np.linalg.inv(zephyr_transformation ))


    print('final T_world_to_camera')
    print('cv2')
    print(np.linalg.inv(cv2_transformation))
    print('zephyr')
    print(np.linalg.inv(zephyr_transformation))

    #zephyr_camera_axis_fix = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    #zephyr_camera_axis_fix.transform(np.linalg.inv(zephyr_transformation))
    #zephyr_camera_axis_fix.rotate(_SAM1005_zephyr_R @ T_fix[:3, :3])


    #zephyr_camera_axis_fix = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    #zephyr_camera_axis_fix.transform(np.linalg.inv(zephyr_transformation))
    #zephyr_camera_axis_fix.rotate(rotation_matrix('y', 90.0))

    # zephyr_camera_axis.translate(_SAM1005_zephyr_t.flatten())
    # zephyr_camera_axis.rotate(_SAM1005_zephyr_R)

    
    geometries.append(zephyr_camera_axis)
    #geometries.append(zephyr_camera_axis_fix)
    geometries.append(cv2_camera_axis)

    frustum_cv2, _ = create_camera_frustum(camera_intrinsics, np.linalg.inv(cv2_transformation), _SAM1005)
    frustum_zephyr, _ = create_camera_frustum(camera_intrinsics,  np.linalg.inv(zephyr_transformation), _SAM1005, color=[0, 1, 0])
    geometries.append(frustum_cv2)
    geometries.append(frustum_zephyr)

    # append origin
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))

    # Zephyr point cloud
    pc_zephyr = copy.deepcopy(pc)

    pc_zephyr.transform(zephyr_transformation)

    geometries.append(pc)
    # geometries.append(pc_zephyr)

    o3d.visualization.draw_geometries(geometries)

    #origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

    #rot_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    #rot_axis.rotate(rotation_matrix('x', 45))

    #rot_axis_2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    # rot_axis_2.rotate(rotation_matrix('x', 45) @ rotation_matrix('y', 45))
    # rot_axis_2.rotate(rotation_matrix('y', 45))
    

    # o3d.visualization.draw_geometries([origin, rot_axis_2])

    # Display transformed point cloud

    # CV2 transformation
    #pc_cv2 = copy.deepcopy(pc)
    #R_cv2, _ = cv2.Rodrigues(r_vec)

    #pc_cv2_transformation = np.identity(4)
    # pc_cv2_transformation[:3, :3] = np.asarray(R_cv2)
    #pc_cv2_transformation[:3, 3] = np.asarray(t_vec).flatten()
    #pc_cv2.transform(pc_cv2_transformation)

    #colors = np.asarray(pc_cv2.colors)
    #colors[:, 0] = 255
    #pc_cv2.colors = o3d.utility.Vector3dVector(colors)

    # Zephyr transformation
    #pc_zephyr = copy.deepcopy(pc)

    # pc_zephyr_transformation = np.identity(4)
    # pc_zephyr_transformation[:3, :3] = rotation_matrix('z', 90.0) 
    # pc_zephyr_transformation[:3, 3] = np.zeros((3,1)) # rotation_matrix('z', -90.0) 
    # pc_zephyr.rotat(pc_zephyr_transformation)

    # pc_zephyr.rotate(rotation_matrix('z', -90.0), center=(0, 0, 0))

    #pc_zephyr_transformation = np.identity(4)
    # pc_zephyr_transformation[:3, :3] = np.asarray(_SAM1005_zephyr_R)  # rotation_matrix('z', -90.0) @ 
    #pc_zephyr_transformation[:3, 3] = _SAM1005_zephyr_t.flatten()   # rotation_matrix('z', -90.0) @ 
    #pc_zephyr.transform(pc_zephyr_transformation)

    # pc_zephyr.rotate(rotation_matrix('z', 90.0), center=_SAM1005_zephyr_t.flatten())

    #colors = np.asarray(pc_zephyr.colors)
    #colors[:, 1] = 255
    #pc_zephyr.colors = o3d.utility.Vector3dVector(colors)

    # pc_cv2_points = np.asarray(pc_cv2.points)

    #axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)
    #o3d.visualization.draw_geometries([pc, pc_cv2, pc_zephyr, axis])
    
 
