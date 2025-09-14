import open3d as o3d
import copy
import cv2
import numpy as np

from utils.visibility import load_visibility
from utils.view import compute_camera_matrix, view_2D_matches, zephyr_to_cv2


PN_8227_zephyr_R = np.reshape(np.fromstring(
    '0.00622278311968 -0.999978189541 0.00221300986316 0.0181916332187 -0.00209948197554 -0.999832314269 0.999815153641 0.00626199791148 0.0181781718298', sep=' ', dtype=np.float64), (3, 3))
PN_8227_zephyr_t = np.fromstring(
    '7.59215368986 13.090140508 45.6693151546', sep=' ', dtype=np.float64)

dante_camera_matrix = compute_camera_matrix(
    4292.80953167, 4292.80953167, 2611.14749763, 1693.79524843)

dante_dist_coeffs = np.array([
    0.0,  # k1
    0.0,  # k2
    0.0,  # p1
    0.0,  # p2
    0.0   # k3
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
    visibility_map_path = 'downloads/PortaNuova/Visibility.txt'
    visibility_map = load_visibility(visibility_map_path)

    # loads reworked point cloud
    pc = o3d.io.read_point_cloud('downloads/PortaNuova/PN_PointCloud.ply')
    # pc_original = o3d.io.read_point_cloud('downloads/dante_dataset/dante_dataset/SamPointCloud.ply')
    pc_points = np.asarray(pc.points)

    # loads media
    PN_8227 = cv2.imread(
        'downloads/PortaNuova/PN_8227.JPG')

    _SAM1073 = cv2.imread(
        'downloads/dante_dataset/dante_dataset/photos/_SAM1073.JPG')
    
    # PN_8227 = _SAM1073


    PN_8227_visibility = visibility_map.get('IMG_8227.JPG')
    _SAM1073_visibility = visibility_map.get('_SAM1073.JPG')

    #PN_8227_visibility = _SAM1073_visibility

    #PN_8227_zephyr_R = _SAM1073_zephyr_R
    #PN_8227_zephyr_t = _SAM1073_zephyr_t

    # T_fix = np.identity(4)
    # T_fix[:3, :3] = np.linalg.inv(PN_8227_zephyr_R) @ rotation_matrix('z', 90.0)


    # PN_8227_zephyr_R = PN_8227_zephyr_R
    

    PN_8227_visibility_array = np.asarray(
        [*map(lambda vis_entry: [vis_entry['w'], vis_entry['h']], PN_8227_visibility)])
    PN_8227_model_points = np.asanyarray(
        [*map(lambda visibility_entry: pc_points[visibility_entry['index']], PN_8227_visibility)])

    success, r_vec, t_vec = cv2.solvePnP(PN_8227_model_points, PN_8227_visibility_array[:, ::-1], cameraMatrix=dante_camera_matrix, distCoeffs=dante_dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    PN_8227_model_reproj, _ = cv2.projectPoints(
        pc_points, r_vec, t_vec, cameraMatrix=dante_camera_matrix, distCoeffs=dante_dist_coeffs)
    PN_8227_model_reproj = np.asarray(PN_8227_model_reproj).squeeze(axis=1)

    zephyr_r_vec, _ = cv2.Rodrigues(PN_8227_zephyr_R)
    zephyr_t_vec = PN_8227_zephyr_t
    PN_8227_model_zephyr_reproj, _ = cv2.projectPoints(
        pc_points, zephyr_r_vec, zephyr_t_vec, cameraMatrix=dante_camera_matrix, distCoeffs=dante_dist_coeffs)
    PN_8227_model_zephyr_reproj = np.asarray(
        PN_8227_model_zephyr_reproj).squeeze(axis=1)
    

    print('####################\nzephyr\n')
    print(zephyr_r_vec)
    print(zephyr_t_vec)

    print('####################\ncv2\n')
    print(r_vec)
    print(t_vec)


    # view_2D_matches(PN_8227_visibility_array[:, ::-1], plot_name='vis', bg_image=PN_8227)

    view_2D_matches(PN_8227_model_reproj, plot_name='cv2 pose', bg_image=PN_8227)

    # view_2D_matches(PN_8227_model_zephyr_reproj, plot_name='zephyr pose', bg_image=cv2.rotate(PN_8227,  cv2.ROTATE_90_CLOCKWISE)) # with cv2.rotate(PN_8227,  cv2.ROTATE_90_CLOCKWISE) the projection is aligned
    view_2D_matches(PN_8227_model_zephyr_reproj, plot_name='zephyr pose', bg_image=PN_8227) # with cv2.rotate(PN_8227,  cv2.ROTATE_90_CLOCKWISE) the projection is aligned

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    geometries = []

    # cameras POV

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsics.set_intrinsics(PN_8227.shape[1], PN_8227.shape[0], 4292.80953167, 4292.80953167, 2611.14749763, 1693.79524843)

    cv2_transformation = np.identity(4)
    R_cv2, _ = cv2.Rodrigues(r_vec)
    cv2_transformation[:3, :3] = np.asarray(R_cv2)
    cv2_transformation[:3, 3] = np.asarray(t_vec).flatten()


    zephyr_transformation = np.identity(4)
    zephyr_transformation[:3, :3] = PN_8227_zephyr_R
    zephyr_transformation[:3, 3] = PN_8227_zephyr_t.flatten()
    
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
    #zephyr_camera_axis_fix.rotate(PN_8227_zephyr_R @ T_fix[:3, :3])


    #zephyr_camera_axis_fix = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    #zephyr_camera_axis_fix.transform(np.linalg.inv(zephyr_transformation))
    #zephyr_camera_axis_fix.rotate(rotation_matrix('y', 90.0))

    # zephyr_camera_axis.translate(PN_8227_zephyr_t.flatten())
    # zephyr_camera_axis.rotate(PN_8227_zephyr_R)

    
    geometries.append(zephyr_camera_axis)
    #geometries.append(zephyr_camera_axis_fix)
    geometries.append(cv2_camera_axis)

    frustum_cv2, _ = create_camera_frustum(camera_intrinsics, np.linalg.inv(cv2_transformation), PN_8227)
    frustum_zephyr, _ = create_camera_frustum(camera_intrinsics,  np.linalg.inv(zephyr_transformation), PN_8227, color=[0, 1, 0])
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
    # pc_zephyr_transformation[:3, :3] = np.asarray(PN_8227_zephyr_R)  # rotation_matrix('z', -90.0) @ 
    #pc_zephyr_transformation[:3, 3] = PN_8227_zephyr_t.flatten()   # rotation_matrix('z', -90.0) @ 
    #pc_zephyr.transform(pc_zephyr_transformation)

    # pc_zephyr.rotate(rotation_matrix('z', 90.0), center=PN_8227_zephyr_t.flatten())

    #colors = np.asarray(pc_zephyr.colors)
    #colors[:, 1] = 255
    #pc_zephyr.colors = o3d.utility.Vector3dVector(colors)

    # pc_cv2_points = np.asarray(pc_cv2.points)

    #axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)
    #o3d.visualization.draw_geometries([pc, pc_cv2, pc_zephyr, axis])
    
 
