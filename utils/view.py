import numpy as np
import cv2
import open3d as o3d

def zephyr_to_cv2(coords: np.ndarray, width: int, height: int) -> np.ndarray:
    converted = np.empty_like(coords)
    converted[:, 0] = coords[:, 0]               # Keep x (columns)
    converted[:, 1] = height - 1 - coords[:, 1]  # Flip y (rows)
    return converted


def cv2_to_zephyr(coords: np.ndarray, width: int, height: int) -> np.ndarray:
    converted = np.empty_like(coords)
    converted[:, 0] = coords[:, 0]               # Keep x (columns)
    converted[:, 1] = height - 1 - coords[:, 1]  # Flip y (rows)
    return converted


def compute_camera_matrix(fx, fy, cx, cy):
    return np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1.0],
    ])

def draw_matches(ref_points, dst_points, img0, img1):

    if type(img0) is str:
        img0 =  cv2.imread(img0)
    
    if type(img1) is str:
        img1 =  cv2.imread(img1)
    
    # Prepare keypoints and matches for drawMatches function
    keypoints0 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1000) for p in ref_points]
    keypoints1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1000) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(ref_points))]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img0, keypoints0, img1, keypoints1, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches



def view_2D_matches(matches, plot_name='', point_radius=3, bg_image=[], show=True, color = (0,0,255)):

    matches = np.asarray(matches)
    if matches.shape[1] != 2:
        raise ValueError("Each match point must be 2D (x, y).")

    # Use background image if provided
    if isinstance(bg_image, np.ndarray) and bg_image.size > 0:
        image = bg_image.copy()
    elif isinstance(bg_image, str):
        image = cv2.imread(bg_image)
    else:
        print("No valid background image provided, using default black image.")
        image = np.zeros((800, 800, 3), dtype=np.uint8)

    img_h, img_w = image.shape[:2]

    matches = np.rint(matches).astype(int)

    # Plot each point
    for point in matches:
        x, y = point
        if 0 <= x < img_w and 0 <= y < img_h:
            cv2.circle(image, (x, y), radius=point_radius,
                       color=color, thickness=-1)
        else:
            pass
           # print(f"[view_2D_matches] Point {point} is out of bounds for the image size {image.shape[:2]}.")

    if show:
        window_w, window_h = img_w // 2, img_h // 2
        cv2.namedWindow(plot_name, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(plot_name, image)
        cv2.resizeWindow(plot_name, window_w, window_h)

    return image


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