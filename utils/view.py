import numpy as np
import cv2

def convert_coords(coords: np.ndarray, width: int, height: int) -> np.ndarray:

    converted = np.empty_like(coords)
    converted[:, 0] = coords[:, 0] # width - 1 -   # Flip x (columns)
    converted[:, 1] = height - 1 - coords[:, 1]  # Flip y (rows)
    return converted

def view_2D_matches(matches, name='',point_radius = 3, shape=(500, 500),bg=None):
    image = np.ones((shape[1],  shape[0], 3), dtype=np.uint8) * 255

    if bg is not None:
        image = cv2.resize(bg, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

    rounded_matches = np.rint(matches).astype(int)

    

    # Plot each point
    for point in rounded_matches:
        x, y = point
        cv2.circle(image, (x, y), radius=point_radius, color=(0, 0, 255), thickness=-1)

    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, image)
    cv2.resizeWindow(name, 800, 800)
    
    return image