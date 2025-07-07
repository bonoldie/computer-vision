import numpy as np
import cv2

def convert_coords(coords: np.ndarray, width: int, height: int) -> np.ndarray:

    converted = np.empty_like(coords)
    converted[:, 0] = coords[:, 0] # width - 1 -   # Flip x (columns)
    converted[:, 1] = height - 1 - coords[:, 1]  # Flip y (rows)
    return converted

def view_2D_matches(matches, plot_name='',point_radius = 3 , bg_image=[]):
    
    matches = np.asarray(matches)
    if matches.shape[1] != 2:
        raise ValueError("Each match point must be 2D (x, y).")

    # Use background image if provided
    if isinstance(bg_image, np.ndarray) and bg_image.size > 0:
        image = bg_image.copy()
    else:
        print("No valid background image provided, using default black image.")
        image = np.zeros((800, 800, 3), dtype=np.uint8)

    img_h, img_w = image.shape[:2]

    matches = np.rint(matches).astype(int)

    # Plot each point
    for point in matches:
        x, y = point
        if 0 <= x < img_w and 0 <= y < img_h:
            cv2.circle(image, (x, y), radius=point_radius, color=(0, 0, 255), thickness=-1)
        else:
            print(f"[view_2D_matches] Point {point} is out of bounds for the image size {image.shape[:2]}.")


    window_w, window_h = img_w // 2, img_h // 2
    cv2.namedWindow(plot_name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(plot_name, image)
    cv2.resizeWindow(plot_name, window_w, window_h)
    
    return image