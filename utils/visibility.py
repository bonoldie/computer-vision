import os
import re 
import numpy as np
import cv2

def load_visibility(filepath):
    vis_map = {}

    with open(filepath, "r") as vis_file:
        vis_file_content = vis_file.read()
        
        # Each match is a tuple as (camera_name, visibility)
        for match in  re.findall(r"Visibility for camera (.*)[.\r\n]([0-9 .\r\n]+)", vis_file_content):
            vis_rows = match[1].split("\n")[1:-1]

        #WATCH OUT for what is width and wat is height
            vis_rows = [np.asarray(row.split(" "), np.float64) for row in vis_rows]
            vis_rows = [{"index": row[0].astype(np.int64), "w": row[2], "h": row[1]} for row in vis_rows]

            vis_map[match[0]] = vis_rows

    return vis_map
        
#visualize the visibility map on top of the corr. image
def visualize_visibility(vis_map, image_path, radius):
    image_name = os.path.basename(image_path)

    if image_name not in vis_map:
        print(f"Image '{image_name}' not found in visibility map.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image from '{image_path}'.")
        return

    h_img,w_img = img.shape[:2]


    ws = [p["w"] for p in vis_map[image_name]]
    hs = [p["h"] for p in vis_map[image_name]]
    print(f"w: min={min(ws):.2f}, max={max(ws):.2f}")
    print(f"h: min={min(hs):.2f}, max={max(hs):.2f}")

    for p in vis_map[image_name]:
        w, h = int(round(p["w"])), int(round(p["h"]))
        if 0 <= w < w_img and 0 <= h < h_img:
            cv2.circle(img, (w, (h_img-h)), radius, (20, 128, 255), thickness=-1)  # green filled circle

    resized_img = cv2.resize(img, (round(w_img/5),round(h_img/5)))
    cv2.imshow(f"Visibility - {image_name}", resized_img)
    #cv2.imshow(f"Visibility - {image_name}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   
    
    
def count_matches(source_matches, target_matches, image_path, search_radius):
    image_name = os.path.basename(image_path)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image from '{image_path}'.")
        return

    h_img,w_img = img.shape[:2]
    
    src_img = np.zeros((h_img, w_img), dtype=np.uint8)
    tgt_img = np.zeros((h_img, w_img), dtype=np.uint8)

    # Place source match points
    for pt in source_matches:
        w, h = int(round(pt["w"])), int(round(pt["h"]))
        if 0 <= w < w_img and 0 <= h < h_img:
            #src_img[h, w] = 1
            cv2.circle(src_img, (w, h), search_radius,(255, 0, 0), thickness=-1)  # filled circle
        else:
            print(f"Source match point ({w}, {h}) is out of bounds for image size ({w_img}, {h_img}).")
    
    cv2.imwrite("source_matches.jpeg", src_img)
            

    # Place target match points
    for pt in target_matches:
        w, h = int(round(pt[0])), int(round(pt[1]))
        if 0 <= w < w_img and 0 <= h < h_img:
            #tgt_img[h, w] = 1
            cv2.circle(tgt_img, (w, h), search_radius,(255, 0, 0), thickness=-1)
        else:
            print(f"Target match point ({w}, {h}) is out of bounds for image size ({w_img}, {h_img}).")
            
    cv2.imwrite("target_matches.jpeg", tgt_img)
    


