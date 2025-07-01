import os
import re 
import numpy as np

def load_visibility(filepath):
    vis_map = {}

    with open(filepath, "r") as vis_file:
        vis_file_content = vis_file.read()
        
        # Each match is a tuple as (camera_name, visibility)
        for match in  re.findall(r"Visibility for camera (.*)[.\r\n]([0-9 .\r\n]+)", vis_file_content):
            vis_rows = match[1].split("\n")[1:-1]

            vis_rows = [np.asarray(row.split(" "), np.float64) for row in vis_rows]
            vis_rows = [{"index": row[0].astype(np.int64), "w": row[1], "h": row[2]} for row in vis_rows]

            vis_map[match[0]] = vis_rows

    return vis_map
        

        
    


