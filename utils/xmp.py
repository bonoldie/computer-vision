import xml.etree.ElementTree as ET
import numpy as np

def parse_xmp_camera(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # ---- Intrinsic parameters ----
    calib = root.find("calibration")
    fx = float(calib.attrib["fx"])
    fy = float(calib.attrib["fy"])
    cx = float(calib.attrib["cx"])
    cy = float(calib.attrib["cy"])
    skew = float(calib.attrib.get("skew", 0.0))

    intrinsics = np.array([
        [fx, skew, cx],
        [0,  fy,   cy],
        [0,   0,    1]
    ])

    # ---- Distortion coefficients ----
    k1 = float(calib.attrib.get("k1", 0.0))
    k2 = float(calib.attrib.get("k2", 0.0))
    p1 = float(calib.attrib.get("p1", 0.0))
    p2 = float(calib.attrib.get("p2", 0.0))
    k3 = float(calib.attrib.get("k3", 0.0))

    dist_coeffs = np.array([k1, k2, p1, p2, k3])

    # ---- Rotation matrix ----
    rotation_text = root.find("extrinsics/rotation").text.strip().split()
    rotation = np.array(list(map(float, rotation_text))).reshape((3, 3))

    # ---- Translation vector ----
    translation_text = root.find("extrinsics/translation").text.strip().split()
    translation = np.array(list(map(float, translation_text))).reshape((3, 1))

    return intrinsics, rotation, translation, dist_coeffs
