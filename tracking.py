'''
Script for calculating marker positions from a pair of calibrated camera images.
'''
import arucoX as ax
import matplotlib.pyplot as plt
import numpy as np

# Initialize camera modules
from camera_params import INTRINSICS_DICT, EXTRINSICS_DICT, CAMERA_IDS_DICT

current_cameras = ['815412070907', '818312070212']
MARKER_IDS = [7, 6]
MARKER_SIZES = [0.05, 0.05]

def get_markers_pos(imgs_ti, intrinsics=None, extrinsics=None, marker_ids=[7, 6], marker_sizes=[0.05, 0.05]):
    c1 = ax.CameraModule()
    c2 = ax.CameraModule()
    cameras = [c1, c2]
    scene = ax.Scene(cameras=cameras)

    for i in range(len(marker_ids)):
        scene.register_marker_size(marker_ids[i], marker_sizes[i])

    if intrinsics is None:
        intrinsics = [INTRINSICS_DICT[i] for i in current_cameras]
    if extrinsics is None:
        extrinsics = [EXTRINSICS_DICT[i] for i in current_cameras]

    for i in range(len(cameras)):
        matrix, dist_coeffs = intrinsics[i]
        cameras[i].set_intrinsics(matrix=matrix, dist_coeffs=dist_coeffs)
        scene.cameras[i] = scene.cameras[i]._replace(pose=extrinsics[i])

    markers_pos = {}
    for m in marker_ids:
        markers_pos[m] = np.array([0, 0, 0])
    markers = scene.detect_markers(imgs_ti)
    for marker in markers:
        if marker.id in marker_ids:
            pos, quat = ax.utils.se3_to_xyz_quat(marker.pose)
            markers_pos[marker.id] = pos

    return markers_pos



