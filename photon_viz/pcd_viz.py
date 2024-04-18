"""
    Generate point cloud pairs
    IO utilities and visualization
    @author: Qianyue He
    @date: 12.4
"""
import os
import numpy as np
import open3d as o3d
import configargparse

from typing import Dict
from rich.console import Console
from o3d_utils import make_o3d_app, create_geometry_at_points

COLORS = [(0.7, 0, 0), (0, 0.7, 0), (0., 0.3, 0.7), (0.7, 0.5, 0), (1, 1, 1), (0.7, 0, 0.5)]
CONSOLE = Console(width = 128)

def get_options(delayed_parse = False):
    # IO parameters
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config',  
                                     is_config_file=True, help='Config file path')
    parser.add_argument("--pcd_path",      required = True,  help = "point cloud path", type = str)
    parser.add_argument("--surf_prefix",   default = "surf", help = "surface point cloud prefix", type = str)
    parser.add_argument("--volm_prefix",   default = "vol",  help = "volume point cloud prefix", type = str)

    parser.add_argument("--comps",         nargs = "*", default = [], help = "Components to be compared", type = str)
    parser.add_argument("--camera",        nargs = "*", default = [], help = "camera position", type = float)
    
    if delayed_parse:
        return parser
    return parser.parse_args()

def load_xyz_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    points = []

    for line in lines:
        coords = line.strip().split()
        if len(coords) >= 3:
            x = float(coords[0])
            y = float(coords[1])
            z = float(coords[2])
            points.append([x, y, z])
    points_np = np.array(points, dtype = np.float64)
    CONSOLE.log(f"pcd '{file_path}', shape: {points_np.shape}")
    return points_np

def make_point_cloud(pcd, color = (0.5, 0.5, 0.5)):
    """ Make open3d point cloud """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcd)
    colors = np.tile(np.float64(color).reshape(1, 3), (pcd.shape[0], 1))
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

def visualize(opts, pcd_list: Dict[str, np.ndarray]):
    """ Deprecated: Visualize point pairs """
    app, vis = make_o3d_app((0, 0, 0, 1), False, False)
    for i, (key, pcd) in enumerate(pcd_list.items()):
        vis.add_geometry(f'{key}', make_point_cloud(pcd, COLORS[i]))
    create_geometry_at_points(vis, [opts.camera], "camera", (0.7, 0.7, 0.7), 0.01)
    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()

if __name__ == "__main__":
    opts = get_options()
    pcd_dict = {}
    for comp_name in opts.comps:
        surf_path = os.path.join(opts.pcd_path, f"{opts.surf_prefix}-{comp_name}.xyz")
        volm_path = os.path.join(opts.pcd_path, f"{opts.volm_prefix}-{comp_name}.xyz")
        pcd_dict[f"{opts.surf_prefix}-{comp_name}"] = load_xyz_file(surf_path)
        pcd_dict[f"{opts.volm_prefix}-{comp_name}"] = load_xyz_file(volm_path)
    visualize(opts, pcd_dict)