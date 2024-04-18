import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui

from rich.console import Console

CONSOLE = Console(width = 128)

__all__ = ['make_o3d_app']

def make_o3d_app(bg_color = (0.3, 0.3, 0.3, 1), show_ground = True, show_axis = True):
    """ Instantiate an open3d application """
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Open3D - Point Cloud Visualizer", 1024, 768)
    vis.set_background(bg_color, None)
    
    vis.show_settings = True
    vis.show_skybox(False)
    vis.show_axes = show_axis
    vis.show_ground = show_ground
    return app, vis

def point_picking(pcd) -> list:
    """ Point cloud point selection """
    CONSOLE.log("")
    CONSOLE.log(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    CONSOLE.log("   Press [shift + right click] to undo point picking")
    CONSOLE.log("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    point_indices = vis.get_picked_points()
    CONSOLE.log("Selected point indices:", point_indices)
    return point_indices

def create_geometry_at_points(vis, points, name: str, color = (1, 0, 0), radius = 0.1):
    geometry = o3d.geometry.TriangleMesh()
    for point in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius = radius) #create a small sphere to represent point
        sphere.translate(point) #translate this sphere to point
        geometry += sphere
    geometry.paint_uniform_color(color)
    vis.add_geometry(name, geometry)