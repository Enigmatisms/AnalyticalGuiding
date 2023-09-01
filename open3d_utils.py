import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui

from numpy import ndarray as Arr

def open3d_plot(ell_points: Arr, target: Arr, input_dir: Arr):
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Open3D - Rendering path visualization", 1024, 768)
    vis.set_background((0.3, 0.3, 0.3, 1), None)
    vis.show_settings = True
    vis.show_skybox(False)
    vis.show_axes = True

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ell_points)
    
    tar_pcd = o3d.geometry.PointCloud()
    tar_pcd.points = o3d.utility.Vector3dVector(target[None, :])
    tar_pcd.colors = o3d.utility.Vector3dVector(np.float32([[1., 0., 0.]]))
    vis.add_geometry('ellipse pos', pcd)
    vis.add_geometry('target pos', tar_pcd)
    vis.reset_camera_to_default()
    arrow = get_arrow(np.zeros(3), input_dir, 0.5, 0.1, color = [0, 1, 0])
    vis.add_geometry(f"arrow", arrow)

    app.add_window(vis)
    app.run()
    app.quit()
    
def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array([
                    [np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]
                ])

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([
                    [np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]
                ])
    return Rz, Ry

def get_arrow(origin, end, length, scale=1, color = [1, 0, 0]):
    import open3d as o3d
    assert(not np.all(end == origin))
    vec = (end - origin) * length
    size = np.sqrt(np.sum(vec**2))

    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(cone_radius=size/17.5 * scale,
        cone_height=size*0.2 * scale,
        cylinder_radius=size/30 * scale,
        cylinder_height=size*(1 - 0.2*scale))
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    mesh.paint_uniform_color(color)
    return (mesh)