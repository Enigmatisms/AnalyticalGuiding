""" Visualize the ellipse calculated by my method
    @author: Qianyue He
    @date: 2023.6.26
"""

import numpy as np
import dearpygui.dearpygui as dpg

from utils.create_plot import PlotTools
from viz_dpg import create_slider, value_updator


def get_ellipse_distance(T: float, d: float, to_f2: np.ndarray, direction: np.ndarray):
    cos = np.dot(to_f2, direction)
    return 0.5 * (T + d) * (T - d) / (T - cos * d)

def mouse_move_callback():
    global configs
    if show_ray:
        scale = configs['scale']
        pos_x, pos_y = dpg.get_mouse_pos()

        f1_x = configs["half_w"] - scale * configs["half_x"]
        f2_x = configs["half_w"] + scale * configs["half_x"]
        y_pos = configs["y_pos"]

        f1_pos = np.float32([f1_x, y_pos])
        f2_pos = np.float32([f2_x, y_pos])

        to_f2      = f2_pos - f1_pos
        direction  = np.float32([pos_x, pos_y]) - f1_pos
        to_f2     /= np.linalg.norm(to_f2)
        direction /= np.linalg.norm(direction)

        distance = get_ellipse_distance(configs["target_time"] * scale, 2. * configs["half_x"] * scale, to_f2, direction)
        oval_pos = f1_pos + direction * distance
        dpg.configure_item("ray1", p1 = f1_pos, p2 = oval_pos)
        dpg.configure_item("ray2", p1 = oval_pos, p2 = f2_pos)

def show_ray_callback():
    global show_ray
    show_ray = not show_ray
    dpg.configure_item("ray1", show = show_ray)
    dpg.configure_item("ray2", show = show_ray)

W = 800
H = 600
show_ray = False

if __name__ == "__main__":
    configs = {
        "y_pos"       :H / 2,
        "half_w"      :W / 2,
        "half_x"      :20,
        "scale"       :8.0,
        "target_time" :60,
    }
    skip_params = {"y_pos", "half_w"}

    dpg.create_context()

    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(width=W, height=H, 
                            default_value=np.zeros((W, H, 4), dtype = np.float32), format=dpg.mvFormat_Float_rgba, tag="ellipse")

    with dpg.window(label="Ellipse viz", tag = "display", no_bring_to_front_on_focus = True):
        dpg.add_image("ellipse")
        dpg.draw_circle(center = [configs["half_w"] - configs["scale"] * configs["half_x"], configs["y_pos"]], 
                        radius = 5, fill = (255, 255, 255), tag = "f1", show = True)
        dpg.draw_circle(center = [configs["half_w"] + configs["scale"] * configs["half_x"], configs["y_pos"]], 
                        radius = 5, fill = (255, 255, 255), tag = "f2", show = True)
        dpg.draw_line(p1 = [configs["half_w"] - configs["scale"] * configs["half_x"], configs["y_pos"]],
                      p2 = [configs["half_w"] + configs["scale"] * configs["half_x"], configs["y_pos"]],
                      color = (255, 0, 0), tag = "ray1", show = show_ray)
        dpg.draw_line(p1 = [configs["half_w"] + configs["scale"] * configs["half_x"], configs["y_pos"]],
                      p2 = [configs["half_w"] + configs["scale"] * configs["half_x"], configs["y_pos"]],
                      color = (255, 0, 0), tag = "ray2", show = show_ray)
        PlotTools.create_ellipse(configs["half_w"] - configs["scale"] * configs["half_x"], 
                                    configs["half_w"] + configs["scale"] * configs["half_x"], configs["y_pos"],
                                    configs["target_time"] * configs["scale"])
            
    with dpg.window(label="Control panel", tag = "control"):
        create_slider("target time (2a)", "target_time", 1, 100, configs["target_time"])
        create_slider("distance (c)", 'half_x', 0.4, 40, configs["half_x"])
        create_slider("scale", 'scale', 1.0, 10, configs["scale"])
        dpg.add_button(label = 'Show Ray', tag = 'show_ray', width = 100, callback = show_ray_callback)

    with dpg.handler_registry():
        dpg.add_mouse_move_handler(callback=mouse_move_callback)

    dpg.create_viewport(title='Ellipse Visualization', width = W + 50, height = H + 100)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.configure_item("time_ellipse", show = True)

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        configs = value_updator(configs, skip_params)
        configs["half_x"] = min(configs["half_x"], configs["target_time"] / 2 - 0.1)
        PlotTools.toggle_ellipse(
            configs["half_w"] - configs["scale"] * configs["half_x"], 
            configs["half_w"] + configs["scale"] * configs["half_x"], configs["y_pos"],
            configs["target_time"] * configs["scale"]
        )
        dpg.set_value("half_x", configs["half_x"])
        dpg.configure_item("half_x", max_value = configs["target_time"] / 2 - 0.1)
        dpg.configure_item("f1", center = [configs["half_w"] - configs["scale"] * configs["half_x"], configs["y_pos"]])
        dpg.configure_item("f2", center = [configs["half_w"] + configs["scale"] * configs["half_x"], configs["y_pos"]])
        dpg.configure_item("ray1", p1 = [configs["half_w"] - configs["scale"] * configs["half_x"], configs["y_pos"]])
        dpg.configure_item("ray2", p2 = [configs["half_w"] + configs["scale"] * configs["half_x"], configs["y_pos"]])

    dpg.destroy_context()