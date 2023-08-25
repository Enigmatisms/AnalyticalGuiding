""" Visualize the ellipse calculated by my method
    @author: Qianyue He
    @date: 2023.6.26
"""

import numpy as np
import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt

from functools import partial
from utils.create_plot import PlotTools
from viz_dpg import create_slider, value_updator

def esc_callback(_sender, app_data):
    """ Exit on pressing ESC: ESC in dearpygui is 256 """
    if app_data == 256:          # ESC
        dpg.stop_dearpygui()

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
        configs["cur_angle"] = np.arctan2(direction[1], direction[0])

        distance = get_ellipse_distance(configs["target_time"] * scale, 2. * configs["half_x"] * scale, to_f2, direction)
        oval_pos = f1_pos + direction * distance
        angle = np.arctan2(direction[1], direction[0])
        maxi_1 = max(dpg.get_value("series_tag_0")[1])
        maxi_2 = max([max(dpg.get_value(f"series_tag_{i + 2}")[i]) for i in range(2)])
        dpg.set_value("cursor_0", [[angle, angle], [0, maxi_1]])
        dpg.set_value("cursor_1", [[angle, angle], [0, maxi_2]])
        dpg.set_value("cursor_2", [[angle, angle], [-1, 1]])
        dpg.configure_item("ray1", p1 = f1_pos, p2 = oval_pos)
        dpg.configure_item("ray2", p1 = oval_pos, p2 = f2_pos)

        
def mouse_release_callback(sender, app_data):
    """ Set direction here w.r.t the left focus
        We will record a direction and plot it on the screen
    """
    global configs
    if app_data == 1:
        dpg.configure_item("origin_dir", show = True)
        scale = configs['scale']
        pos_x, pos_y = dpg.get_mouse_pos()

        f1_x = configs["half_w"] - scale * configs["half_x"]
        f2_x = configs["half_w"] + scale * configs["half_x"]
        y_pos = configs["y_pos"]
        f1_pos = np.float32([f1_x, y_pos])
        f2_pos = np.float32([f2_x, y_pos])
        to_f2      = f2_pos - f1_pos
        direction  = np.float32([pos_x, pos_y]) - f1_pos
        direction /= np.linalg.norm(direction)
        to_f2     /= np.linalg.norm(to_f2)
        configs["ori_angle"] = np.arctan2(direction[1], direction[0])
        # print(f"angle = {configs['ori_angle'] * 180 / np.pi}, cur_angle = {configs['cur_angle'] * 180 / np.pi}")
        distance = get_ellipse_distance(configs["target_time"] * scale, 2. * configs["half_x"] * scale, to_f2, direction)
        oval_pos = f1_pos + direction * distance
        dpg.configure_item("origin_dir", p1 = f1_pos, p2 = oval_pos)
        updator_callback()
    
def phase_hg(cos_theta: float, g: float):
    g2 = g * g
    denom = 1. + g2 - 2. * g * cos_theta
    return (1. - g2) / (np.sqrt(denom) * denom) * 0.25 / np.pi

def get_two_cosines(ori_angle: float, sample_angle: float, T: float, d: float):
    """ Get the cosine of the first / second scattering term """
    cos_1 = np.cos(ori_angle - sample_angle)
    cos_a = np.cos(sample_angle)
    x_scatter =  0.5 * (T + d) * (T - d) / (T - cos_a * d)
    cos_2 = (x_scatter - d * cos_a) / (x_scatter - T)
    return cos_1, cos_2

def show_ray_callback():
    global show_ray
    show_ray = not show_ray
    dpg.configure_item("ray1", show = show_ray)
    dpg.configure_item("ray2", show = show_ray)
    
def updator_callback():
    global configs
    scale = configs['scale']
    f1_x = configs["half_w"] - scale * configs["half_x"]
    f2_x = configs["half_w"] + scale * configs["half_x"]
    thetas, cos_1, cos_2, phase_1, phase_2 = evaluate_phase(configs['ori_angle'], 
                configs['target_time'] * scale, f2_x - f1_x, configs['g'])
    phase_product = phase_1 * phase_2
    phase_max = phase_product.max()
    cos_product = cos_1 * cos_2
    cos_product *= phase_max / cos_product.max()

    dpg.set_axis_limits("y_axis_0", 0.0, phase_max * 1.1)
    dpg.set_axis_limits("y_axis_1", 0.0, max(phase_1.max(), phase_2.max()) * 1.1)
    dpg.set_axis_limits("y_axis_2", -1.1, 1.1)
    dpg.set_value("series_tag_0", [thetas, phase_product])
    dpg.set_value("series_tag_1", [thetas, phase_1])
    dpg.set_value("series_tag_2", [thetas, phase_2])
    dpg.set_value("series_tag_3", [thetas, cos_1])
    dpg.set_value("series_tag_4", [thetas, cos_2])

def evaluate_phase(ori_angle: float, T: float, d: float, g: float, num_samples: int = 600):
    delta = 2 * np.pi / num_samples
    thetas = np.linspace(-np.pi + delta, np.pi, num_samples)
    cos_1, cos_2 = get_two_cosines(ori_angle, thetas, T, d)
    phase_1 = phase_hg(cos_1, g)
    phase_2 = phase_hg(cos_2, g)
    return thetas, cos_1, cos_2, phase_1, phase_2
    
def dummy_input():
    xs = np.linspace(-10, 10, 600)
    ys = np.arctan(xs) / np.pi * 2
    return xs, ys

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
        "g"           :0,
        "ori_angle"   :0,
        "cur_angle"   :0,
    }
    skip_params = {"y_pos", "half_w", "ori_angle", "cur_angle"}

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
        dpg.draw_line(p1 = [configs["half_w"] - configs["scale"] * configs["half_x"], configs["y_pos"]],
                      p2 = [configs["half_w"] + configs["scale"] * configs["half_x"], configs["y_pos"]],
                      color = (0, 255, 0), tag = "origin_dir", show = False)
        dpg.draw_line(p1 = [configs["half_w"] + configs["scale"] * configs["half_x"], configs["y_pos"]],
                      p2 = [configs["half_w"] + configs["scale"] * configs["half_x"], configs["y_pos"]],
                      color = (255, 0, 0), tag = "ray2", show = show_ray)
        PlotTools.create_ellipse(configs["half_w"] - configs["scale"] * configs["half_x"], 
                                    configs["half_w"] + configs["scale"] * configs["half_x"], configs["y_pos"],
                                    configs["target_time"] * configs["scale"])
            
    with dpg.window(label="Control panel", tag = "control"):
        other_callbacks = partial(mouse_release_callback, sender = None, app_data = 1)
        create_slider("target time (2a)", "target_time", 1, 100, configs["target_time"], other_callback = other_callbacks)
        create_slider("distance (c)", 'half_x', 0.4, 40, configs["half_x"], other_callback = other_callbacks)
        create_slider("scale", 'scale', 1.0, 10, configs["scale"], other_callback = other_callbacks)
        create_slider("g", 'g', -0.999, 0.999, configs["g"], other_callback = other_callbacks)
        with dpg.group(horizontal = True):
            dpg.add_button(label = 'Show Ray', tag = 'show_ray', width = 100, callback = show_ray_callback)
            dpg.add_button(label = 'Update Curve', tag = 'updator', width = 100, callback = updator_callback)
        
    with dpg.window(label="2D analytical result plots", tag="plots", show = True, pos = (W + 25, 0),
                    no_bring_to_front_on_focus = True, no_focus_on_appearing = True):
        PlotTools.make_plot(540, 250, "Product Curves", ["phase product"], 600, xy_labels = ['angle', 'value'], use_cursor = True)
        PlotTools.make_plot(540, 250, "Phase Curves", ["1st scatter", "2nd scatter"], 600, xy_labels = ['angle', 'value'], use_cursor = True)
        PlotTools.make_plot(540, 250, "Cos Curves", ["1st cos", "2nd cos"], 600, xy_labels = ['angle', 'value'], use_cursor = True)

    with dpg.handler_registry():
        dpg.add_key_release_handler(callback=esc_callback)
        dpg.add_mouse_move_handler(callback=mouse_move_callback)
        dpg.add_mouse_release_handler(callback=mouse_release_callback)

    dpg.create_viewport(title='Ellipse Visualization', width = W + 600, height = H + 200)
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
