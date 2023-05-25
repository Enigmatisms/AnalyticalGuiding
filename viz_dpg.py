"""
    Visualizing Diffusion Approximation Result
    As a basis for 2D sampling case study
    
    This is the second experiment verification phase for DA
    if I'm that lucky to pass this phase, the algorithm can be
    implemented in pbrt-v3-transient. Wish me good luck
    @author: Qianyue He
    @date: 2023-5-23
"""

import numpy as np
import taichi as ti
import dearpygui.dearpygui as dpg

from copy import deepcopy
from options import get_options_2d
from diffusion_viz import DiffusionViz

""" TODO:
(1) Add analysis code (for sampling in a specific direction)
    displaying the path length and visualize path
"""

SKIP_PARAMS = {"width", "height", "diffuse_mode", "mode"}

def value_sync(set_tag: str):
    """ Call back function for synchronizing input/slider """
    def value_sync_inner(_sender, app_data, _user_data):
        dpg.set_value(set_tag, app_data)
    return value_sync_inner

def create_slider(label: str, tag: str, min_v: float, max_v: float, default: float):
    """ Create horizontally grouped (and synced) input box and slider """
    with dpg.group(horizontal = True):
        dpg.add_input_float(tag = f"{tag}_input", default_value = default, 
                                width = 110, callback = value_sync(tag))
        dpg.add_slider_float(label = label, tag = tag, min_value = min_v,
                            max_value = max_v, default_value = default, width = 120, callback = value_sync(f"{tag}_input"))

def value_updator(config_dict: dict, skip_params: set):
    """ Get values from sliders """
    for attr in config_dict.keys():
        if attr in skip_params: continue
        config_dict[attr] = dpg.get_value(attr)
    return config_dict

def button_callback(sender):
    """ Setting plot mode via button press """
    global mode
    mode = sender
    if mode == "da_only":
        dpg.set_value("v_scale", -1.5)
        dpg.configure_item("max_time", label = "Max time")
        dpg.configure_item("time", label = "Time")
    else:
        dpg.configure_item("max_time", label = "Target time")
        dpg.configure_item("time", label = "Sampled time")
        dpg.set_value("v_scale", 6)

def esc_callback(sender, app_data):
    """ Exit on pressing ESC """
    if app_data == 27:          # ESC
        dpg.stop_dearpygui()

def key_callback(sender, app_data):
    """ Keyboard responses to be used """
    global vertex_x
    if app_data == 65:
        vertex_x -= 0.02
    elif app_data == 68:
        vertex_x += 0.02

def pad_rgba(img: np.ndarray):
    """ Convert RGB images to RGBA images """
    alpha = np.ones((*img.shape[:-1], 1), dtype = img.dtype)
    img = np.concatenate((img, alpha), axis = -1).transpose((1, 0, 2))
    return np.ascontiguousarray(img)

def dist_sample_callback():
    """ Distance sampling checkbox """
    val = dpg.get_item_configuration("ring")["show"]
    dpg.configure_item("ring", show = not val)

def reset_callback():
    """ Reset configurations to the initial state """
    global config_dict, init_configs, mode
    config_dict.update(init_configs)
    for attr in config_dict.keys():
        if attr in SKIP_PARAMS: continue
        dpg.set_value(attr, init_configs[attr])
        dpg.set_value(f"{attr}_input", init_configs[attr])
    button_callback(config_dict["mode"])
    dpg.set_value("use_tr", True)
    dpg.set_value("use_wvf", True)
    dpg.set_value("dist_sample", True)

if __name__ == "__main__":
    ti.init(arch = ti.cuda, default_fp = ti.f32, default_ip = ti.i32, device_memory_fraction = 0.4)
    opts = get_options_2d()
    vertex_x  = opts.v_pos

    config_dict = {}
    for attr in dir(opts):
        if attr.startswith("_"): continue
        if attr == "config" or attr == "v_pos": continue
        config_dict[attr] = getattr(opts, attr)
    mode = config_dict["mode"]
    init_configs = deepcopy(config_dict)

    diff_viz = DiffusionViz(config_dict)
    dpg.create_context()

    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(width=opts.width, height=opts.height, 
                            default_value=pad_rgba(diff_viz.pixels.to_numpy()), format=dpg.mvFormat_Float_rgba, tag="diffusion")

    with dpg.window(label="Diffusion Display", tag = "display", no_bring_to_front_on_focus = True):
        dpg.add_image("diffusion")
        # TODO: a lot of sliders to be added
        dpg.draw_circle((opts.v_pos * opts.scale, diff_viz.cy), 5, fill = (0, 0, 255, 128), tag=f"vertex")
        dpg.draw_circle((opts.emitter_pos * opts.scale, diff_viz.cy), 5, fill = (255, 0, 0, 128), tag=f"emitter")
        dpg.draw_circle((opts.v_pos * opts.scale, diff_viz.cy), 20, color = (255, 255, 255, 100), tag=f"ring")
        dpg.draw_circle((opts.emitter_pos * opts.scale, diff_viz.cy), 
                        opts.time * opts.scale, color = (255, 255, 255, 80), tag=f"wavefront")
        dpg.draw_circle((0, 0), 20, color = (255, 255, 255, 100), tag=f"test")
        dpg.draw_arrow((opts.v_pos * opts.scale, opts.height - 40), 
                       (opts.emitter_pos * opts.scale, opts.height - 40),
                        tag = "to_emitter", color = (100, 100, 255, 128)
                       )
        dpg.draw_text(((opts.v_pos + opts.emitter_pos) * opts.scale * 0.5 - 32, opts.height - 64), 
                      f"L: {abs(opts.emitter_pos - opts.v_pos):.4f}", tag = "to_emitter_str", size = 20)

    with dpg.handler_registry():
        dpg.add_key_release_handler(callback=esc_callback)
        dpg.add_key_press_handler(callback=key_callback)

    with dpg.window(label="Control panel", tag = "control"):
        create_slider("Value scale", "v_scale", -3, 10, opts.v_scale)
        create_slider("Max time", "max_time", 0.5, 10, opts.max_time)
        create_slider("Time", "time", 0.0, opts.max_time, opts.time)
        create_slider("Emitter position", "emitter_pos", 0.05, 5.0, opts.emitter_pos)
        create_slider("Canvas scale", "scale", 10.0, 1000.0, opts.scale)
        create_slider("Sigma A", "ua", 0.01, 1.0, opts.ua)
        create_slider("Sigma S", "us", 5.0, 200.0, opts.us)
        with dpg.group(horizontal = True):
            dpg.add_checkbox(label = 'Distance sampling', tag = 'dist_sample', 
                             default_value = True, callback = dist_sample_callback)
            dpg.add_checkbox(label = 'Use Tr', tag = 'use_tr', default_value = True)
            dpg.add_checkbox(label = 'Wavefront', tag = 'use_wvf', default_value = True)
        with dpg.group(horizontal = True):
            dpg.add_button(label = 'DA only', tag = 'da_only', width = 100, callback = button_callback)
            dpg.add_button(label = 'DA Tr', tag = 'da_tr', width = 100, callback = button_callback)
            dpg.add_button(label = 'Reset', tag = 'reset', width = 100, callback = reset_callback)

    dpg.create_viewport(title='Analytical Guiding 2D visualization', width=opts.width + 50, height=opts.height + 100)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    
    mode         = config_dict['mode']
    diffuse_mode = 'half' if config_dict['mode'] else 'full'

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        diff_viz.pixels.fill(0)
        config_dict = value_updator(config_dict, SKIP_PARAMS)
        diff_viz.setter(config_dict)
        use_tr    = dpg.get_value("use_tr")

        dpg.configure_item("emitter", center = (config_dict["emitter_pos"] * config_dict["scale"], diff_viz.cy))
        dpg.configure_item("vertex", center = (vertex_x * config_dict["scale"], diff_viz.cy))
        dpg.configure_item("ring", center = (vertex_x * config_dict["scale"], diff_viz.cy),
                radius = config_dict["scale"] * config_dict["time"])
        dpg.configure_item("wavefront", center = (config_dict["emitter_pos"] * config_dict["scale"], diff_viz.cy),
                radius = config_dict["scale"] * config_dict["time"])
        dpg.configure_item("to_emitter", 
                p1 = (vertex_x * config_dict["scale"], opts.height - 40),
                p2 = (config_dict["emitter_pos"] * config_dict["scale"], opts.height - 40))
        text_x_pos = (config_dict["emitter_pos"] + vertex_x) * config_dict["scale"] * 0.5 - 32
        dpg.configure_item("to_emitter_str", 
                pos = (text_x_pos, opts.height - 64),
                text = f"L: {abs(config_dict['emitter_pos'] - vertex_x):.4f}"
                )
        dpg.configure_item("time", max_value = config_dict['max_time'])

        if mode == "da_only":
            diff_viz.draw_da()
        else:
            # convert from window coordinates to world coordinates
            # TODO: refactoring, we can draw circles outside of the kernel function
            diff_viz.draw_mult(vertex_x, use_tr)
        raw_data = pad_rgba(diff_viz.pixels.to_numpy())
        dpg.configure_item("diffusion", default_value=raw_data)
    dpg.destroy_context()
