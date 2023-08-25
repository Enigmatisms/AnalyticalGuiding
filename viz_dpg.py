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
from taichi.math import vec2
import dearpygui.dearpygui as dpg

from copy import deepcopy
from options import get_options_2d
from diffusion_viz import DiffusionViz
from utils.create_plot import PlotTools
from utils.ctrl_utils import ControlInfo
from utils.analysis_2d import AnalysisTool

SLIDER_WIDTH = 120
BUTTON_WIDTH = 120
SKIP_PARAMS = {"width", "height", "diffuse_mode", "mode"}

def value_sync(set_tag: str):
    """ Call back function for synchronizing input/slider """
    def value_sync_inner(_sender, app_data, _user_data):
        dpg.set_value(set_tag, app_data)
    return value_sync_inner

def create_slider(label: str, tag: str, min_v: float, max_v: float, default: float, in_type: str = "float"):
    """ Create horizontally grouped (and synced) input box and slider """
    slider_func = dpg.add_slider_float if in_type == "float" else dpg.add_slider_int
    input_func  = dpg.add_input_float if in_type == "float" else dpg.add_input_int
    with dpg.group(horizontal = True):
        input_func(tag = f"{tag}_input", default_value = default, 
                                width = 110, callback = value_sync(tag))
        slider_func(label = label, tag = tag, min_value = min_v,
                    max_value = max_v, default_value = default, width = SLIDER_WIDTH, callback = value_sync(f"{tag}_input"))

def value_updator(config_dict: dict, skip_params: set):
    """ Get values from sliders """
    for attr in config_dict.keys():
        if attr in skip_params: continue
        config_dict[attr] = dpg.get_value(attr)
    return config_dict

def radio_callback(sender, app_data):
    """ Setting plot mode via button press """
    global mode
    if app_data == "DA only":
        mode = "da_only"
        dpg.set_value("v_scale", -1.5)
        dpg.set_value("v_scale_input", -1.5)
        dpg.configure_item('max_time', label = "Max time")
        dpg.configure_item('time', label = "Time")
    else:
        mode = "da_tr"
        dpg.configure_item('max_time', label = "Target time")
        dpg.configure_item('time', label = "Sampled time")
        dpg.set_value("v_scale", 3)
        dpg.set_value("v_scale_input", 3)

def esc_callback(sender, app_data):
    """ Exit on pressing ESC: ESC in dearpygui is 256 """
    if app_data == 256:          # ESC
        dpg.stop_dearpygui()

def key_callback(sender, app_data):
    """ Keyboard responses to be used """
    global ctrl
    if app_data == 65:
        ctrl.vertex_x -= 0.02
    elif app_data == 68:
        ctrl.vertex_x += 0.02

def pad_rgba(img: np.ndarray):
    """ Convert RGB images to RGBA images """
    alpha = np.ones((*img.shape[:-1], 1), dtype = img.dtype)
    img = np.concatenate((img, alpha), axis = -1).transpose((1, 0, 2))
    return np.ascontiguousarray(img)

def reset_callback():
    """ Reset configurations to the initial state """
    global config_dict, init_configs, mode, ctrl, opts
    config_dict.update(init_configs)
    for attr in config_dict.keys():
        if attr in SKIP_PARAMS: continue
        dpg.set_value(attr, init_configs[attr])
        dpg.set_value(f"{attr}_input", init_configs[attr])
    ctrl.reset(opts.v_pos)
    mode_name = "DA only" if config_dict["mode"] == "da_only" else "DA Tr"
    radio_callback(None, mode_name)
    dpg.set_value("use_tr", True)
    dpg.set_value("use_wvf", True)
    dpg.set_value("dist_sample", True)

def pdf_draw_callback():
    global ctrl
    ctrl.calculate_pdf = ctrl.dir_selected

def sampling_callback():
    global ctrl
    ctrl.calculate_sample = ctrl.dir_selected

def show_status_callback(tags: list, labels: list = None):
    def closure(sender):
        val = dpg.get_item_configuration(tags[0])["show"]       # unified status
        for tag in tags:
            dpg.configure_item(tag, show = not val)
        if labels is not None:      # switch labels
            dpg.configure_item(sender, label = labels[int(val)])
    return closure

def mouse_release_callback(sender, app_data):
    global ctrl, config_dict
    if app_data == 1:       # Freeze scale
        scale = config_dict['scale']
        ctrl.pos_x, ctrl.pos_y = dpg.get_mouse_pos()
        ctrl.pos_x /= scale
        ctrl.pos_y /= scale
        ctrl.dir_selected = True

if __name__ == "__main__":
    ti.init(arch = ti.cuda, default_fp = ti.f32, default_ip = ti.i32, device_memory_fraction = 0.4)
    opts = get_options_2d()
    ctrl = ControlInfo(opts.v_pos)

    config_dict = {}
    for attr in dir(opts):
        if attr.startswith("_"): continue
        if attr == "config" or attr == "v_pos": continue
        config_dict[attr] = getattr(opts, attr)
    mode = config_dict["mode"]
    init_configs = deepcopy(config_dict)

    diff_viz = DiffusionViz(config_dict)
    analyzer = AnalysisTool(sol = 1, diff_mode = opts.diffuse_mode, max_ray_march_num = opts.rm_num, sample_num = opts.samp_num)
    dpg.create_context()

    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(width=opts.width, height=opts.height, 
                            default_value=pad_rgba(diff_viz.pixels.to_numpy()), format=dpg.mvFormat_Float_rgba, tag="diffusion")

    with dpg.window(label="Diffusion Display", tag = "display", no_bring_to_front_on_focus = True):
        dpg.add_image("diffusion")
        # Vertex position
        dpg.draw_circle((opts.v_pos * opts.scale, diff_viz.cy), 5, fill = (0, 0, 255, 128), tag=f"vertex")
        # Emitter position
        dpg.draw_circle((opts.emitter_pos * opts.scale, diff_viz.cy), 5, fill = (255, 0, 0, 128), tag=f"emitter")
        # Sampling 'ring'
        dpg.draw_circle((opts.v_pos * opts.scale, diff_viz.cy), 20, color = (255, 255, 255, 100), tag=f"ring")
        # Emission wavefront
        dpg.draw_circle((opts.emitter_pos * opts.scale, diff_viz.cy), 
                        opts.time * opts.scale, color = (255, 255, 255, 80), tag=f"wavefront")
        # Sampling direction
        dpg.draw_line((0, 0), 20, color = (255, 255, 255, 80), tag=f"sample_dir", show = False)
        # Length indicating arrow
        # dpg.draw_ellipse(, )
        dpg.draw_arrow((opts.v_pos * opts.scale, opts.height - 40), 
                       (opts.emitter_pos * opts.scale, opts.height - 40),
                        tag = "to_emitter", color = (100, 100, 255, 128)
                       )
        # Length annotation
        dpg.draw_text(((opts.v_pos + opts.emitter_pos) * opts.scale * 0.5 - 32, opts.height - 64), 
                      f"L: {abs(opts.emitter_pos - opts.v_pos):.4f}", tag = "to_emitter_str", size = 20)
        PlotTools.create_ellipse(opts.v_pos * opts.scale, opts.emitter_pos * opts.scale, diff_viz.cy,
                        opts.max_time * opts.scale)
        PlotTools.create_peak_plots(opts.v_pos, opts.emitter_pos, diff_viz.cy, opts.scale)

    with dpg.handler_registry():
        dpg.add_key_release_handler(callback=esc_callback)
        dpg.add_key_press_handler(callback=key_callback)
        dpg.add_mouse_release_handler(callback=mouse_release_callback)

    with dpg.window(label="Control panel", tag = "control"):
        create_slider("Value scale", "v_scale", -3, 10, opts.v_scale)
        create_slider("Max time", 'max_time', 0.5, 10, opts.max_time)
        create_slider("Time", 'time', 0.0, opts.max_time, opts.time)
        create_slider("Emitter position", 'emitter_pos', 0.05, 5.0, opts.emitter_pos)
        create_slider("Canvas scale", 'scale', 10.0, 1000.0, opts.scale)
        create_slider("Sigma A", "ua", 0.01, 1.0, opts.ua)
        create_slider("Sigma S", "us", 1.0, 200.0, opts.us)
        create_slider("Ray marching", "rm_num", 32, 256, opts.rm_num, "int")
        create_slider("Sample num", "samp_num", 64, 4096, opts.samp_num, "int")
        create_slider("Bin num", "bin_num", 10, 100, opts.bin_num, "int")

        with dpg.group(horizontal = True):
            dpg.add_checkbox(label = 'Distance sampling', tag = 'dist_sample', 
                             default_value = True, callback = show_status_callback(["ring"]))
            dpg.add_checkbox(label = 'Use Tr', tag = 'use_tr', default_value = True)
            dpg.add_checkbox(label = 'Wavefront', tag = 'use_wvf', default_value = True)
        with dpg.group(horizontal = True):
            default_ratio_v = "DA only" if mode == "da_only" else "DA Tr"
            dpg.add_radio_button(["DA only", "DA Tr"], horizontal = True, 
                        default_value = default_ratio_v, callback = radio_callback, tag = "mode_select")
            dpg.add_radio_button(["remaining", "forward"], horizontal = True, 
                        default_value = "forward", tag = "time_select")
        with dpg.group(horizontal = True):
            dpg.add_button(label = 'Reset', tag = 'reset', width = BUTTON_WIDTH, callback = reset_callback)
            dpg.add_button(label = 'Plot show', tag = 'plot_show', width = BUTTON_WIDTH, 
                           callback = show_status_callback(["plots"]))
        with dpg.group(horizontal = True):
            dpg.add_button(label = 'Draw PDF', tag = 'draw_pdf', width = BUTTON_WIDTH, callback = pdf_draw_callback)
            dpg.add_button(label = 'Draw sample', tag = 'draw_sample', width = BUTTON_WIDTH, 
                           callback = sampling_callback)
        with dpg.group(horizontal = True):
            peak_related_tags = ["peak_line1", "peak_point", "peak_line2", "peak_text1", "peak_text2"]
            dpg.add_button(label = 'Show Ellipse', tag = 'ellipse', width = BUTTON_WIDTH, 
                           callback = show_status_callback(["time_ellipse"], ['No ellipse', 'Show ellipse']))
            dpg.add_button(label = 'Show peaks', tag = 'peak', width = BUTTON_WIDTH, 
                           callback = show_status_callback(peak_related_tags, ['No peaks', 'Show peaks']))
        if mode == "da_only":
            dpg.configure_item('max_time', label = "Max time")
            dpg.configure_item('time', label = "Time")
        else:
            dpg.configure_item('max_time', label = "Target time")
            dpg.configure_item('time', label = "Sampled time")

    with dpg.window(label="2D analytical result plots", tag="plots", show = False, pos = (opts.width + 50, 0),
                    no_bring_to_front_on_focus = True, no_focus_on_appearing = True):
        PlotTools.make_plot(540, 270, "Sampling PDF plot", ["Transmittance", "DA (* Tr)"], opts.rm_num)
        PlotTools.make_plot(540, 270, "Sampled distances", ["Tr sample", "RTS"], [opts.bin_num, opts.bin_num], mode = "bar")

    dpg.create_viewport(title='Analytical Guiding 2D visualization', width=opts.width + 50, height=opts.height + 100)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    
    mode         = config_dict['mode']
    diffuse_mode = 'half' if config_dict['mode'] else 'full'

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        diff_viz.pixels.fill(0)
        ctrl.use_tr      = dpg.get_value("use_tr")
        config_dict = value_updator(config_dict, SKIP_PARAMS)
        
        emitter_pos = config_dict['emitter_pos']
        set_scale   = config_dict['scale']
        max_time    = config_dict['max_time']
        cur_time    = config_dict['time']

        PlotTools.toggle_ellipse(ctrl.vertex_x * set_scale, emitter_pos * set_scale, diff_viz.cy, set_scale * max_time)
        dpg.configure_item("emitter", center = (emitter_pos * set_scale, diff_viz.cy))
        dpg.configure_item("vertex", center = (ctrl.vertex_x * set_scale, diff_viz.cy))
        
        dpg.configure_item("to_emitter", 
                p1 = (ctrl.vertex_x * set_scale, opts.height - 40),
                p2 = (emitter_pos * set_scale, opts.height - 40))
        text_x_pos = (emitter_pos + ctrl.vertex_x) * set_scale * 0.5 - 32
        dpg.configure_item("to_emitter_str", 
                pos = (text_x_pos, opts.height - 64),
                text = f"L: {abs(emitter_pos - ctrl.vertex_x):.4f}"
                )
        dpg.configure_item('time', max_value = max_time)
        if dpg.get_value('time') > max_time:
            dpg.set_value('time', max_time - 1e-3)
            dpg.set_value('time_input', max_time - 1e-3)
            cur_time = max_time - 1e-3
            config_dict['time'] = max_time - 1e-3
        dpg.configure_item('time', max_value = max_time - 1e-3)
        dpg.configure_item("ring", center = (ctrl.vertex_x * set_scale, diff_viz.cy),
                radius = set_scale * cur_time)
        if mode == 'da_only':
            dpg.configure_item("wavefront", center = (emitter_pos * set_scale, diff_viz.cy),
                    radius = set_scale * cur_time)
        else:
            dpg.configure_item("wavefront", center = (emitter_pos * set_scale, diff_viz.cy),
                    radius = set_scale * (max_time - cur_time))
        if ctrl.dir_selected:
            dpg.configure_item("sample_dir", 
                p1 = (ctrl.vertex_x * set_scale, diff_viz.cy), 
                p2 = (ctrl.pos_x * set_scale, ctrl.pos_y * set_scale), show = True)

        diff_viz.setter(config_dict)
        analyzer.setter(config_dict)

        if ctrl.calculate_pdf:
            # This logic is for button "Draw PDF"
            ctrl.calculate_pdf = False
            
            scale = diff_viz.scale
            eps_y = diff_viz.cy / scale
            target = vec2([ctrl.pos_x, ctrl.pos_y])
            target_time = -1 if dpg.get_value("time_select") == "forward" else max_time
            ctrl.length = analyzer.ray_marching(ctrl.vertex_x, eps_y, emitter_pos, eps_y, 
                                  config_dict["ua"], config_dict["us"], target, ctrl.use_tr, target_time)
            xs = np.linspace(0, ctrl.length, analyzer.max_ray_march_num + 1)[1:]
            transmittance  = analyzer.transmittance
            transmittance /= transmittance.sum()
            da_solution    = analyzer.da_solution
            da_solution   /= da_solution.sum()
            max_value = max(transmittance.max(), da_solution.max()) * 1.1
            dpg.set_axis_limits("y_axis_0", 0.0, max_value)
            dpg.set_value("series_tag_0", [xs, transmittance])
            dpg.set_value("series_tag_1", [xs, da_solution / da_solution.sum()])

            peak_x, peak_y = AnalysisTool.get_peaks(xs, da_solution, ctrl.vertex_x,
                                    diff_viz.cy / set_scale, ctrl.pos_x, ctrl.pos_y)
            PlotTools.toggle_peak_plots(peak_x, peak_y, ctrl.vertex_x, emitter_pos, diff_viz.cy, set_scale)
            
            analyzer.ray_marched = True
        if ctrl.calculate_sample:
            # This logic is for button "Draw samples"
            ctrl.calculate_sample = False
            if analyzer.ray_marched:
                da_samples, tr_samples = analyzer.inverse_sampling(ctrl.length,
                                  config_dict["ua"], config_dict["us"])
                bin_tr, hist_tr = AnalysisTool.get_histograms(tr_samples, config_dict['bin_num'], ctrl.length)
                bin_da, hist_da = AnalysisTool.get_histograms(da_samples, config_dict['bin_num'], ctrl.length)
                dpg.configure_item("series_tag_2", weight = bin_tr[1] - bin_tr[0])
                dpg.configure_item("series_tag_3", weight = bin_da[1] - bin_da[0])
                dpg.set_value("series_tag_2", [bin_tr, hist_tr])
                dpg.set_value("series_tag_3", [bin_da, hist_da])

        if mode == "da_only":
            diff_viz.draw_da()
        else:
            diff_viz.draw_mult(ctrl.vertex_x, ctrl.use_tr)
        raw_data = pad_rgba(diff_viz.pixels.to_numpy())
        dpg.configure_item("diffusion", default_value=raw_data)
    dpg.destroy_context()
