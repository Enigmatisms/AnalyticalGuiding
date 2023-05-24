"""
    Visualizing Diffusion Approximation Result
    As a basis for 2D sampling case study
    
    This is the second experiment verification phase for DA
    if I'm that lucky to pass this phase, the algorithm can be
    implemented in pbrt-v3-transient. Wish me good luck
    @author: Qianyue He
    @date: 2023-5-23
"""

import glfw
import numpy as np
import taichi as ti
import taichi.ui as tui
import taichi.math as tm

from taichi.math import vec2
from options import get_options_2d
from diffusion_viz_base import DiffusionVizBase
from da import (half_diffusion_2d, full_diffusion_2d, 
        tr_2d, get_diffusion_length_ti)

@ti.data_oriented
class DiffusionViz(DiffusionVizBase):
    """ Note that this visualization should serve more than visualizing 
        2D temporal responses of a point source in scattering media, I want
        to extend it for further use (that would prove me correct), for example
        Visualizing a ring area, which displays [Tr(x) * DA(x, t - d / c)]. Toggle 
        the bar to modify the visualizing time
    """
    def __init__(self, prop: dict) -> None:
        super().__init__(prop)
        self.cy = self.h // 2

        # pixels coordinates centers at (0, 0)
        self.pixels: ti.Field = ti.Vector.field(3, float, shape = (self.w, self.h))

    @ti.kernel
    def draw_da(self):
        """ Visualize only the Diffusion Approximation (2D version) """
        eps = self._emitter_pos[None]
        time = self._time[None]
        us = self.coeffs[0]
        ua = self.coeffs[1]
        pos_scale = self._scale[None]
        val_scale = ti.exp(self._v_scale[None])
        D = get_diffusion_length_ti(ua, us)
        eps_y = self.cy / pos_scale
        for i, j in self.pixels:
            px = float(i) / pos_scale
            py = float(j) / pos_scale
            dist = ti.sqrt((px - eps) ** 2 + (py - eps_y) ** 2)
            value = 0.
            if dist < time:
                if ti.static(self.diffuse_mode == DiffusionViz.FULL_DIFFUSE):
                    value = full_diffusion_2d(px, py, time, 0, eps, eps_y, ua, D, 1)
                else:
                    value = half_diffusion_2d(px, py, time, 0, eps, eps_y, ua, D, 1)
                value *= val_scale
            self.pixels[i, j].fill(value)

    @ti.kernel
    def draw_mult(self, vertex_x: float, sample_time: float, show_dist_samp: int, use_tr: int):
        """ Visualize only the DA * Tr (2D version) 
            In this function, time / max_time have changes of meaning
            time: distance to sample
            max_time: target time

            This visualization does not account for direction, since
            all directions are visualized.
            The vertex pos is at (0, 0), by default
        """
        eps = self._emitter_pos[None]

        target_time = self._max_time[None]
        d2sample = self._time[None]
        us = self.coeffs[0]
        ua = self.coeffs[1]
        D = get_diffusion_length_ti(ua, us)
        pos_scale = self._scale[None]
        val_scale = ti.exp(self._v_scale[None])
        eps_y = self.cy / pos_scale
        for i, j in self.pixels:
            px = float(i) / pos_scale
            py = float(j) / pos_scale
            dist = ti.sqrt((px - eps) ** 2 + (py - eps_y) ** 2)       # distance to the emitter
            res_time = target_time - d2sample
            value = 0.
            if dist < res_time:
                if ti.static(self.diffuse_mode == DiffusionViz.FULL_DIFFUSE):
                    value = full_diffusion_2d(px, py, res_time, 0, eps, eps_y, ua, D, 1)
                else:
                    value = half_diffusion_2d(px, py, res_time, 0, eps, eps_y, ua, D, 1)
            if value > 0:
                if use_tr:
                    value *= tr_2d(px, py, vertex_x, eps_y, ua, us)            # transmittance
                value *= val_scale
            if show_dist_samp:
                to_vertex = ti.sqrt((px - vertex_x) ** 2 + (py - eps_y) ** 2)
                if to_vertex < sample_time:
                    value += 0.05
            self.pixels[i, j].fill(value)

def get_grid_field(h_field: ti.template, v_field: ti.template):
    for i in range(1, 20):
        base = (i << 1) - 2
        h_field[base]     = vec2(0, i / 20)
        h_field[base + 1] = vec2(1, i / 20)
        v_field[base]     = vec2(i / 20, 0)
        v_field[base + 1] = vec2(i / 20, 1)

def get_scale_lines(scale_field: ti.template, eps, width, scale):
    x_pos = eps * scale / width
    scale_field[0] = vec2(0, 0.02)
    scale_field[1] = vec2(x_pos, 0.02)
    scale_field[2] = vec2(x_pos, 0.02)
    scale_field[3] = vec2(x_pos, 0.04)

def window_to_world(window_coord: float, win_w: float, scale: float):
    return window_coord * win_w / scale

def world_to_window(world_coord: float, win_w: float, scale: float):
    return world_coord * scale / win_w

if __name__ == "__main__":
    glfw.init()
    ti.init(arch = ti.cuda, default_fp = ti.f32, default_ip = ti.i32, device_memory_fraction = 0.4)
    opts = get_options_2d()

    h_field     = ti.Vector.field(2, float, 38)        # 99 lines, 198 vertices
    v_field     = ti.Vector.field(2, float, 38)        # 99 lines, 198 vertices
    scale_field = ti.Vector.field(2, float, 6)
    vertex_pos  = ti.Vector.field(2, float, shape = 1)
    emitter_pos = ti.Vector.field(2, float, shape = 1)
    vertex_pos[0]  = vec2(opts.v_pos * opts.scale / opts.width, 0.5)
    emitter_pos[0] = vec2(opts.emitter_pos * opts.scale / opts.width, 0.5)
    get_grid_field(h_field, v_field)

    config_dict = {}
    for attr in dir(opts):
        if attr.startswith("_"): continue
        if attr == "config" or attr == "v_pos": continue
        config_dict[attr] = getattr(opts, attr)

    window   = tui.Window(name = 'Scene Interactive Visualizer', res = (900, 900), fps_limit = 120, pos = (150, 150))
    canvas   = window.get_canvas()
    gui      = window.get_gui()

    gui.slider_float('Value scale', config_dict['v_scale'], -3, 10.)
    gui.slider_float('Max time', config_dict['max_time'], 0.5, 10)
    gui.slider_float('Emitter', config_dict['emitter_pos'], 0.05, 4.0)
    gui.slider_float('Time', config_dict['time'], 0.0, config_dict['max_time'])
    gui.slider_float('Canvas scale', config_dict['scale'], 10., 1000.)
    gui.slider_float('Sigma A', config_dict['ua'], 0., 1.)
    gui.slider_float('Sigma S', config_dict['us'], 5., 200.)
    gui.button('DA only')
    gui.button('Tr DA')
    show_grid  = gui.checkbox('Show grid', False)
    show_scale = gui.checkbox('Show scale', True)
    show_dist  = gui.checkbox('Distance sampling', False)
    use_tr     = gui.checkbox('Use Tr', True)

    mode         = config_dict['mode']
    diffuse_mode = 'half' if config_dict['mode'] else 'full'
    gui.text(f"Mode: {mode} | DA: {diffuse_mode}")
    gui.text(f"Scale: {config_dict['width'] * 0.1 / config_dict['scale']:.5f}")
    diff_viz = DiffusionViz(config_dict)

    while window.running:
        diff_viz.pixels.fill(0)
        config_dict['v_scale']     = gui.slider_float('Value scale', config_dict['v_scale'], -3, 10)
        config_dict['max_time']    = gui.slider_float('Max time', config_dict['max_time'], 0.5, 10)
        config_dict['emitter_pos'] = gui.slider_float('Emitter', config_dict['emitter_pos'], 0.05, 5.0)
        config_dict['time']        = gui.slider_float('Time', config_dict['time'], 0.0, config_dict['max_time'])
        config_dict['scale']       = gui.slider_float('Canvas scale', config_dict['scale'], 10., 1000.)
        config_dict['ua']          = gui.slider_float('Sigma A', config_dict['ua'], 0.01, 1.)
        config_dict['us']          = gui.slider_float('Sigma S', config_dict['us'], 5., 200.)
        show_grid  = gui.checkbox('Show grid', show_grid)
        show_scale = gui.checkbox('Show scale', show_scale)
        show_dist  = gui.checkbox('Distance sampling', show_dist)
        use_tr  = gui.checkbox('Use Tr', use_tr)
        if gui.button('DA only'):
            mode = 'da_only'
        if gui.button('Tr DA'):
            mode = 'da_tr'
        gui.text(f"Mode: {mode} | DA: {diffuse_mode}")
        gui.text(f"Scale: {config_dict['emitter_pos']:.5f}")

        # update configuration dict 
        diff_viz.setter(config_dict)
        # update current emitter pos
        emitter_pos[0][0] = world_to_window(config_dict['emitter_pos'], opts.width, config_dict['scale']) 
        if window.is_pressed(tui.ESCAPE): 
            window.running = False
        elif window.is_pressed("a"): vertex_pos[0][0] -= 0.005
        elif window.is_pressed("d"): vertex_pos[0][0] += 0.005


        if mode == "da_only":
            diff_viz.draw_da()
        else:
            # convert from window coordinates to world coordinates
            vertex_world_x = window_to_world(vertex_pos[0][0], opts.width, config_dict['scale'])
            diff_viz.draw_mult(vertex_world_x, config_dict['time'], show_dist, use_tr)

        canvas.set_image(diff_viz.pixels)
        canvas.circles(vertex_pos, 0.01, color = (0.9, 0.0, 0.0))
        canvas.circles(emitter_pos, 0.005, color = (0.0, 0.0, 0.9))

        if show_grid:
            canvas.lines(h_field, width = 0.002, color = (0.3, 0.3, 0.3))
            canvas.lines(v_field, width = 0.002, color = (0.3, 0.3, 0.3))
        if show_scale:
            get_scale_lines(scale_field, config_dict['emitter_pos'], opts.width, config_dict['scale'])
            canvas.lines(scale_field, width = 0.004, color = (0.0, 0.0, 0.9))
            
        window.show()
        if window.running == False: break
