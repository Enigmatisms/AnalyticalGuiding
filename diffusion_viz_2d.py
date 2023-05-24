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

from taichi.math import vec3
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
        self.pixels.fill(0)
        eps = self._emitter_pos[None]
        time = self._time[None]
        us = self.coeffs[0]
        ua = self.coeffs[1]
        pos_scale = self._scale[None]
        val_scale = self._v_scale[None]
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
    def draw_mult(self):
        """ Visualize only the DA * Tr (2D version) 
            In this function, time / max_time have changes of meaning
            time: distance to sample
            max_time: target time

            This visualization does not account for direction, since
            all directions are visualized.
            The vertex pos is at (0, 0), by default
        """
        self.pixels.fill(0)
        eps = self._emitter_pos[None]

        target_time = self._max_time[None]
        d2sample = self._time[None]
        us = self.coeffs[0]
        ua = self.coeffs[1]
        D = get_diffusion_length_ti(ua, us)
        pos_scale = self._scale[None]
        val_scale = self._v_scale[None]
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

            # TODO: we should visualize a circle (for the current distance)
            if value > 0:
                value *= tr_2d(px, py, 0, 0, ua, us)            # transmittance
                value *= val_scale
            self.pixels[i, j].fill(value)


if __name__ == "__main__":
    ti.init(arch = ti.cuda, default_fp = ti.f32, default_ip = ti.i32, device_memory_fraction = 0.4)
    glfw.init()

    opts = get_options_2d()
    config_dict = {}
    for attr in dir(opts):
        if attr.startswith("_"): continue
        if attr == "config": continue
        config_dict[attr] = getattr(opts, attr)

    window   = tui.Window('Scene Interactive Visualizer', res = (1024, 1024), pos = (150, 150))
    canvas   = window.get_canvas()
    gui      = window.get_gui()

    gui.slider_float('Value scale', config_dict['v_scale'], 0.1, 100.)
    gui.slider_float('Max time', config_dict['max_time'], 0.5, 10)
    gui.slider_float('Emitter', config_dict['emitter_pos'], 0.05, 4.0)
    gui.slider_float('Time', config_dict['time'], 0.0, config_dict['max_time'])
    gui.slider_float('Canvas scale', config_dict['scale'], 10., 1000.)
    gui.slider_float('Sigma A', config_dict['ua'], 0., 1.)
    gui.slider_float('Sigma S', config_dict['us'], 5., 200.)
    gui.button('DA only')
    gui.button('Tr DA')

    mode         = config_dict['mode']
    diffuse_mode = 'half' if config_dict['mode'] else 'full'
    gui.text(f"Mode: {mode} | DA: {diffuse_mode}")
    diff_viz = DiffusionViz(config_dict)

    while window.running:
        config_dict['v_scale']     = gui.slider_float('Value scale', config_dict['v_scale'], 0.05, 5.)
        config_dict['max_time']    = gui.slider_float('Max time', config_dict['max_time'], 0.5, 10)
        config_dict['emitter_pos'] = gui.slider_float('Emitter', config_dict['emitter_pos'], 0.05, 5.0)
        config_dict['time']        = gui.slider_float('Time', config_dict['time'], 0.0, config_dict['max_time'])
        config_dict['scale']       = gui.slider_float('Canvas scale', config_dict['scale'], 10., 1000.)
        config_dict['ua']          = gui.slider_float('Sigma A', config_dict['ua'], 0.01, 1.)
        config_dict['us']          = gui.slider_float('Sigma S', config_dict['us'], 5., 200.)
        if gui.button('DA only'):
            mode = 'da_only'
        if gui.button('Tr DA'):
            mode = 'da_tr'
        diff_viz.setter(config_dict)
        if mode == "da_only":
            diff_viz.draw_da()
        else:
            diff_viz.draw_mult()

        if window.is_pressed(tui.ESCAPE): 
            window.running = False
        canvas.set_image(diff_viz.pixels)
        window.show()
        if window.running == False: break
        