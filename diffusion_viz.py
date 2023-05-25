"""
    Visualizing Diffusion Approximation Result
    As a basis for 2D sampling case study
    
    This is the second experiment verification phase for DA
    if I'm that lucky to pass this phase, the algorithm can be
    implemented in pbrt-v3-transient. Wish me good luck
    @author: Qianyue He
    @date: 2023-5-23
"""
import taichi as ti
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
    def draw_mult(self, vertex_x: float, use_tr: int):
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
            self.pixels[i, j].fill(value)

if __name__ == "__main__":
    """"""
    ti.init(arch = ti.cuda, default_fp = ti.f32, default_ip = ti.i32, device_memory_fraction = 0.4)
    opts = get_options_2d()

    config_dict = {}
    for attr in dir(opts):
        if attr.startswith("_"): continue
        if attr == "config" or attr == "v_pos": continue
        config_dict[attr] = getattr(opts, attr)
    diff_viz = DiffusionViz(config_dict)
